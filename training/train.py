"""PPO training for the Hunter / Evader roles.

modes
  single       one PPO learner vs a fixed opponent (scripted spec, or ppo:<model.pt> frozen)
  independent  both roles learn at the same time from the same games
  selfplay     each role learns against a pool of frozen snapshots of the other role;
               half the envs train the Hunter, half train the Evader

--steps is the number of samples each learner collects, so all modes give every learner
the same amount of data (selfplay therefore plays 2x the env steps of independent).
"""
import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from agents.factory import make_agent
from agents.learned import LearnedAgent
from env.dungeon_env import EVADER, HUNTER, EnvConfig
from env.generator import SEED_SPLITS
from evaluation.evaluate import evaluate
from rl.curriculum import LEVELS, Curriculum, level_cfg, ood_cfg
from rl.nets import save_checkpoint
from rl.pool import FrozenPolicy, OpponentPool
from rl.ppo import PPO, PPOConfig, RolloutBuffer
from rl.vec_env import ROLES, make_vec_env

WIN = {HUNTER: "caught", EVADER: "escaped"}
REF_OPP = {HUNTER: "path", EVADER: "chase"}
OBS_DIM, N_ACTIONS = 496, 5


def default_opponent(role):
    return REF_OPP[role]


class CsvLog:
    def __init__(self, path, header):
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(header)

    def row(self, *vals):
        self.w.writerow([round(v, 5) if isinstance(v, float) else v for v in vals])

    def close(self):
        self.f.close()


def reference_evals(agents, n_maps, split="val", cfg=None):
    """Win rates of each learned agent vs its scripted reference, plus cross-play if both."""
    cfg = cfg or EnvConfig()
    seeds = list(SEED_SPLITS[split])[:n_maps]
    rows = []
    if HUNTER in agents:
        rows.append(("ppo", REF_OPP[HUNTER],
                     evaluate(agents[HUNTER], make_agent(REF_OPP[HUNTER]), seeds, cfg)))
    if EVADER in agents:
        rows.append((REF_OPP[EVADER], "ppo",
                     evaluate(make_agent(REF_OPP[EVADER]), agents[EVADER], seeds, cfg)))
    if len(agents) == 2:
        rows.append(("ppo", "ppo", evaluate(agents[HUNTER], agents[EVADER], seeds, cfg)))
    return rows


def generalization(agents, opp_spec, n_maps):
    """Final train / val / test / OOD win rates of each agent vs its reference opponent."""
    out = {}
    for role, me in agents.items():
        opp = make_agent(opp_spec.get(role, REF_OPP[role]))
        res = {}
        for split, cfg in (("train", EnvConfig()), ("val", EnvConfig()),
                           ("test", EnvConfig()), ("ood", ood_cfg())):
            seeds = list(SEED_SPLITS["test" if split == "ood" else split])[:n_maps]
            h, e = (me, opp) if role == HUNTER else (opp, me)
            st = evaluate(h, e, seeds, cfg)
            res[split] = {"win": st["hunter_win" if role == HUNTER else "evader_win"], **st}
        res["gap_train_test"] = res["train"]["win"] - res["test"]["win"]
        out[role] = res
    return out


def build_setup(args):
    """Returns learner roles, per-env scripted specs and per-env learner/frozen assignment."""
    N = args.n_envs
    if args.mode == "single":
        role = args.role
        other = EVADER if role == HUNTER else HUNTER
        opp = args.opponent or default_opponent(role)
        learner_envs = {role: list(range(N))}
        if ":" in opp:
            frozen = FrozenPolicy(LearnedAgent(opp.split(":", 1)[1]).net, tag=opp,
                                  deterministic=opp.startswith("ppogreedy:"))
            scripted = [{} for _ in range(N)]
            frozen_slots = {other: [frozen] * N}
        else:
            scripted = [{other: opp} for _ in range(N)]
            frozen_slots = {}
        return [role], learner_envs, scripted, frozen_slots, {role: opp}
    if args.mode == "independent":
        return [HUNTER, EVADER], {HUNTER: list(range(N)), EVADER: list(range(N))}, \
            [{} for _ in range(N)], {}, {}
    half = N // 2
    return [HUNTER, EVADER], {HUNTER: list(range(half)), EVADER: list(range(half, N))}, \
        [{} for _ in range(N)], {EVADER: [None] * half, HUNTER: [None] * (N - half)}, {}


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)

    roles, learner_envs, scripted, frozen_slots, opp_spec = build_setup(args)
    N, T = args.n_envs, args.n_steps
    ppo_cfg = PPOConfig(lr=args.lr, epochs=args.epochs, minibatch=args.minibatch,
                        ent_coef=args.ent_coef, gamma=args.gamma)
    learners = {r: PPO(OBS_DIM, N_ACTIONS, ppo_cfg, args.device) for r in roles}
    bufs = {r: RolloutBuffer(T, len(learner_envs[r]), OBS_DIM) for r in roles}

    # self-play: frozen opponents of each role live in a pool; env slots index into it
    pools = {}
    if args.mode == "selfplay":
        for r in roles:
            pools[r] = OpponentPool(args.pool_size, args.p_latest, seed=args.seed + len(pools))
            pools[r].add(learners[r].net, f"{r}@0")
        for opp_role, slots in frozen_slots.items():
            for k in range(len(slots)):
                slots[k] = pools[opp_role].sample()
    # env indices where each frozen-opponent role is played (slot k <-> frozen_envs[role][k])
    frozen_envs = {o: [i for i in range(N) if i not in learner_envs.get(o, [])]
                   for o in frozen_slots}

    single = args.mode == "single"
    threshold = args.cur_threshold if args.cur_threshold is not None else (0.6 if single else 0.8)
    cur = Curriculum(args.curriculum, args.steps, LEVELS, threshold=threshold,
                     window=args.cur_window, min_steps=args.cur_min_steps,
                     max_frac=args.cur_max_frac)
    use_eps = args.curriculum != "off"

    seeds = list(SEED_SPLITS["train"])
    venv = make_vec_env(N, seeds, args.seed, scripted, n_workers=args.workers,
                        level=cur.level, base_cfg=EnvConfig(shaping=args.shaping))
    venv.call("set_level", cur.level, use_eps)

    (out / "config.json").write_text(json.dumps(
        {"args": vars(args), "ppo": ppo_cfg.__dict__, "levels": LEVELS,
         "opponent": opp_spec}, indent=2))
    log_ep = CsvLog(out / "episodes.csv",
                    ["timesteps", "env_steps", "learner", "opponent", "level", "map_seed",
                     "outcome", "length", "ret_hunter", "ret_evader"])
    log_up = CsvLog(out / "updates.csv",
                    ["timesteps", "role", "lr", "pg_loss", "v_loss", "entropy", "approx_kl",
                     "clipfrac", "explained_var", "epochs", "level", "fps", "pool"])
    log_ev = CsvLog(out / "eval.csv",
                    ["timesteps", "level", "hunter", "evader", "split", "hunter_win",
                     "evader_win", "timeout", "mean_len"])
    log_cur = CsvLog(out / "curriculum.csv", ["timesteps", "level", "rate"])
    log_cur.row(0, cur.level, 0.0)

    env_role = ["both" if args.mode == "independent" else
                next(r for r in roles if i in learner_envs[r]) for i in range(N)]
    opp_tag = [""] * N
    for opp_role, slots in frozen_slots.items():
        for k, i in enumerate(frozen_envs[opp_role]):
            opp_tag[i] = slots[k].tag
    for i in range(N):
        for r, s in scripted[i].items():
            opp_tag[i] = s

    recent = {r: deque(maxlen=100) for r in roles}
    obs = venv.reset()
    samples = 0
    env_steps = 0
    n_updates = int(np.ceil(args.steps / T / len(learner_envs[roles[0]])))
    next_eval, next_snap = args.eval_freq, args.snapshot_every
    t0 = time.time()

    frozen_opp = None
    if single and ":" in opp_spec[roles[0]]:
        frozen_opp = make_agent(opp_spec[roles[0]])

    def do_eval(ts):
        cfg = EnvConfig(shaping=args.shaping)
        agents = {r: LearnedAgent(p.net, algo="ppo") for r, p in learners.items()}
        rows = reference_evals(agents, args.n_eval_maps, "val", cfg)
        if frozen_opp is not None:
            seeds = list(SEED_SPLITS["val"])[:args.n_eval_maps]
            me = agents[roles[0]]
            h, e = (me, frozen_opp) if roles[0] == HUNTER else (frozen_opp, me)
            rows.append(("ppo" if roles[0] == HUNTER else "frozen",
                         "frozen" if roles[0] == HUNTER else "ppo", evaluate(h, e, seeds, cfg)))
        for h, e, st in rows:
            log_ev.row(ts, cur.level, h, e, "val", st["hunter_win"], st["evader_win"],
                       st["timeout"], st["mean_len"])
            yield h, e, st

    for u in range(n_updates):
        tu = time.time()
        for _ in range(T):
            actions = np.full((N, 2), -1, np.int64)
            step_data = {}
            for r in roles:
                idx = learner_envs[r]
                a, logp, v = learners[r].act(obs[idx, ROLES.index(r)])
                actions[idx, ROLES.index(r)] = a
                step_data[r] = (obs[idx, ROLES.index(r)], a, logp, v)
            for opp_role, slots in frozen_slots.items():
                ri = ROLES.index(opp_role)
                envs_ = frozen_envs[opp_role]
                groups = {}
                for k, i in enumerate(envs_):
                    groups.setdefault(id(slots[k]), (slots[k], []))[1].append(i)
                for pol, ids in groups.values():
                    actions[ids, ri] = pol.act(obs[ids, ri])

            obs, rew, done, eps = venv.step(actions)
            env_steps += N
            for r in roles:
                idx = learner_envs[r]
                o, a, logp, v = step_data[r]
                bufs[r].add(o, a, logp, v, rew[idx, ROLES.index(r)], done[idx])
            samples += len(learner_envs[roles[0]])

            for i in np.flatnonzero(done):
                ep = eps[i]
                log_ep.row(samples, env_steps, env_role[i], opp_tag[i], ep["level"],
                           ep["map_seed"], ep["outcome"], ep["length"],
                           float(ep["ret_hunter"]), float(ep["ret_evader"]))
                for r in roles:
                    if i in learner_envs[r]:
                        recent[r].append(ep["outcome"] == WIN[r])
                if ep["level"] == cur.level:
                    if single:
                        cur.observe(ep["outcome"] == WIN[roles[0]])
                    else:
                        cur.observe(ep["outcome"] != "timeout")
                if args.mode == "selfplay":
                    for opp_role, slots in frozen_slots.items():
                        if i in frozen_envs[opp_role]:
                            k = frozen_envs[opp_role].index(i)
                            slots[k] = pools[opp_role].sample()
                            opp_tag[i] = slots[k].tag

        progress = samples / args.steps
        fps = T * N / (time.time() - tu)
        for r in roles:
            idx = learner_envs[r]
            last_v = learners[r].value(obs[idx, ROLES.index(r)])
            st = learners[r].update(bufs[r], last_v, progress)
            log_up.row(samples, r, st["lr"], st["pg_loss"], st["v_loss"], st["entropy"],
                       st["approx_kl"], st["clipfrac"], st["explained_var"], st["epochs"],
                       cur.level, fps, len(pools.get(r, [])))

        if cur.update(samples):
            venv.call("set_level", cur.level, use_eps)
            log_cur.row(samples, cur.level, cur.rate())
            print(f"[{samples:>9}] curriculum -> level {cur.level}", flush=True)

        if args.mode == "selfplay" and samples >= next_snap:
            next_snap += args.snapshot_every
            for r in roles:
                pools[r].add(learners[r].net, f"{r}@{samples}")

        if samples >= next_eval or u == n_updates - 1:
            next_eval += args.eval_freq
            for r in roles:
                save_checkpoint(out / "ckpt" / f"{r}_{samples}.pt", "ppo", learners[r].net,
                                {"timesteps": samples, "level": cur.level})
            msg = [f"[{samples:>9}] lvl={cur.level} fps={fps:.0f}"]
            for r in roles:
                tr = sum(recent[r]) / max(1, len(recent[r]))
                msg.append(f"{r[0].upper()}-train={tr:.2f}")
            for h, e, st in do_eval(samples):
                w = st["hunter_win"] if h == "ppo" and e != "ppo" else                     st["evader_win"] if e == "ppo" and h != "ppo" else st["hunter_win"]
                msg.append(f"{h}-vs-{e}={w:.2f}")
            log_ev.f.flush()
            log_ep.f.flush()
            log_up.f.flush()
            log_cur.f.flush()
            print("  ".join(msg), flush=True)

    venv.close()
    minutes = (time.time() - t0) / 60
    for r in roles:
        save_checkpoint(out / f"{r}.pt", "ppo", learners[r].net,
                        {"timesteps": samples, "mode": args.mode})
    for lg in (log_ep, log_up, log_ev, log_cur):
        lg.close()

    summary = {"mode": args.mode, "curriculum": args.curriculum, "roles": roles,
               "samples_per_learner": samples, "env_steps": env_steps,
               "minutes": round(minutes, 2), "curriculum_history": cur.history,
               "generalization": generalization(
                   {r: LearnedAgent(p.net, algo="ppo") for r, p in learners.items()},
                   opp_spec, args.n_final_maps)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({r: {k: (v["win"] if isinstance(v, dict) else v)
                          for k, v in g.items()}
                      for r, g in summary["generalization"].items()}, indent=2))
    print(f"done in {minutes:.1f} min, saved to {out}")
    return summary


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="single", choices=["single", "independent", "selfplay"])
    ap.add_argument("--role", default="hunter", choices=["hunter", "evader"],
                    help="learner role in single mode")
    ap.add_argument("--opponent", default=None,
                    help="single mode: scripted spec or ppo:<model.pt> (default path / chase)")
    ap.add_argument("--curriculum", default="off", choices=["off", "linear", "adaptive"])
    ap.add_argument("--cur-threshold", type=float, default=None,
                    help="adaptive: win rate (single) or decisive rate (two learners)")
    ap.add_argument("--cur-window", type=int, default=300)
    ap.add_argument("--cur-min-steps", type=int, default=50_000)
    ap.add_argument("--cur-max-frac", type=float, default=0.2)
    ap.add_argument("--steps", type=int, default=1_000_000, help="samples per learner")
    ap.add_argument("--n-envs", type=int, default=32)
    ap.add_argument("--n-steps", type=int, default=256, help="rollout length per env")
    ap.add_argument("--workers", type=int, default=8, help="env processes (0 = in-process)")
    ap.add_argument("--threads", type=int, default=4, help="torch threads in main process")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--shaping", type=float, default=0.01)
    ap.add_argument("--pool-size", type=int, default=20)
    ap.add_argument("--p-latest", type=float, default=0.5)
    ap.add_argument("--snapshot-every", type=int, default=50_000)
    ap.add_argument("--eval-freq", type=int, default=100_000)
    ap.add_argument("--n-eval-maps", type=int, default=100)
    ap.add_argument("--n-final-maps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    return ap


def default_out(args):
    if args.mode == "single":
        name = f"{args.role}_vs_{(args.opponent or default_opponent(args.role)).replace(':', '_').replace('/', '_').replace('.', '_')}"
    else:
        name = args.mode
    if args.curriculum != "off":
        name += f"_cur-{args.curriculum}"
    return f"experiments/{name}_s{args.seed}"


def main():
    args = build_parser().parse_args()
    args.out = args.out or default_out(args)
    train(args)


if __name__ == "__main__":
    main()
