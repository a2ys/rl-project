"""DQN baseline: one role vs a scripted opponent (same envs, curriculum and logs as PPO single)."""
import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from agents.learned import LearnedAgent
from env.dungeon_env import EVADER, HUNTER, EnvConfig
from env.generator import SEED_SPLITS
from rl.curriculum import LEVELS, Curriculum
from rl.dqn import DQN, DQNConfig
from rl.nets import save_checkpoint
from rl.vec_env import ROLES, make_vec_env
from training.train import (N_ACTIONS, OBS_DIM, WIN, CsvLog, default_opponent,
                            generalization, reference_evals)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)

    role = args.role
    other = EVADER if role == HUNTER else HUNTER
    ri = ROLES.index(role)
    opp = args.opponent or default_opponent(role)
    N = args.n_envs
    cfg = DQNConfig(lr=args.lr, buffer_size=args.buffer_size, batch_size=args.batch_size,
                    learning_starts=args.learning_starts, gamma=args.gamma)
    dqn = DQN(OBS_DIM, N_ACTIONS, cfg, args.device, args.seed)
    cur = Curriculum(args.curriculum, args.steps, LEVELS, threshold=args.cur_threshold,
                     window=args.cur_window, min_steps=args.cur_min_steps,
                     max_frac=args.cur_max_frac)
    use_eps = args.curriculum != "off"
    venv = make_vec_env(N, list(SEED_SPLITS["train"]), args.seed, [{other: opp}] * N,
                        n_workers=args.workers, level=cur.level,
                        base_cfg=EnvConfig(shaping=args.shaping))
    venv.call("set_level", cur.level, use_eps)

    (out / "config.json").write_text(json.dumps(
        {"args": vars(args), "dqn": cfg.__dict__, "levels": LEVELS, "opponent": opp},
        indent=2))
    log_ep = CsvLog(out / "episodes.csv",
                    ["timesteps", "env_steps", "learner", "opponent", "level", "map_seed",
                     "outcome", "length", "ret_hunter", "ret_evader"])
    log_up = CsvLog(out / "updates.csv",
                    ["timesteps", "role", "loss", "q_mean", "epsilon", "level", "fps"])
    log_ev = CsvLog(out / "eval.csv",
                    ["timesteps", "level", "hunter", "evader", "split", "hunter_win",
                     "evader_win", "timeout", "mean_len"])
    log_cur = CsvLog(out / "curriculum.csv", ["timesteps", "level", "rate"])
    log_cur.row(0, cur.level, 0.0)

    obs = venv.reset()
    recent = deque(maxlen=100)
    losses, qs = deque(maxlen=200), deque(maxlen=200)
    samples, step, next_eval, next_log = 0, 0, args.eval_freq, args.log_every
    t0 = tl = time.time()
    while samples < args.steps:
        eps = dqn.epsilon(samples / args.steps)
        actions = np.full((N, 2), -1, np.int64)
        a = dqn.act(obs[:, ri], eps)
        actions[:, ri] = a
        nobs, rew, done, infos = venv.step(actions)
        # with auto-reset nobs holds the next episode's first obs on done steps; it is
        # never bootstrapped from because the done flag masks it out of the target
        dqn.buf.add(obs[:, ri], a, rew[:, ri], nobs[:, ri], done.astype(np.float32))
        obs = nobs
        samples += N
        step += 1
        for i in np.flatnonzero(done):
            ep = infos[i]
            log_ep.row(samples, samples, role, opp, ep["level"], ep["map_seed"],
                       ep["outcome"], ep["length"], float(ep["ret_hunter"]),
                       float(ep["ret_evader"]))
            recent.append(ep["outcome"] == WIN[role])
            if ep["level"] == cur.level:
                cur.observe(ep["outcome"] == WIN[role])

        if samples >= cfg.learning_starts and step % cfg.train_every == 0:
            for _ in range(cfg.grad_steps):
                l, q = dqn.train_step()
                losses.append(l)
                qs.append(q)

        if cur.update(samples):
            venv.call("set_level", cur.level, use_eps)
            log_cur.row(samples, cur.level, cur.rate())
            print(f"[{samples:>9}] curriculum -> level {cur.level}", flush=True)

        if samples >= next_log:
            next_log += args.log_every
            fps = args.log_every / (time.time() - tl)
            tl = time.time()
            log_up.row(samples, role, float(np.mean(losses)) if losses else 0.0,
                       float(np.mean(qs)) if qs else 0.0, eps, cur.level, fps)

        if samples >= next_eval or samples >= args.steps:
            next_eval += args.eval_freq
            save_checkpoint(out / "ckpt" / f"{role}_{samples}.pt", "dqn", dqn.q,
                            {"timesteps": samples, "level": cur.level})
            agent = LearnedAgent(dqn.q, algo="dqn")
            msg = f"[{samples:>9}] lvl={cur.level} eps={eps:.2f} train={np.mean(recent) if recent else 0:.2f}"
            for h, e, st in reference_evals({role: agent}, args.n_eval_maps, "val",
                                            EnvConfig(shaping=args.shaping)):
                log_ev.row(samples, cur.level, h, e, "val", st["hunter_win"],
                           st["evader_win"], st["timeout"], st["mean_len"])
                w = st["hunter_win"] if role == HUNTER else st["evader_win"]
                msg += f"  val-win={w:.2f}"
            for lg in (log_ep, log_up, log_ev, log_cur):
                lg.f.flush()
            print(msg, flush=True)

    venv.close()
    minutes = (time.time() - t0) / 60
    save_checkpoint(out / f"{role}.pt", "dqn", dqn.q, {"timesteps": samples, "mode": "dqn"})
    for lg in (log_ep, log_up, log_ev, log_cur):
        lg.close()
    summary = {"algo": "dqn", "mode": "single", "curriculum": args.curriculum, "roles": [role],
               "samples_per_learner": samples, "env_steps": samples,
               "minutes": round(minutes, 2), "curriculum_history": cur.history,
               "generalization": generalization({role: LearnedAgent(dqn.q, algo="dqn")},
                                                {role: opp}, args.n_final_maps)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    g = summary["generalization"][role]
    print({k: (v["win"] if isinstance(v, dict) else v) for k, v in g.items()})
    print(f"done in {minutes:.1f} min, saved to {out}")
    return summary


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--role", default="hunter", choices=["hunter", "evader"])
    ap.add_argument("--opponent", default=None, help="scripted spec (default path / chase)")
    ap.add_argument("--curriculum", default="off", choices=["off", "linear", "adaptive"])
    ap.add_argument("--cur-threshold", type=float, default=0.6)
    ap.add_argument("--cur-window", type=int, default=300)
    ap.add_argument("--cur-min-steps", type=int, default=50_000)
    ap.add_argument("--cur-max-frac", type=float, default=0.2)
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--buffer-size", type=int, default=200_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--learning-starts", type=int, default=20_000)
    ap.add_argument("--shaping", type=float, default=0.01)
    ap.add_argument("--log-every", type=int, default=10_000)
    ap.add_argument("--eval-freq", type=int, default=100_000)
    ap.add_argument("--n-eval-maps", type=int, default=100)
    ap.add_argument("--n-final-maps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    return ap


def main():
    args = build_parser().parse_args()
    opp = args.opponent or default_opponent(args.role)
    cur = f"_cur-{args.curriculum}" if args.curriculum != "off" else ""
    args.out = args.out or f"experiments/dqn_{args.role}_vs_{opp}{cur}_s{args.seed}"
    train(args)


if __name__ == "__main__":
    main()
