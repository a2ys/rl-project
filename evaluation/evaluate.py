import argparse
import copy
import csv
from pathlib import Path

import numpy as np

from agents.factory import make_agent
from env.dungeon_env import EVADER, HUNTER, DungeonPursuitEnv, EnvConfig
from env.generator import SEED_SPLITS


def _play_chunk(hunter, evader, seeds, cfg):
    """Plays one episode per seed with all maps in lockstep. Agents with `act_batch`
    get one batched call per step; scripted agents are copied per map (they keep state)."""
    envs = [DungeonPursuitEnv(cfg, seeds=[0]) for _ in seeds]
    obs = [e.reset(seed=int(s))[0] for e, s in zip(envs, seeds)]
    players = {}
    for role, ag in ((HUNTER, hunter), (EVADER, evader)):
        if hasattr(ag, "act_batch"):
            ag.reset()
            players[role] = ag
        else:
            players[role] = [copy.deepcopy(ag) for _ in envs]
            for a in players[role]:
                a.reset()
    active = list(range(len(envs)))
    ret = np.zeros((len(envs), 2))
    outcomes = [""] * len(envs)
    while active:
        acts = {}
        for role, p in players.items():
            if isinstance(p, list):
                acts[role] = [p[i].act(envs[i], role) for i in active]
            else:
                acts[role] = p.act_batch(np.stack([obs[i][role] for i in active]))
        still = []
        for j, i in enumerate(active):
            o, rew, term, trunc, info = envs[i].step(
                {HUNTER: int(acts[HUNTER][j]), EVADER: int(acts[EVADER][j])})
            obs[i] = o
            ret[i] += (rew[HUNTER], rew[EVADER])
            if term or trunc:
                outcomes[i] = info["outcome"]
            else:
                still.append(i)
        active = still
    return outcomes, [e.t for e in envs], ret


def evaluate(hunter, evader, seeds, cfg=None, chunk=100):
    cfg = cfg or EnvConfig()
    seeds = list(seeds)
    outcomes, lens, rets = [], [], []
    for s in range(0, len(seeds), chunk):
        o, l, r = _play_chunk(hunter, evader, seeds[s:s + chunk], cfg)
        outcomes += o
        lens += l
        rets.append(r)
    ret = np.concatenate(rets)
    n = len(outcomes)
    nh = sum(o == "caught" for o in outcomes)
    ne = sum(o == "escaped" for o in outcomes)
    nt = sum(o == "timeout" for o in outcomes)
    rh, re = ret[:, 0], ret[:, 1]
    return dict(n=n, hunter_win=nh / n, evader_win=ne / n, timeout=nt / n,
                mean_len=float(np.mean(lens)), ret_hunter=float(np.mean(rh)),
                ret_evader=float(np.mean(re)))


def split_seeds(split, n):
    return list(SEED_SPLITS[split])[:n]


def print_table(rows):
    print(f"{'hunter':<34}{'evader':<34}{'H-win':>7}{'E-win':>7}{'T/O':>7}{'len':>7}")
    for r in rows:
        print(f"{r['hunter']:<34}{r['evader']:<34}{r['hunter_win']:>7.2f}"
              f"{r['evader_win']:>7.2f}{r['timeout']:>7.2f}{r['mean_len']:>7.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--n-maps", type=int, default=100)
    ap.add_argument("--hunter-model", default=None)
    ap.add_argument("--evader-model", default=None)
    ap.add_argument("--out", default="experiments/baselines.csv")
    args = ap.parse_args()

    matchups = [("random", "random"), ("chase", "random"), ("random", "path"),
                ("chase", "path"), ("chase_oracle", "path")]
    if args.hunter_model:
        matchups += [(f"ppo:{args.hunter_model}", "path"),
                     (f"ppo:{args.hunter_model}", "random")]
    if args.evader_model:
        matchups += [("chase", f"ppo:{args.evader_model}"),
                     ("random", f"ppo:{args.evader_model}")]

    seeds = split_seeds(args.split, args.n_maps)
    cfg = EnvConfig()
    rows = []
    for h, e in matchups:
        st = evaluate(make_agent(h), make_agent(e), seeds, cfg)
        rows.append({"hunter": h, "evader": e, "split": args.split, **st})
    print(f"split={args.split}, maps={len(seeds)}")
    print_table(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("saved", args.out)


if __name__ == "__main__":
    main()
