"""Evaluate matchups on a map split.

python -m evaluation.evaluate --n-maps 100
python -m evaluation.evaluate --hunter-model experiments/hunter_vs_path_s0/model \
                              --evader-model experiments/evader_vs_chase_s0/model
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from agents.factory import make_agent
from env.dungeon_env import EVADER, HUNTER, DungeonPursuitEnv, EnvConfig
from env.generator import SEED_SPLITS


def evaluate(hunter, evader, seeds, cfg=None):
    env = DungeonPursuitEnv(cfg or EnvConfig(), seeds=[0])
    n = nh = ne = nt = 0
    lens, rh, re = [], [], []
    for s in seeds:
        env.reset(seed=int(s))
        hunter.reset()
        evader.reset()
        ret_h = ret_e = 0.0
        while True:
            acts = {HUNTER: hunter.act(env, HUNTER), EVADER: evader.act(env, EVADER)}
            _, rew, term, trunc, info = env.step(acts)
            ret_h += rew[HUNTER]
            ret_e += rew[EVADER]
            if term or trunc:
                break
        o = info["outcome"]
        n += 1
        nh += o == "caught"
        ne += o == "escaped"
        nt += o == "timeout"
        lens.append(env.t)
        rh.append(ret_h)
        re.append(ret_e)
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
