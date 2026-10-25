"""Cross-play matrix between saved checkpoints of one run (or two runs).

Entry [i, j] = Hunter win rate of hunter checkpoint i vs evader checkpoint j on val maps.
Monotone training shows later hunters beating all earlier evaders; cycling (typical of
naive simultaneous learning) shows up as non-transitive bands.

  python -m evaluation.crossplay experiments/independent_s0 --max-ckpts 10
"""
import argparse
import csv
import re
from pathlib import Path

import numpy as np

from agents.learned import LearnedAgent
from env.dungeon_env import EnvConfig
from evaluation.evaluate import evaluate, split_seeds


def checkpoints(run, role, max_n):
    found = []
    for p in (Path(run) / "ckpt").glob(f"{role}_*.pt"):
        m = re.fullmatch(rf"{role}_(\d+)\.pt", p.name)
        if m:
            found.append((int(m.group(1)), p))
    found.sort()
    if len(found) > max_n:
        keep = np.unique(np.linspace(0, len(found) - 1, max_n).round().astype(int))
        found = [found[i] for i in keep]
    return found


def crossplay(hunters, evaders, seeds, cfg=None):
    M = np.zeros((len(hunters), len(evaders)))
    T = np.zeros_like(M)
    for i, (_, hp) in enumerate(hunters):
        h = LearnedAgent(str(hp))
        for j, (_, ep) in enumerate(evaders):
            st = evaluate(h, LearnedAgent(str(ep)), seeds, cfg)
            M[i, j], T[i, j] = st["hunter_win"], st["timeout"]
    return M, T


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--evader-run", default=None, help="take evader checkpoints from here")
    ap.add_argument("--max-ckpts", type=int, default=10)
    ap.add_argument("--n-maps", type=int, default=50)
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    hunters = checkpoints(args.run, "hunter", args.max_ckpts)
    evaders = checkpoints(args.evader_run or args.run, "evader", args.max_ckpts)
    M, T = crossplay(hunters, evaders, split_seeds(args.split, args.n_maps), EnvConfig())

    out = Path(args.run) / "crossplay.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hunter_ckpt", "evader_ckpt", "hunter_win", "timeout"])
        for i, (hs, _) in enumerate(hunters):
            for j, (es, _) in enumerate(evaders):
                w.writerow([hs, es, round(M[i, j], 4), round(T[i, j], 4)])

    print("hunter win rate (rows: hunter ckpt, cols: evader ckpt)")
    print(" " * 10 + "".join(f"{es // 1000:>7}k" for es, _ in evaders))
    for i, (hs, _) in enumerate(hunters):
        print(f"{hs // 1000:>8}k " + "".join(f"{M[i, j]:>8.2f}" for j in range(len(evaders))))
    # Non-transitivity: fraction of hunter pairs (a later than b) where the later hunter
    # does worse against some evader than the earlier one by > 0.1.
    regress = [(M[b] - M[a] > 0.1).mean() for a in range(len(M)) for b in range(a)]
    print(f"regression rate of later hunters = {np.mean(regress) if regress else 0:.2f}")
    print("saved", out)


if __name__ == "__main__":
    main()
