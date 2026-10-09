import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def smooth(x, y, w):
    w = max(1, min(w, len(y) // 4 or 1))
    if len(y) < w:
        return x, y
    k = np.ones(w) / w
    return x[w - 1:], np.convolve(y, k, mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--window", type=int, default=200)
    a = ap.parse_args()
    run = Path(a.run_dir)

    eps = [r for r in read(run / "episodes.csv") if r["reward"] != ""]
    x = np.array([float(r["timesteps"]) for r in eps])
    rew = np.array([float(r["reward"]) for r in eps])
    win = np.array([float(r["win"]) for r in eps])
    length = np.array([float(r["length"]) for r in eps])
    val = read(run / "val.csv") if (run / "val.csv").exists() else []

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    xs, ys = smooth(x, rew, a.window)
    ax[0].plot(xs, ys)
    ax[0].set_title("Episode reward (rolling mean)")
    xs, ys = smooth(x, win, a.window)
    ax[1].plot(xs, ys, label="train episodes")
    if val:
        ax[1].plot([float(r["timesteps"]) for r in val],
                   [float(r["win_rate"]) for r in val], "o-", label="validation maps")
    ax[1].set_ylim(-0.02, 1.02)
    ax[1].legend()
    ax[1].set_title("Win rate")
    xs, ys = smooth(x, length, a.window)
    ax[2].plot(xs, ys)
    ax[2].set_title("Episode length (rolling mean)")
    for axis in ax:
        axis.set_xlabel("training steps")
        axis.grid(alpha=0.3)
    fig.suptitle(run.name)
    fig.tight_layout()
    out = run / "training_curves.png"
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()
