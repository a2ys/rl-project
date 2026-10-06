"""Visitation heatmap of both agents on one map over many episodes.
Use a stochastic agent (random or pposample:) otherwise every episode is identical.

python -m visualization.heatmap --hunter random --evader random --seed 42 --episodes 200
python -m visualization.heatmap --hunter pposample:experiments/hunter_vs_path_s0/model --evader path
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agents.factory import make_agent
from env.dungeon_env import EVADER, HUNTER, DungeonPursuitEnv, EnvConfig
from env.generator import WALL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hunter", default="random")
    ap.add_argument("--evader", default="random")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--out", default="experiments/heatmap.png")
    a = ap.parse_args()

    env = DungeonPursuitEnv(EnvConfig())
    hunter, evader = make_agent(a.hunter), make_agent(a.evader)
    counts = {k: np.zeros((env.H, env.W)) for k in (HUNTER, EVADER)}
    for ep in range(a.episodes):
        if ep == 0:
            env.reset(seed=a.seed)
        else:
            env.reset(options={"map_seed": a.seed})   # keep rng stream running
        hunter.reset()
        evader.reset()
        while True:
            for k in counts:
                counts[k][env.pos[k]] += 1
            acts = {HUNTER: hunter.act(env, HUNTER), EVADER: evader.act(env, EVADER)}
            _, _, term, trunc, _ = env.step(acts)
            if term or trunc:
                for k in counts:
                    counts[k][env.pos[k]] += 1
                break

    d = env.dungeon
    base = np.where(d.grid == WALL, 0.25, 0.92)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, role, title in zip(axes, (HUNTER, EVADER), ("Hunter visits", "Evader visits")):
        ax.imshow(base, cmap="gray", vmin=0, vmax=1)
        m = np.ma.masked_where(counts[role] == 0, counts[role])
        im = ax.imshow(m, cmap="magma_r", alpha=0.85)
        ax.plot(d.exit_pos[1], d.exit_pos[0], "g*", markersize=16, label="exit")
        for r, c in d.doors:
            ax.plot(c, r, "s", color="saddlebrown", markersize=8)
        for r, c in d.keys:
            ax.plot(c, r, "o", color="gold", markersize=8, markeredgecolor="k")
        ax.set_title(f"{title} (seed {a.seed}, {a.episodes} episodes)")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print("saved", a.out)


if __name__ == "__main__":
    main()
