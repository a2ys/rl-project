import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agents.factory import make_agent
from agents.learned import LearnedAgent
from env.dungeon_env import EVADER, HUNTER, MOVES, DungeonPursuitEnv, EnvConfig
from env.generator import WALL
from visualization import theme as th

ROLES = (HUNTER, EVADER)


def _rgb(c):
    return tuple(v / 255 for v in c[:3])


def decision_maps(env, agents, roles=ROLES):
    out = {}
    for role in roles:
        agent = agents[role]
        shape = (env.H, env.W)
        m = dict(value=np.full(shape, np.nan), action=np.full(shape, -1, int),
                 prob=np.full(shape, np.nan), entropy=np.full(shape, np.nan))
        saved = env.pos[role]
        for r in range(env.H):
            for c in range(env.W):
                if env.grid[r, c] == WALL:
                    continue
                env.pos[role] = (r, c)
                obs = env.observe(role)
                p = np.clip(agent.probs(obs), 1e-12, 1.0)
                a = int(p.argmax())
                m["value"][r, c] = agent.value(obs)
                m["action"][r, c], m["prob"][r, c] = a, p[a]
                m["entropy"][r, c] = -(p * np.log(p)).sum()
        env.pos[role] = saved
        out[role] = m
    return out


def _decorate(ax, env, role):
    other = EVADER if role == HUNTER else HUNTER
    ax.plot(env.exit_pos[1], env.exit_pos[0], "*", color=_rgb(th.EXIT_GREEN), ms=15, mec="k")
    for r, c in env.doors:
        ax.plot(c, r, "s", color=_rgb(th.WOOD), ms=8, mec="k")
    for r, c in env.keys:
        ax.plot(c, r, "o", color=_rgb(th.KEY_GOLD), ms=7, mec="k")
    r, c = env.pos[other]
    ax.plot(c, r, "X", color=_rgb(th.ROLE_COLOR[other]), ms=11, mec="w", label=f"{other} (fixed)")
    ax.set_xticks([])
    ax.set_yticks([])


def _panel(fig, ax, env, role, m, kind, algo):
    wall = np.where(env.grid == WALL, 0.0, np.nan)
    ax.imshow(wall, cmap="gray", vmin=0, vmax=1)
    if kind == "value":
        data, cmap, title = m["value"], "viridis", "max Q" if algo == "dqn" else "V(s)"
    else:
        data, cmap, title = m["entropy"], "magma", "policy entropy"
    lim = {} if kind == "value" else dict(vmin=0, vmax=np.log(5))
    im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, **lim)
    fig.colorbar(im, ax=ax, fraction=0.046)
    if kind == "value":
        for r, c in zip(*np.nonzero(m["action"] >= 0)):
            a, p = m["action"][r, c], m["prob"][r, c]
            if a == 4:
                ax.plot(c, r, "o", color="w", ms=2 + 4 * p, alpha=0.4 + 0.6 * p)
            else:
                dr, dc = MOVES[a]
                n = 0.25 + 0.55 * p
                ax.arrow(c - dc * n / 2, r - dr * n / 2, dc * n, dr * n,
                         head_width=0.22, length_includes_head=True,
                         color="w", alpha=0.3 + 0.7 * p, lw=0.8)
    _decorate(ax, env, role)
    ax.set_title(f"{role}: {title}")


def plot_decision_maps(env, agents, roles, out, seed=None, step=0):
    maps = decision_maps(env, agents, roles)
    fig, axes = plt.subplots(len(roles), 2, figsize=(11, 5.5 * len(roles)), squeeze=False)
    for i, role in enumerate(roles):
        algo = getattr(agents[role], "algo", "ppo")
        for j, kind in enumerate(("value", "entropy")):
            _panel(fig, axes[i, j], env, role, maps[role], kind, algo)
    axes[0, 0].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"Decision map (seed {seed}, step {step})")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("saved", out)
    return maps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hunter", default="random")
    ap.add_argument("--evader", default="random")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--role", choices=("hunter", "evader", "both"), default="both")
    ap.add_argument("--step", type=int, default=0, help="steps to play before analysing")
    ap.add_argument("--out", default="experiments/decision_map.png")
    a = ap.parse_args()

    env = DungeonPursuitEnv(EnvConfig())
    env.reset(seed=a.seed)
    agents = {HUNTER: make_agent(a.hunter), EVADER: make_agent(a.evader)}
    for ag in agents.values():
        ag.reset()
    for _ in range(a.step):
        acts = {k: agents[k].act(env, k) for k in ROLES}
        _, _, term, trunc, _ = env.step(acts)
        if term or trunc:
            break
    roles = ROLES if a.role == "both" else (a.role,)
    for r in roles:
        if not isinstance(agents[r], LearnedAgent):
            ap.error(f"--{r} must be a learned agent (ppo:/pposample:/dqn:), got scripted")
    plot_decision_maps(env, agents, roles, a.out, a.seed, env.t)


if __name__ == "__main__":
    main()
