import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPTED = ("path", "chase", "chase_oracle", "random", "idle")
ROLE_COLOR = {"hunter": "#D55E00", "evader": "#0072B2"}


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    out = {}
    for k in rows[0]:
        col = [r[k] for r in rows]
        try:
            out[k] = np.array([float(v) if v != "" else np.nan for v in col])
        except ValueError:
            out[k] = np.array(col)
    return out


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def rolling(x, y, w):
    w = max(1, min(w, len(y)))
    if len(y) < 2:
        return x, y
    c = np.cumsum(np.insert(np.asarray(y, float), 0, 0))
    return x[w - 1:], (c[w:] - c[:-w]) / w


def train_win_curves(run, window=200):
    ep = read_csv(Path(run) / "episodes.csv")
    out = {}
    if not ep:
        return out
    for role, won in (("hunter", "caught"), ("evader", "escaped")):
        learners = (role, "both")
        m = np.isin(ep["learner"], learners)
        if m.any():
            x, y = ep["timesteps"][m], (ep["outcome"][m] == won).astype(float)
            out[role] = rolling(x, y, min(window, max(10, len(y) // 5)))
    return out


def _learned(name):
    return name not in SCRIPTED and name != "frozen"


def eval_curves(run, split="val"):
    """{'ref_hunter','ref_evader','cross'} -> (timesteps, win rate of that role; cross: hunter win)."""
    ev = read_csv(Path(run) / "eval.csv")
    out = {}
    if not ev:
        return out
    sp = ev["split"] == split
    hl = np.array([_learned(h) for h in ev["hunter"]])
    el = np.array([_learned(e) for e in ev["evader"]])
    hs = np.isin(ev["hunter"], SCRIPTED)
    es = np.isin(ev["evader"], SCRIPTED)
    for key, m, col in (("ref_hunter", sp & hl & es, "hunter_win"),
                        ("ref_evader", sp & hs & el, "evader_win"),
                        ("cross", sp & hl & el, "hunter_win")):
        if m.any():
            o = np.argsort(ev["timesteps"][m])
            out[key] = (ev["timesteps"][m][o], ev[col][m][o])
    return out


def level_steps(run):
    cur = read_csv(Path(run) / "curriculum.csv")
    if cur:
        return cur["timesteps"], cur["level"], cur.get("rate")
    h = read_json(Path(run) / "summary.json").get("curriculum_history")
    if h:
        h = np.array(h, float)
        return h[:, 0], h[:, 1], None
    return None


def plot_run(run, window=200):
    run = Path(run)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    ax = axes[0, 0]
    for role, (x, y) in train_win_curves(run, window).items():
        ax.plot(x, y, color=ROLE_COLOR[role], label=f"{role} win (train)")
    _style(ax, "Train win rate (rolling)", "win rate", True)

    ax = axes[0, 1]
    ev = eval_curves(run)
    names = {"ref_hunter": ("hunter vs scripted", ROLE_COLOR["hunter"], "o-"),
             "ref_evader": ("evader vs scripted", ROLE_COLOR["evader"], "o-"),
             "cross": ("cross-play (hunter win)", "#444444", "s--")}
    for k, (x, y) in ev.items():
        lab, c, fmt = names[k]
        ax.plot(x, y, fmt, color=c, ms=3.5, label=lab)
    _style(ax, "Validation win rate", "win rate", True)
    lv = level_steps(run)
    if lv is not None:
        ax2 = ax.twinx()
        ax2.step(lv[0], lv[1], where="post", color="#999999", lw=1.2, alpha=0.8)
        ax2.set_ylabel("curriculum level", color="#777777")
        ax2.yaxis.get_major_locator().set_params(integer=True)
        ax2.grid(False)

    ax = axes[0, 2]
    ep = read_csv(run / "episodes.csv")
    if ep:
        for role in ("hunter", "evader"):
            m = np.isin(ep["learner"], (role, "both"))
            if m.any():
                x, y = rolling(ep["timesteps"][m], ep["length"][m], max(10, m.sum() // 20))
                ax.plot(x, y, color=ROLE_COLOR[role], label=role)
    _style(ax, "Episode length (rolling)", "steps")

    up = read_csv(run / "updates.csv")
    dqn = "entropy" not in up
    panels = ([("loss", "TD loss"), ("q_mean", "mean Q"), ("epsilon", "epsilon")] if dqn else
              [("entropy", "Policy entropy"), ("approx_kl", "Approx. KL"),
               ("explained_var", "Explained variance")])
    for ax, (col, title) in zip(axes[1], panels):
        if col in up:
            for role in ("hunter", "evader"):
                m = up["role"] == role
                if m.any():
                    x, y = rolling(up["timesteps"][m], up[col][m], max(1, m.sum() // 30))
                    ax.plot(x, y, color=ROLE_COLOR[role], label=role)
        _style(ax, title, col)
    fig.suptitle(run.name)
    fig.tight_layout()
    return fig


def _style(ax, title, ylabel, unit=False):
    ax.set_title(title)
    ax.set_xlabel("samples per learner")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    if unit:
        ax.set_ylim(-0.02, 1.02)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", color="#999999")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else Path(a.run_dir) / "training_curves.png"
    fig = plot_run(a.run_dir, a.window)
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()
