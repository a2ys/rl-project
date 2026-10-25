"""Static HTML dashboard over all runs in an experiments root.

  python -m visualization.dashboard --root experiments --out experiments/dashboard.html
"""
import argparse
import base64
import html
import io
import re
import tempfile
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from visualization.plot_training import eval_curves, level_steps, read_csv, read_json

ORDER = ["hunter_vs_path", "hunter_vs_path_cur-adaptive", "evader_vs_chase",
         "evader_vs_chase_cur-adaptive", "independent", "independent_cur-adaptive",
         "selfplay", "selfplay_cur-adaptive", "dqn_hunter_vs_path", "dqn_evader_vs_chase"]
BASE_COLOR = {"hunter_vs_path": "#D55E00", "evader_vs_chase": "#0072B2",
              "independent": "#009E73", "selfplay": "#CC79A7",
              "dqn_hunter_vs_path": "#E69F00", "dqn_evader_vs_chase": "#56B4E9"}
FALLBACK = ["#7f7f7f", "#8c564b", "#bcbd22", "#17becf", "#444444"]
SPLITS = ("train", "val", "test", "ood")
SPLIT_COLOR = dict(zip(SPLITS, ("#c7d3e3", "#8fb0d6", "#4f7fb5", "#1f3f6e")))
ADVERSARIAL = ("independent", "independent_cur-adaptive", "selfplay", "selfplay_cur-adaptive")
REF_KEY = {"hunter": "ref_hunter", "evader": "ref_evader"}
CUR = "_cur-adaptive"
RUN_RE = re.compile(r"^(.+)_s(\d+)$")
EXPLOIT_RE = re.compile(r"^(?:hunter|evader)_(.+)_s(\d+)$")

CSS = """
body{font:15px/1.5 system-ui,sans-serif;background:#f6f7f9;color:#222;margin:0}
main{max-width:1100px;margin:0 auto;padding:16px 20px 48px}
h1{margin:.6em 0 .2em} h2{margin-top:2em;border-bottom:2px solid #d5d9e0;padding-bottom:4px}
figure{margin:1em 0;background:#fff;border:1px solid #dde1e7;border-radius:6px;padding:10px}
figure img{max-width:100%;height:auto;display:block;margin:0 auto}
figcaption{font-size:13.5px;color:#444;margin-top:6px}
table{border-collapse:collapse;background:#fff;font-size:13.5px;margin:1em 0}
th,td{border:1px solid #dde1e7;padding:4px 10px;text-align:right}
th{background:#eceff3} td:first-child,th:first-child,td:nth-child(2),th:nth-child(2){text-align:left}
.na{background:#fff;border:1px dashed #aab;border-radius:6px;padding:14px;color:#777;margin:1em 0}
.meta{color:#666;font-size:13px}
"""


def color_of(t):
    base = t.replace(CUR, "")
    c = BASE_COLOR.get(base)
    if c is None:
        c = FALLBACK[sum(map(ord, base)) % len(FALLBACK)]
    if t.endswith(CUR):
        c = mcolors.to_hex(np.array(mcolors.to_rgb(c)) * 0.6)
    return c


def style_of(t):
    return "--" if t.endswith(CUR) else "-"


def roles_of(t):
    if "hunter" in t.split("_vs_")[0]:
        return ["hunter"]
    if "evader" in t.split("_vs_")[0]:
        return ["evader"]
    return ["hunter", "evader"]


def discover(root):
    runs = defaultdict(list)
    for d in sorted(Path(root).iterdir()) if Path(root).exists() else []:
        m = RUN_RE.match(d.name)
        if d.is_dir() and m and (d / "eval.csv").exists() and (d / "episodes.csv").exists():
            runs[m.group(1)].append((int(m.group(2)), d))
    types = [t for t in ORDER if t in runs] + sorted(t for t in runs if t not in ORDER)
    return {t: [d for _, d in sorted(runs[t])] for t in types}


def b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def figure_html(png_b64, title, caption):
    return (f'<figure><img alt="{html.escape(title)}" src="data:image/png;base64,{png_b64}">'
            f"<figcaption>{caption}</figcaption></figure>")


def placeholder(msg):
    return f'<div class="na">{msg}</div>'


def table(headers, rows):
    h = "".join(f"<th>{c}</th>" for c in headers)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{h}</tr>{b}</table>"


def ms(vals, nd=2):
    v = np.array([x for x in vals if x is not None and not np.isnan(x)], float)
    if not len(v):
        return "-"
    return f"{v.mean():.{nd}f} &plusmn; {v.std():.{nd}f}" if len(v) > 1 else f"{v[0]:.{nd}f}"


def aggregate(curves):
    curves = [(np.asarray(x, float), np.asarray(y, float)) for x, y in curves if len(x)]
    if not curves:
        return None
    lo, hi = max(c[0][0] for c in curves), min(c[0][-1] for c in curves)
    xs = np.unique(np.concatenate([c[0] for c in curves]))
    xs = xs[(xs >= lo) & (xs <= hi)]
    if not len(xs):
        xs = max(curves, key=lambda c: len(c[0]))[0]
    ys = np.stack([np.interp(xs, x, y) for x, y in curves])
    return xs, ys.mean(0), ys.std(0), len(curves)


def band(ax, curves, t, label=None, color=None):
    a = aggregate(curves)
    if a is None:
        return False
    xs, m, s, n = a
    c = color or color_of(t)
    ax.plot(xs, m, color=c, ls=style_of(t), lw=1.8, marker="o" if len(xs) < 25 else None,
            ms=3, label=f"{label or t} (n={n})")
    if n > 1:
        ax.fill_between(xs, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1), color=c, alpha=0.15, lw=0)
    return True


def finish(ax, title, xlabel="samples per learner", ylabel="val win rate", unit=True):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    if unit:
        ax.set_ylim(-0.02, 1.02)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7.5, loc="best")
    else:
        ax.text(0.5, 0.5, "not run yet", transform=ax.transAxes, ha="center", color="#999999")


def val_curves(runs, t, key):
    return [eval_curves(d).get(key) for d in runs.get(t, []) if key in eval_curves(d)]


def summaries(runs, t):
    return [read_json(d / "summary.json") for d in runs.get(t, [])]


def sec_overview(runs):
    rows = []
    for t in runs:
        sums = summaries(runs, t)
        sums = [s for s in sums if s]
        for role in roles_of(t):
            g = [s.get("generalization", {}).get(role, {}) for s in sums]
            g = [x for x in g if x]
            if not g:
                continue
            cell = lambda sp: ms([x.get(sp, {}).get("win") for x in g])
            rows.append([t, role, len(g)] + [cell(sp) for sp in SPLITS]
                        + [ms([x.get("gap_train_test") for x in g]),
                           ms([s.get("minutes") for s in sums], 1),
                           ms([s.get("samples_per_learner") for s in sums], 0)])
    if not rows:
        return placeholder("No finished runs (summary.json) found yet.")
    cap = ("Final greedy win rate of the learner's role against its reference opponent "
           "(scripted path/chase evader/hunter), mean &plusmn; std across seeds. "
           "Gap = train minus test win rate. Look for small gaps (generalization) and "
           "high test/OOD numbers.")
    return table(["run type", "role", "seeds", "train", "val", "test", "OOD", "gap (train-test)",
                  "minutes", "samples/learner"], rows) + f'<p class="meta">{cap}</p>'


def sec_curriculum(runs):
    panels = [("Hunter vs path", "hunter", "hunter_vs_path"),
              ("Evader vs chase", "evader", "evader_vs_chase"),
              ("Independent: hunter", "hunter", "independent"),
              ("Independent: evader", "evader", "independent"),
              ("Self-play: hunter", "hunter", "selfplay"),
              ("Self-play: evader", "evader", "selfplay")]
    if not any(t in runs for _, _, b in panels for t in (b, b + CUR)):
        return placeholder("No runs found for the curriculum comparison.")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (title, role, base) in zip(axes.flat, panels):
        for t, lab in ((base, "curriculum off"), (base + CUR, "adaptive curriculum")):
            band(ax, val_curves(runs, t, REF_KEY[role]), t, lab)
            if t.endswith(CUR) and runs.get(t):
                lv = level_steps(runs[t][0])
                if lv is not None:
                    ch = np.nonzero(np.diff(lv[1], prepend=lv[1][0] - 1))[0]
                    for i in ch:
                        ax.axvline(lv[0][i], color=color_of(t), alpha=0.25, lw=0.8)
                        ax.text(lv[0][i], 0.0, f"L{int(lv[1][i])}", fontsize=6,
                                color=color_of(t), rotation=90, va="bottom", ha="right")
        finish(ax, title)
    fig.suptitle("Validation win rate vs scripted reference: curriculum off vs adaptive", y=0.995)
    fig.tight_layout()
    return figure_html(b64(fig), "curriculum", (
        "Mean &plusmn; std over seeds of the learner's win rate on validation maps against the "
        "scripted opponent. Thin vertical lines mark curriculum level changes of the first "
        "adaptive seed. Look for faster rise and a higher plateau with the curriculum, and for "
        "dips right after a level change."))


def stability_stats(curve):
    x, y = curve
    half = y[len(y) // 2:]
    drops = np.diff(y)
    return (float(half.std()) if len(half) > 1 else np.nan,
            float(max(0.0, -drops.min())) if len(drops) else np.nan)


def sec_stability(runs):
    types = [t for t in ADVERSARIAL if t in runs]
    if not types:
        return placeholder("No independent / self-play runs found yet.")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (key, title, yl) in zip(axes, (("ref_hunter", "Hunter vs scripted path", "hunter win"),
                                           ("ref_evader", "Evader vs scripted chase", "evader win"),
                                           ("cross", "Cross-play: learned hunter vs learned evader",
                                            "hunter win"))):
        for t in types:
            band(ax, val_curves(runs, t, key), t)
        finish(ax, title, ylabel=yl)
    fig.tight_layout()
    out = figure_html(b64(fig), "stability", (
        "Left/middle: val win rate against the fixed scripted reference during training. "
        "Right: win rate of the hunter when playing the evader that was trained alongside it. "
        "Stable training gives smooth, monotone curves; oscillations or collapses indicate "
        "non-stationarity from co-adapting learners."))
    rows = []
    for t in types:
        for key in ("ref_hunter", "ref_evader", "cross"):
            cs = val_curves(runs, t, key)
            st = [stability_stats(c) for c in cs]
            if st:
                rows.append([t, key.replace("ref_", "vs scripted: ") if key != "cross" else "cross-play",
                             len(st), ms([s[0] for s in st], 3), ms([s[1] for s in st], 3)])
    out += table(["run type", "curve", "seeds", "std (2nd half)", "max drop between evals"], rows)
    out += ('<p class="meta">Lower is more stable. Std is of the val win curve over its '
            "second half; max drop is the largest decrease between consecutive evaluations.</p>")

    mats = []
    for t in types:
        for d in runs[t][:1]:
            cp = read_csv(d / "crossplay.csv")
            if cp:
                mats.append((t, d.name, cp))
    if not mats:
        return out + placeholder("Cross-play matrices (crossplay.csv) not computed yet.")
    n = len(mats)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.2), squeeze=False)
    for ax, (t, name, cp) in zip(axes[0], mats):
        hs, es = np.unique(cp["hunter_ckpt"]), np.unique(cp["evader_ckpt"])
        M = np.full((len(hs), len(es)), np.nan)
        for h, e, w in zip(cp["hunter_ckpt"], cp["evader_ckpt"], cp["hunter_win"]):
            M[np.searchsorted(hs, h), np.searchsorted(es, e)] = w
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis", origin="lower", aspect="auto")
        ax.set_xticks(range(len(es)), [f"{int(e) // 1000}k" for e in es], rotation=60, fontsize=7)
        ax.set_yticks(range(len(hs)), [f"{int(h) // 1000}k" for h in hs], fontsize=7)
        ax.set_xlabel("evader checkpoint (samples)")
        ax.set_ylabel("hunter checkpoint (samples)")
        ax.set_title(name, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, label="hunter win rate")
    fig.suptitle("Checkpoint cross-play matrices (val maps)")
    fig.tight_layout()
    return out + figure_html(b64(fig), "crossplay", (
        "Hunter win rate for every pair of hunter and evader checkpoints. A transitive, "
        "improving pair shows a smooth gradient (later hunters beat earlier evaders, and later "
        "evaders resist earlier hunters). Checkerboard or banded patterns indicate cycling."))


def sec_generalization(runs):
    if not runs:
        return placeholder("No finished runs yet.")
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    any_data = False
    for ax, role in zip(axes, ("hunter", "evader")):
        types = [t for t in runs if role in roles_of(t)]
        w = 0.8 / len(SPLITS)
        for k, sp in enumerate(SPLITS):
            m, s = [], []
            for t in types:
                v = [x.get("generalization", {}).get(role, {}).get(sp, {}).get("win")
                     for x in summaries(runs, t)]
                v = np.array([a for a in v if a is not None], float)
                m.append(v.mean() if len(v) else 0)
                s.append(v.std() if len(v) else 0)
                any_data |= len(v) > 0
            ax.bar(np.arange(len(types)) + (k - 1.5) * w, m, w, yerr=s, capsize=2,
                   color=SPLIT_COLOR[sp], edgecolor="#333", lw=0.5, label=sp)
        ax.set_xticks(range(len(types)), types, rotation=30, ha="right", fontsize=8)
        finish(ax, f"{role.capitalize()} win rate by map split", xlabel="run type",
               ylabel=f"{role} win rate")
    if not any_data:
        plt.close(fig)
        return placeholder("No generalization results yet.")
    fig.tight_layout()
    return figure_html(b64(fig), "generalization", (
        "Win rate against the scripted reference on training maps, held-out validation and test "
        "maps, and out-of-distribution (larger / harder) maps; error bars are std across seeds. "
        "A large drop from train to test/OOD means the policy memorised layouts."))


def load_exploit(root):
    ex = Path(root) / "exploitability"
    curves = defaultdict(list)
    budget = {}
    bars = defaultdict(list)
    if not ex.exists():
        return curves, bars, budget
    for d in sorted(ex.iterdir()):
        m = EXPLOIT_RE.match(d.name)
        j = read_json(d / "exploit.json") if d.is_dir() and m else {}
        if j.get("curve"):
            curves[(j.get("frozen_role", d.name.split("_")[0]), m.group(1))].append(
                np.array(j["curve"], float).T)
    for f in sorted(ex.glob("exploitability_*_s*.csv")):
        c = read_csv(f)
        for i in range(len(c.get("label", []))):
            role = str(c["frozen_role"][i])
            bars[(role, str(c["label"][i]))].append(c["samples_to_threshold"][i])
            budget[role] = max(budget.get(role, 0), c["steps"][i])
    return curves, bars, budget


def sec_exploit(root):
    curves, bars, budget = load_exploit(root)
    if not curves and not bars:
        return placeholder("Exploitability phase not run yet.")
    out = ""
    roles = ("hunter", "evader")
    if curves:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
        for ax, fr in zip(axes, roles):
            for (r, lab), cs in sorted(curves.items()):
                if r == fr:
                    band(ax, cs, lab, lab)
            other = "evader" if fr == "hunter" else "hunter"
            finish(ax, f"Fresh {other} trained against frozen {fr}",
                   xlabel="fresh learner samples", ylabel=f"fresh {other} val win rate")
        fig.tight_layout()
        out += figure_html(b64(fig), "exploit curves", (
            "A fresh opponent is trained against each frozen final policy (one curve per source "
            "run type). The higher and faster the fresh learner's win rate rises, the more "
            "exploitable the frozen policy is."))
    if bars:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
        for ax, fr in zip(axes, roles):
            labs = [lab for (r, lab) in sorted(bars) if r == fr]
            for i, lab in enumerate(labs):
                v = np.array(bars[(fr, lab)], float)
                b = budget[fr]
                full = np.where(np.isnan(v), b, v)
                miss = np.isnan(v).any()
                ax.bar(i, full.mean(), yerr=full.std() if len(full) > 1 else 0, capsize=3,
                       color=color_of(lab), edgecolor="#222", hatch="//" if miss else None, alpha=0.9)
                ax.text(i, full.mean(), f"{int((~np.isnan(v)).sum())}/{len(v)}", ha="center",
                        va="bottom", fontsize=7)
            if labs:
                ax.axhline(budget[fr], color="#888", ls=":", lw=1, label="budget (not reached)")
            ax.set_xticks(range(len(labs)), labs, rotation=30, ha="right", fontsize=8)
            other = "evader" if fr == "hunter" else "hunter"
            finish(ax, f"Samples for fresh {other} to beat frozen {fr}", xlabel="frozen policy from",
                   ylabel="samples to threshold", unit=False)
        fig.tight_layout()
        out += figure_html(b64(fig), "exploit bars", (
            "Samples the fresh opponent needs to reach the win-rate threshold (mean &plusmn; std). "
            "Hatched bars: at least one seed never reached it and is drawn at the full budget; "
            "labels show seeds that reached / total. Taller bars mean a more robust frozen policy."))
    return out


def decision_section(runs, root, seed, max_maps=3):
    from agents.factory import make_agent
    from env.dungeon_env import EVADER, HUNTER, DungeonPursuitEnv, EnvConfig
    from visualization.decision_map import plot_decision_maps
    pref = ["selfplay_cur-adaptive", "independent", "hunter_vs_path", "selfplay", "evader_vs_chase",
            "independent_cur-adaptive", "hunter_vs_path_cur-adaptive", "dqn_hunter_vs_path"]
    chosen = []
    for t in pref:
        for d in runs.get(t, [])[:1]:
            if (d / "hunter.pt").exists() or (d / "evader.pt").exists():
                chosen.append((t, d))
    chosen = chosen[:max_maps]
    if not chosen:
        return placeholder("No trained models (hunter.pt / evader.pt) found yet.")
    map_seed = 10_000 + seed
    out = ""
    for t, d in chosen:
        try:
            env = DungeonPursuitEnv(EnvConfig())
            env.reset(seed=map_seed)
            hp, ep = d / "hunter.pt", d / "evader.pt"
            agents = {HUNTER: make_agent(f"ppo:{hp}") if hp.exists() else make_agent("chase"),
                      EVADER: make_agent(f"ppo:{ep}") if ep.exists() else make_agent("path")}
            roles = tuple(r for r, p in ((HUNTER, hp), (EVADER, ep)) if p.exists())
            opp = {r: ("learned " + ("evader" if r == HUNTER else "hunter") if (ep if r == HUNTER else hp).exists()
                       else "scripted " + ("path" if r == HUNTER else "chase")) for r in roles}
            with tempfile.TemporaryDirectory() as tmp:
                plot_decision_maps(env, agents, roles, Path(tmp) / "m.png", map_seed, 0)
                png = base64.b64encode((Path(tmp) / "m.png").read_bytes()).decode()
            out += figure_html(png, d.name, (
                f"<b>{d.name}</b> on val map seed {map_seed}; opponent fixed at its start position "
                f"({', '.join(f'{r}: {o}' for r, o in opp.items())}). Left: value V(s) (DQN: max Q) "
                "with greedy action arrows (opacity = action probability); right: policy entropy. "
                "A sensible hunter's value rises near the evader and the exit; an evader's value "
                "falls near the hunter."))
        except Exception as e:
            out += placeholder(f"{d.name}: decision map failed ({html.escape(repr(e))})")
    return out


def sec_dqn(runs):
    pairs = [("hunter", "hunter_vs_path", "dqn_hunter_vs_path"),
             ("evader", "evader_vs_chase", "dqn_evader_vs_chase")]
    if not any(t in runs for _, a, b in pairs for t in (a, b) if t.startswith("dqn")):
        return placeholder("No DQN runs found yet.")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    rows = []
    for ax, (role, ppo, dqn) in zip(axes, pairs):
        for t, lab in ((ppo, "PPO"), (dqn, "DQN")):
            band(ax, val_curves(runs, t, REF_KEY[role]), t, lab)
            sums = [s for s in summaries(runs, t) if s]
            if sums:
                reach = []
                for c in val_curves(runs, t, REF_KEY[role]):
                    hit = np.nonzero(c[1] >= 0.5)[0]
                    reach.append(c[0][hit[0]] if len(hit) else np.nan)
                g = [s.get("generalization", {}).get(role, {}) for s in sums]
                rows.append([role, lab, t, len(sums)]
                            + [ms([x.get(sp, {}).get("win") for x in g]) for sp in ("val", "test", "ood")]
                            + [ms([s.get("minutes") for s in sums], 1),
                               ms(reach, 0) + f" ({int((~np.isnan(reach)).sum())}/{len(reach)})"])
        finish(ax, f"{role.capitalize()}: PPO vs DQN")
    fig.tight_layout()
    out = figure_html(b64(fig), "dqn vs ppo", (
        "Same task and opponent, different learner. Compare sample efficiency (how early the "
        "curve rises) and the final plateau."))
    return out + table(["role", "algo", "run type", "seeds", "val", "test", "OOD", "minutes",
                        "samples to 0.5 val win (reached)"], rows)


def build(root, out, decision_seed):
    runs = discover(root)
    n = sum(len(v) for v in runs.values())
    secs = [("Overview", sec_overview(runs)),
            ("Training curves: curriculum off vs adaptive", sec_curriculum(runs)),
            ("Independent vs self-play stability", sec_stability(runs)),
            ("Generalization to held-out maps", sec_generalization(runs)),
            ("Exploitability", sec_exploit(root)),
            ("Decision heatmaps", decision_section(runs, root, decision_seed)),
            ("DQN vs PPO", sec_dqn(runs))]
    body = "".join(f"<section><h2>{i}. {t}</h2>{h}</section>" for i, (t, h) in enumerate(secs, 1))
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>Hunter vs Evader dashboard</title><style>{CSS}</style></head><body><main>"
            f"<h1>Hunter vs Evader: experiment dashboard</h1>"
            f'<p class="meta">{n} runs in {html.escape(str(root))}: '
            + ", ".join(f"{t} x{len(v)}" for t, v in runs.items()) + f"</p>{body}</main></body></html>")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print("saved", out, f"({out.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="experiments")
    ap.add_argument("--out", default="experiments/dashboard.html")
    ap.add_argument("--decision-seed", type=int, default=42,
                    help="offset into the val map range (map seed = 10000 + offset)")
    a = ap.parse_args()
    build(a.root, a.out, a.decision_seed)


if __name__ == "__main__":
    main()
