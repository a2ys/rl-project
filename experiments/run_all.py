"""Runs the full experiment matrix of the project (all phases, every seed).

  python -m experiments.run_all --dry-run              # print commands and time estimate
  python -m experiments.run_all --seeds 0 1 2 --steps 2000000
  python -m experiments.run_all --phase baselines      # one phase only

Phases
  baselines     PPO single-role vs scripted opponents, curriculum off / adaptive; DQN baselines
  adversarial   independent vs self-play, curriculum off / adaptive
  crossplay     checkpoint cross-play matrices of the adversarial runs (stability)
  exploit       freeze each final Hunter / Evader and train a fresh opponent against it

Existing runs (summary.json present) are skipped, so the script can be resumed.
"""
import argparse
import subprocess
import sys
from pathlib import Path

PY = sys.executable

# measured on an i5-13500H (12 cores), 32 envs / 8 workers, CPU torch:
# minutes per 1M samples per learner (incl. periodic eval)
MIN_PER_M = {"single": 5.5, "independent": 4.5, "selfplay": 14.0, "dqn": 15.0}


def run_dir(name, seed):
    return Path("experiments") / f"{name}_s{seed}"


def jobs(seeds, steps, exploit_steps):
    out = []
    for s in seeds:
        for cur in ("off", "adaptive"):
            suf = "" if cur == "off" else "_cur-adaptive"
            for role, opp in (("hunter", "path"), ("evader", "chase")):
                name = f"{role}_vs_{opp}{suf}"
                out.append(("baselines", "single", run_dir(name, s), steps,
                            ["-m", "training.train", "--mode", "single", "--role", role,
                             "--curriculum", cur, "--steps", str(steps), "--seed", str(s),
                             "--out", str(run_dir(name, s))]))
            for mode in ("independent", "selfplay"):
                name = f"{mode}{suf}"
                out.append(("adversarial", mode, run_dir(name, s), steps,
                            ["-m", "training.train", "--mode", mode, "--curriculum", cur,
                             "--steps", str(steps), "--seed", str(s),
                             "--out", str(run_dir(name, s))]))
        for role, opp in (("hunter", "path"), ("evader", "chase")):
            name = f"dqn_{role}_vs_{opp}"
            out.append(("baselines", "dqn", run_dir(name, s), steps,
                        ["-m", "training.train_dqn", "--role", role, "--steps", str(steps),
                         "--seed", str(s), "--out", str(run_dir(name, s))]))
        for name in ("independent", "selfplay", "independent_cur-adaptive",
                     "selfplay_cur-adaptive"):
            d = run_dir(name, s)
            out.append(("crossplay", None, d / "crossplay.csv", 0,
                        ["-m", "evaluation.crossplay", str(d)]))
        for frozen, single in (("hunter", "hunter_vs_path"), ("evader", "evader_vs_chase")):
            names = [single, "independent", "selfplay", "independent_cur-adaptive",
                     "selfplay_cur-adaptive"]
            models = [str(run_dir(n, s) / f"{frozen}.pt") for n in names]
            marker = Path("experiments/exploitability") / f"exploitability_{frozen}_s{s}.csv"
            out.append(("exploit", "single", marker, exploit_steps * len(names),
                        ["-m", "evaluation.exploitability", "--frozen-role", frozen,
                         "--models", *models, "--labels", *names,
                         "--steps", str(exploit_steps), "--seed", str(s)]))
    return out


def done(target):
    return (target / "summary.json").exists() if target.suffix == "" else target.exists()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=2_000_000, help="samples per learner")
    ap.add_argument("--exploit-steps", type=int, default=500_000)
    ap.add_argument("--phase", default=None,
                    choices=["baselines", "adversarial", "crossplay", "exploit"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    todo = [j for j in jobs(args.seeds, args.steps, args.exploit_steps)
            if (args.phase is None or j[0] == args.phase) and not done(j[2])]
    est = sum(MIN_PER_M.get(kind, 0) * n / 1e6 for _, kind, _, n, _ in todo if kind)
    est += 3 * sum(1 for j in todo if j[0] == "crossplay")
    print(f"{len(todo)} jobs, estimated {est / 60:.1f} h on the reference machine")
    for phase, _, target, _, cmd in todo:
        print(f"[{phase}] python {' '.join(cmd)}", flush=True)
        if not args.dry_run:
            subprocess.run([PY, *cmd], check=True)


if __name__ == "__main__":
    main()
