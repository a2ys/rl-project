"""Train / val / test / OOD win rates of one agent vs a fixed opponent.

  python -m evaluation.generalization --role hunter --agent ppo:experiments/selfplay_s0/hunter.pt
"""
import argparse

from agents.factory import make_agent
from env.dungeon_env import EnvConfig
from evaluation.evaluate import evaluate, split_seeds
from rl.curriculum import ood_cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", required=True, choices=["hunter", "evader"])
    ap.add_argument("--agent", required=True, help="agent spec, e.g. ppo:<model.pt>")
    ap.add_argument("--opponent", default=None, help="default path (hunter) / chase (evader)")
    ap.add_argument("--n-maps", type=int, default=200)
    args = ap.parse_args()
    opp = args.opponent or ("path" if args.role == "hunter" else "chase")
    me = make_agent(args.agent)

    print(f"{'split':<8}{'maps':>6}{'win':>8}{'timeout':>9}{'len':>8}")
    wins = {}
    for split, cfg in (("train", EnvConfig()), ("val", EnvConfig()), ("test", EnvConfig()),
                       ("ood", ood_cfg())):
        seeds = split_seeds("test" if split == "ood" else split, args.n_maps)
        if args.role == "hunter":
            st = evaluate(me, make_agent(opp), seeds, cfg)
            w = st["hunter_win"]
        else:
            st = evaluate(make_agent(opp), me, seeds, cfg)
            w = st["evader_win"]
        wins[split] = w
        print(f"{split:<8}{st['n']:>6}{w:>8.2f}{st['timeout']:>9.2f}{st['mean_len']:>8.1f}")
    print(f"generalization gap (train - test) = {wins['train'] - wins['test']:+.2f}")
    print(f"OOD drop (test - ood)             = {wins['test'] - wins['ood']:+.2f}")


if __name__ == "__main__":
    main()
