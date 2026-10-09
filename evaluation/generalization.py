import argparse

from agents.factory import make_agent
from agents.ppo_agent import SB3Agent
from env.dungeon_env import EnvConfig
from evaluation.evaluate import evaluate, split_seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["hunter", "evader"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--opponent", default=None)
    ap.add_argument("--n-maps", type=int, default=100)
    args = ap.parse_args()
    opp = args.opponent or ("path" if args.role == "hunter" else "chase")
    cfg = EnvConfig()

    print(f"{'split':<8}{'maps':>6}{'win':>8}{'timeout':>9}{'len':>8}")
    wins = {}
    for split in ("train", "val", "test"):
        agent = SB3Agent(args.model)
        if args.role == "hunter":
            st = evaluate(agent, make_agent(opp), split_seeds(split, args.n_maps), cfg)
            w = st["hunter_win"]
        else:
            st = evaluate(make_agent(opp), agent, split_seeds(split, args.n_maps), cfg)
            w = st["evader_win"]
        wins[split] = w
        print(f"{split:<8}{st['n']:>6}{w:>8.2f}{st['timeout']:>9.2f}{st['mean_len']:>8.1f}")
    print(f"generalization gap (train - test) = {wins['train'] - wins['test']:+.2f}")


if __name__ == "__main__":
    main()
