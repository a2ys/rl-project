"""Train a PPO baseline for one role against a scripted opponent.

python -m training.train --role hunter --steps 300000      # vs scripted 'path' Evader
python -m training.train --role evader --steps 300000      # vs scripted 'chase' Hunter
python -m training.train --role hunter --steps 20000 --eval-freq 10000   # smoke test
"""
import argparse
import csv
import json
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from agents.factory import make_agent
from agents.ppo_agent import SB3Agent
from env.dungeon_env import EVADER, HUNTER, EnvConfig
from env.generator import SEED_SPLITS
from env.wrappers import SingleAgentEnv
from evaluation.evaluate import evaluate

WIN_OUTCOME = {HUNTER: "caught", EVADER: "escaped"}


def make_env(role, opp_spec, seeds, cfg, rng_seed):
    def _init():
        return Monitor(SingleAgentEnv(role, make_agent(opp_spec), cfg=cfg,
                                      seeds=seeds, rng_seed=rng_seed))
    return _init


def role_eval(model, role, opp_spec, seeds, cfg):
    agent = SB3Agent(model)
    if role == HUNTER:
        st = evaluate(agent, make_agent(opp_spec), seeds, cfg)
        return st, st["hunter_win"]
    st = evaluate(make_agent(opp_spec), agent, seeds, cfg)
    return st, st["evader_win"]


class LogCallback(BaseCallback):
    def __init__(self, role, opp, cfg, val_seeds, eval_freq, out_dir):
        super().__init__()
        self.role, self.opp, self.cfg = role, opp, cfg
        self.val_seeds, self.eval_freq, self.out = val_seeds, eval_freq, Path(out_dir)
        self.next_eval = eval_freq
        self.recent = deque(maxlen=100)

    def _on_training_start(self):
        self.f_ep = open(self.out / "episodes.csv", "w", newline="")
        self.w_ep = csv.writer(self.f_ep)
        self.w_ep.writerow(["timesteps", "reward", "length", "win", "outcome"])
        self.f_val = open(self.out / "val.csv", "w", newline="")
        self.w_val = csv.writer(self.f_val)
        self.w_val.writerow(["timesteps", "win_rate", "timeout_rate", "mean_len"])

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                ep = info.get("episode")
                outcome = info.get("outcome", "")
                win = int(outcome == WIN_OUTCOME[self.role])
                self.recent.append(win)
                self.w_ep.writerow([self.num_timesteps,
                                    round(float(ep["r"]), 4) if ep else "",
                                    int(ep["l"]) if ep else "", win, outcome])
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.eval_freq
            st, w = role_eval(self.model, self.role, self.opp, self.val_seeds, self.cfg)
            self.w_val.writerow([self.num_timesteps, round(w, 4),
                                 round(st["timeout"], 4), round(st["mean_len"], 2)])
            self.f_val.flush()
            self.f_ep.flush()
            tr = sum(self.recent) / max(1, len(self.recent))
            print(f"[{self.num_timesteps:>9}] train-win(last100)={tr:.2f}  "
                  f"val-win={w:.2f}  val-timeout={st['timeout']:.2f}  "
                  f"val-len={st['mean_len']:.1f}", flush=True)
        return True

    def _on_training_end(self):
        self.f_ep.close()
        self.f_val.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["hunter", "evader"])
    ap.add_argument("--opponent", default=None)
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-freq", type=int, default=25_000)
    ap.add_argument("--n-eval-maps", type=int, default=50)
    ap.add_argument("--n-doors", type=int, default=1)
    ap.add_argument("--shaping", type=float, default=0.01)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--subproc", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    role = args.role
    opp = args.opponent or ("path" if role == HUNTER else "chase")
    cfg = EnvConfig(n_doors=args.n_doors, shaping=args.shaping)
    out = Path(args.out or f"experiments/{role}_vs_{opp}_s{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(
        {"args": vars(args), "env": asdict(cfg), "opponent": opp}, indent=2))

    set_random_seed(args.seed)
    train_seeds = list(SEED_SPLITS["train"])
    val_seeds = list(SEED_SPLITS["val"])[:args.n_eval_maps]

    fns = [make_env(role, opp, train_seeds, cfg, args.seed * 1000 + i)
           for i in range(args.n_envs)]
    venv = SubprocVecEnv(fns) if args.subproc else DummyVecEnv(fns)

    model = PPO("MlpPolicy", venv, n_steps=256, batch_size=256, n_epochs=10,
                learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, ent_coef=args.ent_coef,
                policy_kwargs=dict(net_arch=[256, 256]), device="cpu", verbose=0)

    cb = LogCallback(role, opp, cfg, val_seeds, args.eval_freq, out)
    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=cb)
    model.save(out / "model")

    st_tr, w_tr = role_eval(model, role, opp, train_seeds[:args.n_eval_maps], cfg)
    st_va, w_va = role_eval(model, role, opp, val_seeds, cfg)
    summary = {"role": role, "opponent": opp, "steps": args.steps,
               "minutes": round((time.time() - t0) / 60, 2),
               "train_maps_win": w_tr, "val_maps_win": w_va,
               "train_stats": st_tr, "val_stats": st_va}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("saved to", out)


if __name__ == "__main__":
    main()
