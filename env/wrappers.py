"""Single-agent Gymnasium view of the two-player env: the opponent is a scripted
(or frozen) agent from agents.*, so SB3 can train one role at a time."""
import gymnasium as gym
import numpy as np

from env.dungeon_env import EVADER, HUNTER, DungeonPursuitEnv, EnvConfig


class SingleAgentEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, role, opponent, cfg=None, seeds=None, rng_seed=0):
        super().__init__()
        assert role in (HUNTER, EVADER)
        self.role = role
        self.other = EVADER if role == HUNTER else HUNTER
        self.opponent = opponent
        self.env = DungeonPursuitEnv(cfg or EnvConfig(), seeds=seeds, rng_seed=rng_seed)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.env.reset(seed=seed, options=options)
        self.opponent.reset()
        return obs[self.role], info

    def step(self, action):
        opp = self.opponent.act(self.env, self.other)
        obs, rew, term, trunc, info = self.env.step(
            {self.role: int(action), self.other: opp})
        return obs[self.role], float(rew[self.role]), term, trunc, info
