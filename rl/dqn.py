from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from rl.nets import QNet


@dataclass
class DQNConfig:
    lr: float = 2.5e-4
    gamma: float = 0.99
    buffer_size: int = 200_000
    batch_size: int = 256
    learning_starts: int = 20_000
    train_every: int = 1          # vec-env steps between gradient steps
    grad_steps: int = 2           # gradient steps per training call
    target_every: int = 2_000     # gradient steps between hard target syncs
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_frac: float = 0.3         # fraction of training over which eps decays linearly
    max_grad_norm: float = 10.0
    double: bool = True
    hidden: list = field(default_factory=lambda: [256, 256])


class ReplayBuffer:
    """Ring buffer. Observations are stored as float16 (values are in [-1, 1])."""

    def __init__(self, size, obs_dim, seed=0):
        self.size, self.n, self.i = size, 0, 0
        self.obs = np.zeros((size, obs_dim), np.float16)
        self.next_obs = np.zeros((size, obs_dim), np.float16)
        self.actions = np.zeros(size, np.int64)
        self.rewards = np.zeros(size, np.float32)
        self.dones = np.zeros(size, np.float32)
        self.rng = np.random.default_rng(seed)

    def add(self, obs, actions, rewards, next_obs, dones):
        for k in range(len(actions)):
            j = self.i
            self.obs[j], self.next_obs[j] = obs[k], next_obs[k]
            self.actions[j], self.rewards[j], self.dones[j] = actions[k], rewards[k], dones[k]
            self.i = (self.i + 1) % self.size
            self.n = min(self.n + 1, self.size)

    def sample(self, b):
        idx = self.rng.integers(self.n, size=b)
        return (self.obs[idx].astype(np.float32), self.actions[idx], self.rewards[idx],
                self.next_obs[idx].astype(np.float32), self.dones[idx])


class DQN:
    def __init__(self, obs_dim, n_actions, cfg: DQNConfig | None = None, device="cpu", seed=0):
        self.cfg = cfg or DQNConfig()
        self.device = torch.device(device)
        self.n_actions = n_actions
        self.q = QNet(obs_dim, n_actions, self.cfg.hidden).to(self.device)
        self.target = QNet(obs_dim, n_actions, self.cfg.hidden).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=self.cfg.lr)
        self.buf = ReplayBuffer(self.cfg.buffer_size, obs_dim, seed)
        self.rng = np.random.default_rng(seed)
        self.n_grad = 0

    def epsilon(self, progress):
        c = self.cfg
        f = min(1.0, progress / c.eps_frac)
        return c.eps_start + f * (c.eps_end - c.eps_start)

    @torch.no_grad()
    def act(self, obs, eps):
        a = self.q(torch.as_tensor(obs, device=self.device)).argmax(-1).cpu().numpy()
        rand = self.rng.random(len(a)) < eps
        a[rand] = self.rng.integers(self.n_actions, size=int(rand.sum()))
        return a

    def train_step(self):
        c = self.cfg
        o, a, r, o2, d = (torch.as_tensor(x, device=self.device)
                          for x in self.buf.sample(c.batch_size))
        q = self.q(o).gather(-1, a.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            if c.double:
                a2 = self.q(o2).argmax(-1, keepdim=True)
                q2 = self.target(o2).gather(-1, a2).squeeze(-1)
            else:
                q2 = self.target(o2).max(-1).values
            y = r + c.gamma * (1 - d) * q2
        loss = F.smooth_l1_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), c.max_grad_norm)
        self.opt.step()
        self.n_grad += 1
        if self.n_grad % c.target_every == 0:
            self.target.load_state_dict(self.q.state_dict())
        return loss.item(), q.mean().item()
