from dataclasses import dataclass, field

import numpy as np
import torch

from rl.nets import ActorCritic, sample_categorical


@dataclass
class PPOConfig:
    lr: float = 3e-4
    min_lr_frac: float = 0.1
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 512
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03
    hidden: list = field(default_factory=lambda: [256, 256])


class RolloutBuffer:
    """Fixed-length rollout of T steps over N parallel envs."""

    def __init__(self, T, N, obs_dim):
        self.T, self.N = T, N
        self.obs = np.zeros((T, N, obs_dim), np.float32)
        self.actions = np.zeros((T, N), np.int64)
        self.logp = np.zeros((T, N), np.float32)
        self.values = np.zeros((T, N), np.float32)
        self.rewards = np.zeros((T, N), np.float32)
        self.dones = np.zeros((T, N), np.float32)
        self.t = 0

    def add(self, obs, actions, logp, values, rewards, dones):
        t = self.t
        self.obs[t], self.actions[t], self.logp[t] = obs, actions, logp
        self.values[t], self.rewards[t], self.dones[t] = values, rewards, dones
        self.t += 1

    def full(self):
        return self.t == self.T

    def compute_gae(self, last_values, gamma, lam):
        # dones[t] = the episode ended after step t, so step t does not bootstrap.
        # Timeouts are treated as terminal: the time fraction is part of the observation.
        adv = np.zeros_like(self.rewards)
        gae = np.zeros(self.N, np.float32)
        next_v = last_values
        for t in reversed(range(self.T)):
            nonterm = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_v * nonterm - self.values[t]
            gae = delta + gamma * lam * nonterm * gae
            adv[t] = gae
            next_v = self.values[t]
        return adv, adv + self.values


class PPO:
    def __init__(self, obs_dim, n_actions, cfg: PPOConfig | None = None, device="cpu"):
        self.cfg = cfg or PPOConfig()
        self.device = torch.device(device)
        self.net = ActorCritic(obs_dim, n_actions, self.cfg.hidden).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr, eps=1e-5)

    def _t(self, x):
        return torch.as_tensor(x, device=self.device)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        o = self._t(obs)
        a, logp = sample_categorical(self.net.logits(o), deterministic)
        v = self.net.value(o)
        return a.cpu().numpy(), logp.cpu().numpy(), v.cpu().numpy()

    @torch.no_grad()
    def value(self, obs):
        return self.net.value(self._t(obs)).cpu().numpy()

    def update(self, buf: RolloutBuffer, last_values, progress=0.0):
        c = self.cfg
        lr = c.lr * max(c.min_lr_frac, 1.0 - progress) if c.anneal_lr else c.lr
        for g in self.opt.param_groups:
            g["lr"] = lr

        adv, ret = buf.compute_gae(last_values, c.gamma, c.gae_lambda)
        n = buf.T * buf.N
        obs = self._t(buf.obs.reshape(n, -1))
        act = self._t(buf.actions.reshape(n))
        old_logp = self._t(buf.logp.reshape(n))
        adv_t = self._t(adv.reshape(n))
        ret_t = self._t(ret.reshape(n))

        stats = {"pg_loss": [], "v_loss": [], "entropy": [], "approx_kl": [], "clipfrac": []}
        epochs_done, stop = 0, False
        for _ in range(c.epochs):
            perm = torch.randperm(n, device=self.device)
            for s in range(0, n, c.minibatch):
                idx = perm[s:s + c.minibatch]
                logp_all = torch.log_softmax(self.net.logits(obs[idx]), dim=-1)
                logp = logp_all.gather(-1, act[idx].unsqueeze(-1)).squeeze(-1)
                log_ratio = logp - old_logp[idx]
                ratio = log_ratio.exp()

                a = adv_t[idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                pg_loss = -torch.min(ratio * a,
                                     ratio.clamp(1 - c.clip, 1 + c.clip) * a).mean()
                v_loss = 0.5 * (self.net.value(obs[idx]) - ret_t[idx]).pow(2).mean()
                entropy = -(logp_all.exp() * logp_all).sum(-1).mean()
                loss = pg_loss + c.vf_coef * v_loss - c.ent_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), c.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    kl = ((ratio - 1) - log_ratio).mean().item()
                    clipfrac = ((ratio - 1).abs() > c.clip).float().mean().item()
                for k, v in (("pg_loss", pg_loss.item()), ("v_loss", v_loss.item()),
                             ("entropy", entropy.item()), ("approx_kl", kl),
                             ("clipfrac", clipfrac)):
                    stats[k].append(v)
                if c.target_kl is not None and kl > 1.5 * c.target_kl:
                    stop = True
                    break
            epochs_done += 1
            if stop:
                break

        var_y = np.var(ret)
        ev = float("nan") if var_y == 0 else float(1 - np.var(ret - buf.values) / var_y)
        out = {k: float(np.mean(v)) for k, v in stats.items()}
        out.update(lr=lr, explained_var=ev, epochs=epochs_done)
        buf.t = 0
        return out
