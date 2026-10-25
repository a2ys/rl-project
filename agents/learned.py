import numpy as np
import torch

from agents.scripted import Agent
from rl.nets import load_checkpoint, sample_categorical


class LearnedAgent(Agent):
    """A trained PPO (actor-critic) or DQN (Q-network) checkpoint.

PPO agents sample from their policy by default: under partial observability the argmax
policy tends to oscillate and time out (greedy ~0.45 vs sampled ~0.70 win rate).

    probs(obs) -> action distribution (DQN: softmax over Q-values)
    value(obs) -> state value (DQN: max Q)
    q_values(obs) -> per-action Q-values (DQN only)
    """

    def __init__(self, path_or_net, deterministic=False, algo="ppo", seed=0):
        if isinstance(path_or_net, str):
            self.algo, self.net, self.extra = load_checkpoint(path_or_net)
        else:
            self.algo, self.net, self.extra = algo, path_or_net, {}
        self.deterministic = deterministic
        self.seed = seed
        self.gen = torch.Generator().manual_seed(seed)

    def reset(self):
        # re-seeded per evaluation so stochastic play is reproducible
        self.gen.manual_seed(self.seed)

    @torch.no_grad()
    def _logits(self, obs):
        o = torch.as_tensor(np.asarray(obs, np.float32))
        return self.net(o) if self.algo == "dqn" else self.net.logits(o)

    def act_batch(self, obs):
        logits = self._logits(obs)
        if self.algo == "dqn":
            return logits.argmax(-1).numpy()
        a, _ = sample_categorical(logits, self.deterministic, self.gen)
        return a.numpy()

    def act(self, env, role):
        return int(self.act_batch(env.observe(role)[None])[0])

    def probs(self, obs):
        return torch.softmax(self._logits(obs[None])[0], -1).numpy()

    @torch.no_grad()
    def value(self, obs):
        o = torch.as_tensor(np.asarray(obs, np.float32)[None])
        if self.algo == "dqn":
            return float(self.net(o).max())
        return float(self.net.value(o)[0])

    def q_values(self, obs):
        assert self.algo == "dqn"
        return self._logits(obs[None])[0].numpy()
