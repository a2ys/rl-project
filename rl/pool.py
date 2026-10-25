import copy

import numpy as np
import torch

from rl.nets import sample_categorical


class FrozenPolicy:
    """Inference-only copy of a policy (used as an opponent)."""

    def __init__(self, net, tag="", deterministic=False):
        self.net = copy.deepcopy(net).cpu().eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.tag, self.deterministic = tag, deterministic

    @torch.no_grad()
    def act(self, obs):
        a, _ = sample_categorical(self.net.logits(torch.as_tensor(obs)), self.deterministic)
        return a.numpy()


class OpponentPool:
    """Frozen snapshots of one role. Sampling: the latest snapshot with probability
    `p_latest`, otherwise uniform over the whole pool. Oldest snapshots are dropped."""

    def __init__(self, max_size=20, p_latest=0.5, seed=0):
        self.items, self.max_size, self.p_latest = [], max_size, p_latest
        self.rng = np.random.default_rng(seed)

    def add(self, net, tag):
        self.items.append(FrozenPolicy(net, tag))
        if len(self.items) > self.max_size:
            self.items.pop(0)

    def sample(self):
        if self.rng.random() < self.p_latest:
            return self.items[-1]
        return self.items[int(self.rng.integers(len(self.items)))]

    def __len__(self):
        return len(self.items)
