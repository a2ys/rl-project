import numpy as np
import torch
import torch.nn as nn


def mlp(sizes, act=nn.Tanh, out_gain=1.0):
    layers = []
    for i in range(len(sizes) - 1):
        lin = nn.Linear(sizes[i], sizes[i + 1])
        last = i == len(sizes) - 2
        nn.init.orthogonal_(lin.weight, out_gain if last else np.sqrt(2))
        nn.init.zeros_(lin.bias)
        layers.append(lin)
        if not last:
            layers.append(act())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Separate policy and value MLPs (no shared trunk)."""

    def __init__(self, obs_dim, n_actions, hidden=(256, 256)):
        super().__init__()
        self.obs_dim, self.n_actions, self.hidden = obs_dim, n_actions, list(hidden)
        self.pi = mlp([obs_dim, *hidden, n_actions], out_gain=0.01)
        self.v = mlp([obs_dim, *hidden, 1], out_gain=1.0)

    def logits(self, obs):
        return self.pi(obs)

    def value(self, obs):
        return self.v(obs).squeeze(-1)

    def spec(self):
        return {"arch": "actor_critic", "obs_dim": self.obs_dim,
                "n_actions": self.n_actions, "hidden": self.hidden}


class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=(256, 256)):
        super().__init__()
        self.obs_dim, self.n_actions, self.hidden = obs_dim, n_actions, list(hidden)
        self.q = mlp([obs_dim, *hidden, n_actions], act=nn.ReLU)

    def forward(self, obs):
        return self.q(obs)

    def spec(self):
        return {"arch": "qnet", "obs_dim": self.obs_dim,
                "n_actions": self.n_actions, "hidden": self.hidden}


ARCHS = {"actor_critic": ActorCritic, "qnet": QNet}


def build(spec):
    return ARCHS[spec["arch"]](spec["obs_dim"], spec["n_actions"], spec["hidden"])


def save_checkpoint(path, algo, net, extra=None):
    torch.save({"algo": algo, "spec": net.spec(),
                "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                "extra": extra or {}}, path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=True)
    net = build(ck["spec"]).to(device)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return ck["algo"], net, ck.get("extra", {})


def sample_categorical(logits, deterministic=False, generator=None):
    """Returns actions and their log-probabilities."""
    logp_all = torch.log_softmax(logits, dim=-1)
    if deterministic:
        a = logits.argmax(-1)
    else:
        a = torch.multinomial(logp_all.exp(), 1, generator=generator).squeeze(-1)
    return a, logp_all.gather(-1, a.unsqueeze(-1)).squeeze(-1)
