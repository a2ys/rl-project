import csv
import json

import numpy as np
import pytest
import torch

from agents.factory import make_agent
from agents.learned import LearnedAgent
from agents.scripted import NoisyAgent
from env.dungeon_env import EVADER, HUNTER, DungeonPursuitEnv, EnvConfig
from env.generator import SEED_SPLITS
from evaluation.evaluate import evaluate
from rl.curriculum import LEVELS, OOD, Curriculum, level_cfg, ood_cfg
from rl.nets import ActorCritic, QNet, load_checkpoint, sample_categorical, save_checkpoint
from rl.pool import FrozenPolicy, OpponentPool
from rl.ppo import PPO, PPOConfig, RolloutBuffer
from rl.vec_env import make_vec_env


def ref_gae(r, v, d, last, gamma, lam):
    T, N = r.shape
    adv = np.zeros((T, N))
    for n in range(N):
        for t in range(T):
            a, disc = 0.0, 1.0
            for k in range(t, T):
                nv = v[k + 1, n] if k + 1 < T else last[n]
                delta = r[k, n] + gamma * nv * (1 - d[k, n]) - v[k, n]
                a += disc * delta
                disc *= gamma * lam * (1 - d[k, n])
            adv[t, n] = a
    return adv


def fill(T, N, D, rng):
    b = RolloutBuffer(T, N, D)
    for _ in range(T):
        b.add(rng.normal(size=(N, D)), rng.integers(0, 3, N), np.zeros(N),
              rng.normal(size=N), rng.normal(size=N), rng.random(N) < 0.2)
    return b


def test_gae_matches_reference():
    rng = np.random.default_rng(0)
    b = fill(12, 3, 4, rng)
    last = rng.normal(size=3).astype(np.float32)
    adv, ret = b.compute_gae(last, 0.97, 0.9)
    ref = ref_gae(b.rewards, b.values, b.dones, last, 0.97, 0.9)
    np.testing.assert_allclose(adv, ref, atol=1e-4)
    np.testing.assert_allclose(ret, ref + b.values, atol=1e-4)


def test_gae_hand_checked():
    b = RolloutBuffer(3, 1, 1)
    for _ in range(3):
        b.add(np.zeros((1, 1)), np.zeros(1), np.zeros(1), np.zeros(1), np.ones(1), np.zeros(1))
    adv, _ = b.compute_gae(np.array([0.0], np.float32), 0.9, 1.0)
    np.testing.assert_allclose(adv[:, 0], [1 + .9 + .81, 1 + .9, 1], atol=1e-6)
    b.dones[1, 0] = 1
    adv, _ = b.compute_gae(np.array([10.0], np.float32), 0.9, 1.0)
    np.testing.assert_allclose(adv[:, 0], [1 + .9, 1, 1 + .9 * 10], atol=1e-5)


def test_sample_categorical():
    torch.manual_seed(0)
    logits = torch.tensor([[0.1, 2.0, -1.0, 0.5, 0.0], [3.0, 0.0, 0.0, 0.0, 1.0]])
    a, lp = sample_categorical(logits, deterministic=True)
    assert a.tolist() == [1, 0]
    ls = torch.log_softmax(logits, -1)
    assert torch.allclose(lp, ls.gather(1, a[:, None])[:, 0])
    a, lp = sample_categorical(logits[:1].repeat(20000, 1))
    assert torch.allclose(lp, ls[0][a])
    freq = torch.bincount(a, minlength=5).float() / len(a)
    assert torch.allclose(freq, ls[0].exp(), atol=0.02)


def test_ppo_update_stats_and_params():
    torch.manual_seed(0)
    ppo = PPO(6, 3, PPOConfig(hidden=[16], minibatch=32))
    b = fill(16, 4, 6, np.random.default_rng(0))
    before = [p.detach().clone() for p in ppo.net.parameters()]
    st = ppo.update(b, np.zeros(4, np.float32), 0.5)
    assert {"pg_loss", "v_loss", "entropy", "approx_kl", "clipfrac", "lr",
            "explained_var", "epochs"} <= set(st)
    assert all(np.isfinite(v) for v in st.values())
    assert b.t == 0
    assert any(not torch.equal(p, q) for p, q in zip(before, ppo.net.parameters()))


def test_ppo_learns_contextual_bandit():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    cfg = PPOConfig(hidden=[16], lr=3e-3, minibatch=64, gamma=0.0, target_kl=None,
                    anneal_lr=False, ent_coef=0.0)
    ppo = PPO(2, 2, cfg)
    T, N = 8, 16
    for _ in range(30):
        b = RolloutBuffer(T, N, 2)
        for _ in range(T):
            s = rng.integers(0, 2, N)
            o = np.eye(2, dtype=np.float32)[s]
            a, lp, v = ppo.act(o)
            b.add(o, a, lp, v, (a == s).astype(np.float32), np.ones(N))
        ppo.update(b, np.zeros(N, np.float32))
    a, _, _ = ppo.act(np.eye(2, dtype=np.float32), deterministic=True)
    assert a.tolist() == [0, 1]


@pytest.mark.parametrize("algo,cls", [("ppo", ActorCritic), ("dqn", QNet)])
def test_checkpoint_roundtrip_and_learned_agent(tmp_path, algo, cls):
    torch.manual_seed(0)
    net = cls(496, 5, (8, 8))
    p = str(tmp_path / "m.pt")
    save_checkpoint(p, algo, net, {"k": 1})
    algo2, net2, extra = load_checkpoint(p)
    assert algo2 == algo and extra == {"k": 1} and isinstance(net2, cls)
    x = torch.randn(3, 496)
    f = (lambda n: n.logits(x)) if algo == "ppo" else (lambda n: n(x))
    assert torch.equal(f(net), f(net2))
    ag = LearnedAgent(p)
    obs = np.random.default_rng(0).normal(size=496).astype(np.float32)
    assert ag.probs(obs).sum() == pytest.approx(1.0, abs=1e-5)
    assert isinstance(ag.value(obs), float)
    if algo == "dqn":
        assert ag.q_values(obs).shape == (5,)
    else:
        with pytest.raises(AssertionError):
            ag.q_values(obs)
    env = DungeonPursuitEnv()
    env.reset(seed=1)
    a = ag.act(env, HUNTER)
    assert isinstance(a, int) and a in range(5)


def test_curriculum_off():
    c = Curriculum("off", 1000)
    assert c.level == len(LEVELS) - 1
    for _ in range(400):
        c.observe(1)
    assert not c.update(10 ** 6) and c.level == len(LEVELS) - 1


def test_curriculum_linear():
    c = Curriculum("linear", 3000, ramp=0.6)
    assert not c.update(599) and c.level == 0
    assert c.update(600) and c.level == 1
    assert c.update(1300) and c.level == 2
    assert c.update(1800) and c.level == 3 and c.final
    assert not c.update(3000)
    assert [h[1] for h in c.history] == [0, 1, 2, 3]


def test_curriculum_adaptive():
    c = Curriculum("adaptive", 10 ** 6, threshold=0.6, window=10, min_steps=100, max_frac=0.5)
    for _ in range(10):
        c.observe(1)
    assert not c.update(99)
    assert c.update(100) and c.level == 1 and c.history[-1] == (100, 1)
    assert len(c.recent) == 0
    for _ in range(5):
        c.observe(1)
    assert not c.update(10 ** 5)
    for _ in range(5):
        c.observe(0)
    assert not c.update(10 ** 5)
    for _ in range(10):
        c.observe(1)
    assert c.update(10 ** 5 + 1) and c.level == 2
    assert not c.update(10 ** 5 + 2)
    assert c.update(c.level_start + 500_000) and c.level == 3
    assert c.final and not c.update(10 ** 9)


@pytest.mark.parametrize("cfg", [level_cfg(l) for l in LEVELS] + [ood_cfg()])
def test_level_cfgs_observations(cfg):
    env = DungeonPursuitEnv(EnvConfig())
    env.cfg = cfg
    for s in range(3):
        obs, _ = env.reset(seed=s)
        assert env.H == env.W == cfg.size
        assert env.grid.shape == (cfg.size, cfg.size)
        for r in (HUNTER, EVADER):
            assert obs[r].shape == (496,)
            assert obs[r].min() >= -1 and obs[r].max() <= 1
        for _ in range(5):
            obs, *_ = env.step({HUNTER: 0, EVADER: 1})
            assert obs[HUNTER].shape == (496,)
    assert ood_cfg().size == OOD["size"] == 25


def test_noisy_agent():
    env = DungeonPursuitEnv()
    env.reset(seed=2)
    ref = make_agent("chase_oracle").act(env, HUNTER)
    assert all(NoisyAgent(make_agent("chase_oracle"), 0.0).act(env, HUNTER) == ref
               for _ in range(30))
    acts = [NoisyAgent(make_agent("chase_oracle"), 1.0).act(env, HUNTER) for _ in range(300)]
    assert set(acts) == set(range(5))


def test_opponent_pool_and_frozen():
    torch.manual_seed(0)
    net = ActorCritic(4, 5, (8,))
    pool = OpponentPool(max_size=3, p_latest=1.0)
    for i in range(5):
        pool.add(net, f"t{i}")
    assert len(pool) == 3 and [p.tag for p in pool.items] == ["t2", "t3", "t4"]
    assert all(pool.sample().tag == "t4" for _ in range(20))
    pool.p_latest = 0.0
    assert {pool.sample().tag for _ in range(100)} == {"t2", "t3", "t4"}
    obs = np.random.default_rng(0).normal(size=(6, 4)).astype(np.float32)
    a = FrozenPolicy(net).act(obs)
    assert a.shape == (6,) and ((a >= 0) & (a < 5)).all()
    fd = FrozenPolicy(net, deterministic=True)
    before = fd.act(obs)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(5.0)
    assert np.array_equal(fd.act(obs), before)
    assert not any(p.requires_grad for p in fd.net.parameters())


def _run(venv, acts):
    out, eps = [venv.reset()], []
    for a in acts:
        o, r, d, e = venv.step(a)
        out.append((o, r, d))
        eps += [x for x in e if x is not None]
    return out, eps


def test_sync_vs_subproc_vec_env():
    seeds = list(SEED_SPLITS["train"])[:20]
    scripted = [{"evader": "path"}, {"evader": "path", "hunter": "chase"}] * 2
    rng = np.random.default_rng(0)
    acts = []
    for _ in range(230):
        a = rng.integers(0, 5, (4, 2))
        a[:, 1] = -1
        a[1::2, 0] = -1
        acts.append(a)
    sync = make_vec_env(4, seeds, 3, scripted, n_workers=0)
    sub = make_vec_env(4, seeds, 3, scripted, n_workers=2)
    try:
        o1, e1 = _run(sync, acts)
        o2, e2 = _run(sub, acts)
        assert sub.call("set_level", 0, True) == [None] * 4
    finally:
        sub.close()
    assert np.array_equal(o1[0], o2[0])
    for x, y in zip(o1[1:], o2[1:]):
        for u, v in zip(x, y):
            assert np.array_equal(u, v)
    assert e1 == e2 and len(e1) > 0
    assert {"outcome", "length", "ret_hunter", "ret_evader", "map_seed", "level"} <= set(e1[0])
    assert e1[0]["outcome"] in ("caught", "escaped", "timeout")


def test_evaluate_deterministic_and_matches_sequential():
    seeds = list(SEED_SPLITS["val"])[:20]
    cfg = EnvConfig()
    h, e = make_agent("chase"), make_agent("path")
    r1, r2 = evaluate(h, e, seeds, cfg), evaluate(h, e, seeds, cfg)
    assert r1 == r2 and r1["n"] == 20
    env = DungeonPursuitEnv(cfg, seeds=[0])
    outs, lens, rh, re = [], [], [], []
    for s in seeds:
        env.reset(seed=s)
        h.reset()
        e.reset()
        tot, done = [0.0, 0.0], False
        while not done:
            _, rew, term, trunc, info = env.step(
                {HUNTER: h.act(env, HUNTER), EVADER: e.act(env, EVADER)})
            tot[0] += rew[HUNTER]
            tot[1] += rew[EVADER]
            done = term or trunc
        outs.append(info["outcome"])
        lens.append(env.t)
        rh.append(tot[0])
        re.append(tot[1])
    n = len(seeds)
    assert r1["hunter_win"] == outs.count("caught") / n
    assert r1["evader_win"] == outs.count("escaped") / n
    assert r1["timeout"] == outs.count("timeout") / n
    assert r1["mean_len"] == pytest.approx(np.mean(lens))
    assert r1["ret_hunter"] == pytest.approx(np.mean(rh), abs=1e-5)
    assert r1["ret_evader"] == pytest.approx(np.mean(re), abs=1e-5)


TINY = ["--steps", "2048", "--n-envs", "4", "--n-steps", "64", "--workers", "0",
        "--eval-freq", "1000000", "--n-eval-maps", "2", "--n-final-maps", "2",
        "--minibatch", "128", "--snapshot-every", "512"]


def _train(out, *extra):
    from training.train import build_parser, train
    args = build_parser().parse_args([*TINY, "--out", str(out), *extra])
    return train(args)


def _check_outputs(out, ckpts, summary):
    for f in ckpts + ["episodes.csv", "updates.csv", "eval.csv", "summary.json"]:
        assert (out / f).exists(), f
    disk = json.loads((out / "summary.json").read_text())
    assert {"mode", "roles", "samples_per_learner", "env_steps", "curriculum_history",
            "generalization"} <= set(disk)
    assert set(disk) == set(summary)
    assert disk["samples_per_learner"] >= 2048
    with open(out / "updates.csv") as f:
        assert len(list(csv.DictReader(f))) > 0
    return disk


def test_train_single(tmp_path):
    d = _check_outputs(tmp_path, ["hunter.pt"], _train(tmp_path, "--mode", "single"))
    assert not (tmp_path / "evader.pt").exists() and d["roles"] == ["hunter"]
    assert set(d["generalization"]["hunter"]) >= {"train", "val", "test", "ood"}


def test_train_independent(tmp_path):
    s = _train(tmp_path, "--mode", "independent", "--curriculum", "linear")
    d = _check_outputs(tmp_path, ["hunter.pt", "evader.pt"], s)
    assert set(d["generalization"]) == {"hunter", "evader"}


def test_train_selfplay(tmp_path):
    s = _train(tmp_path, "--mode", "selfplay")
    _check_outputs(tmp_path, ["hunter.pt", "evader.pt"], s)
    with open(tmp_path / "updates.csv") as f:
        assert max(int(r["pool"]) for r in csv.DictReader(f)) > 1


def test_train_single_frozen_ppo_opponent(tmp_path):
    ck = tmp_path / "opp.pt"
    save_checkpoint(str(ck), "ppo", ActorCritic(496, 5, (16,)))
    out = tmp_path / "run"
    s = _train(out, "--mode", "single", "--role", "evader", "--opponent", f"ppo:{ck}")
    d = _check_outputs(out, ["evader.pt"], s)
    assert d["roles"] == ["evader"]
