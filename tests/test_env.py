import numpy as np
import pytest

from agents.scripted import ChaseHunter, IdleAgent, PathEvader, RandomAgent
from env.dungeon_env import EVADER, HUNTER, STAY, DungeonPursuitEnv, EnvConfig
from env.generator import generate_dungeon, validate
from env.wrappers import SingleAgentEnv


def test_generator_solvable():
    for s in range(200):
        validate(generate_dungeon(s))


def test_reset_deterministic():
    env = DungeonPursuitEnv()
    a, _ = env.reset(seed=42)
    b, _ = env.reset(seed=42)
    c, _ = env.reset(seed=43)
    assert np.array_equal(a[HUNTER], b[HUNTER])
    assert np.array_equal(a[EVADER], b[EVADER])
    assert not np.array_equal(a[HUNTER], c[HUNTER])


def test_obs_in_space():
    env = DungeonPursuitEnv()
    obs, _ = env.reset(seed=1)
    for role in (HUNTER, EVADER):
        assert obs[role].shape == env.observation_space.shape
        assert env.observation_space.contains(obs[role])


def test_random_rollouts_finish():
    env = DungeonPursuitEnv()
    for s in range(10):
        env.reset(seed=s)
        for _ in range(500):
            acts = {HUNTER: int(env.rng.integers(5)), EVADER: int(env.rng.integers(5))}
            obs, rew, term, trunc, info = env.step(acts)
            assert env.observation_space.contains(obs[HUNTER])
            assert np.isfinite(rew[HUNTER]) and np.isfinite(rew[EVADER])
            if term or trunc:
                break
        assert term or trunc
        assert info["outcome"] in ("caught", "escaped", "timeout")


def test_catch_same_cell():
    env = DungeonPursuitEnv()
    env.reset(seed=1)
    env.pos[HUNTER] = env.pos[EVADER]
    _, rew, term, trunc, info = env.step({HUNTER: STAY, EVADER: STAY})
    assert term and info["outcome"] == "caught"
    assert rew[HUNTER] > 0 > rew[EVADER]


def test_timeout():
    env = DungeonPursuitEnv(EnvConfig(max_steps=15))
    env.reset(seed=3)
    for _ in range(15):
        _, _, term, trunc, info = env.step({HUNTER: STAY, EVADER: STAY})
    assert trunc and not term and info["outcome"] == "timeout"


def test_path_evader_solves_keys_and_doors():
    env = DungeonPursuitEnv(EnvConfig(max_steps=300))
    ev = PathEvader(flee_radius=0)
    for s in range(30):
        env.reset(seed=s)
        env.pos[HUNTER] = (0, 0)
        done = False
        while not done:
            _, _, term, trunc, info = env.step({HUNTER: STAY, EVADER: ev.act(env, EVADER)})
            done = term or trunc
        assert info["outcome"] == "escaped", f"seed {s}: {info}"


def test_oracle_hunter_catches_idle_evader():
    env = DungeonPursuitEnv(EnvConfig(max_steps=300))
    hu = ChaseHunter(oracle=True)
    for s in range(20):
        env.reset(seed=s)
        hu.reset()
        done = False
        while not done:
            _, _, term, trunc, info = env.step({HUNTER: hu.act(env, HUNTER), EVADER: STAY})
            done = term or trunc
        assert info["outcome"] == "caught", f"seed {s}: {info}"


def test_gymnasium_api_check():
    from stable_baselines3.common.env_checker import check_env
    check_env(SingleAgentEnv(HUNTER, RandomAgent()), warn=True)
    check_env(SingleAgentEnv(EVADER, ChaseHunter()), warn=True)


def test_capture_radius():
    env = DungeonPursuitEnv(EnvConfig(capture_radius=1))
    env.reset(seed=1)
    h = env.pos[HUNTER]
    adj = next(((h[0] + dr, h[1] + dc) for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0))
                if env.grid[h[0] + dr, h[1] + dc] == 0), None)
    env.pos[EVADER] = adj
    _, _, term, _, info = env.step({HUNTER: STAY, EVADER: STAY})
    assert term and info["outcome"] == "caught"
    env0 = DungeonPursuitEnv(EnvConfig(capture_radius=0))
    env0.reset(seed=1)
    env0.pos[EVADER] = adj
    _, _, term, _, _ = env0.step({HUNTER: STAY, EVADER: STAY})
    assert not term


def test_hunter_sprint_moves_twice():
    env = DungeonPursuitEnv(EnvConfig(hunter_sprint_every=2))
    env.reset(seed=1)
    env.pos[EVADER] = (0, 0)
    start = env.pos[HUNTER]
    for a in range(4):
        r, c = start
        dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[a]
        if env.grid[r + dr, c + dc] == 0 and env.grid[r + 2 * dr, c + 2 * dc] == 0:
            break
    else:
        pytest.skip("no straight corridor at spawn")
    env.step({HUNTER: STAY, EVADER: STAY})
    env.step({HUNTER: a, EVADER: STAY})
    assert env.pos[HUNTER] == (start[0] + 2 * dr, start[1] + 2 * dc)


def test_hunter_spawns_near_exit():
    from env.generator import _bfs
    near = sum(_bfs((d := generate_dungeon(s, hunter_exit_dist=6)).grid, d.exit_pos, False)
               [d.hunter_start] <= 6 for s in range(100))
    assert near >= 80
