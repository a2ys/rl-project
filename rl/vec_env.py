"""Vectorized two-role pursuit envs.

Actions come in as an (N, 2) int array ordered (hunter, evader). A value of -1 means the
role is played inside the worker by its scripted agent; any other value is the action
chosen by a policy in the main process. Envs auto-reset when an episode ends.
"""
import multiprocessing as mp
from functools import partial

import numpy as np

from agents.factory import make_agent
from agents.scripted import NoisyAgent
from env.dungeon_env import EVADER, HUNTER, DungeonPursuitEnv
from rl.curriculum import LEVELS, level_cfg

ROLES = (HUNTER, EVADER)


class PursuitTask:
    def __init__(self, seeds, rng_seed, scripted=None, levels=LEVELS, level=None,
                 base_cfg=None):
        self.levels, self.base_cfg = levels, base_cfg
        self.level = len(levels) - 1 if level is None else level
        self.env = DungeonPursuitEnv(level_cfg(levels[self.level], base_cfg),
                                     seeds=seeds, rng_seed=rng_seed)
        self.specs = dict(scripted or {})
        self.use_eps = True
        self._build_agents()

    def _build_agents(self):
        eps = self.levels[self.level].get("opp_eps", 0.0) if self.use_eps else 0.0
        self.agents = {r: NoisyAgent(make_agent(s), eps) for r, s in self.specs.items()}

    def set_level(self, level, use_eps=True):
        """Applies from the next episode on."""
        self.level, self.use_eps = level, use_eps
        self.env.cfg = level_cfg(self.levels[level], self.base_cfg)
        self._build_agents()

    def reset(self):
        obs, info = self.env.reset()
        self.map_seed = info["map_seed"]
        for a in self.agents.values():
            a.reset()
        self.ret = [0.0, 0.0]
        self.ep_level = self.level
        return np.stack([obs[HUNTER], obs[EVADER]])

    def step(self, actions):
        acts = {}
        for i, r in enumerate(ROLES):
            acts[r] = int(actions[i]) if actions[i] >= 0 else self.agents[r].act(self.env, r)
        obs, rew, term, trunc, info = self.env.step(acts)
        r = np.array([rew[HUNTER], rew[EVADER]], np.float32)
        self.ret[0] += r[0]
        self.ret[1] += r[1]
        done = term or trunc
        ep = None
        if done:
            ep = dict(outcome=info["outcome"], length=self.env.t, ret_hunter=self.ret[0],
                      ret_evader=self.ret[1], map_seed=self.map_seed, level=self.ep_level)
            o = self.reset()
        else:
            o = np.stack([obs[HUNTER], obs[EVADER]])
        return o, r, done, ep


def make_task(seeds, rng_seed, scripted=None, level=None, base_cfg=None):
    return PursuitTask(seeds, rng_seed, scripted=scripted, level=level, base_cfg=base_cfg)


class SyncVecEnv:
    def __init__(self, task_fns):
        self.tasks = [f() for f in task_fns]
        self.n = len(self.tasks)

    def reset(self):
        return np.stack([t.reset() for t in self.tasks])

    def step(self, actions):
        out = [t.step(a) for t, a in zip(self.tasks, actions)]
        obs, rew, done, eps = zip(*out)
        return np.stack(obs), np.stack(rew), np.array(done), list(eps)

    def call(self, method, *args, indices=None):
        idx = range(self.n) if indices is None else indices
        return [getattr(self.tasks[i], method)(*args) for i in idx]

    def close(self):
        pass


def _worker(conn, task_fns):
    tasks = [f() for f in task_fns]
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "step":
                out = [t.step(a) for t, a in zip(tasks, data)]
                obs, rew, done, eps = zip(*out)
                conn.send((np.stack(obs), np.stack(rew), np.array(done), list(eps)))
            elif cmd == "reset":
                conn.send(np.stack([t.reset() for t in tasks]))
            elif cmd == "call":
                method, args, local_idx = data
                conn.send([getattr(tasks[i], method)(*args) for i in local_idx])
            elif cmd == "close":
                break
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        conn.close()


class SubprocVecEnv:
    """Splits the tasks over `n_workers` processes; each worker steps its chunk serially."""

    def __init__(self, task_fns, n_workers):
        self.n = len(task_fns)
        n_workers = max(1, min(n_workers, self.n))
        chunks = np.array_split(np.arange(self.n), n_workers)
        self.chunks = [c.tolist() for c in chunks]
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for chunk in self.chunks:
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, [task_fns[i] for i in chunk]),
                            daemon=True)
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)

    def reset(self):
        for c in self.conns:
            c.send(("reset", None))
        return np.concatenate([c.recv() for c in self.conns])

    def step(self, actions):
        for c, chunk in zip(self.conns, self.chunks):
            c.send(("step", actions[chunk[0]:chunk[-1] + 1]))
        res = [c.recv() for c in self.conns]
        obs = np.concatenate([r[0] for r in res])
        rew = np.concatenate([r[1] for r in res])
        done = np.concatenate([r[2] for r in res])
        eps = [e for r in res for e in r[3]]
        return obs, rew, done, eps

    def call(self, method, *args, indices=None):
        want = set(range(self.n) if indices is None else indices)
        sent = []
        for c, chunk in zip(self.conns, self.chunks):
            local = [j for j, i in enumerate(chunk) if i in want]
            if local:
                c.send(("call", (method, args, local)))
                sent.append(c)
        return [x for c in sent for x in c.recv()]

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for p in self.procs:
            p.join(timeout=5)


def make_vec_env(n_envs, seeds, base_seed, scripted_per_env, n_workers=0, level=None,
                 base_cfg=None):
    fns = [partial(make_task, list(seeds), base_seed * 1000 + i, scripted_per_env[i],
                   level, base_cfg) for i in range(n_envs)]
    return SubprocVecEnv(fns, n_workers) if n_workers > 0 else SyncVecEnv(fns)
