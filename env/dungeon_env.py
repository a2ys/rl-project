"""Dungeon Pursuit: two-player (Hunter vs Evader) simultaneous-move environment.

Multi-agent dict API:
    obs, info = env.reset(seed=42)                  # obs[role] -> float32 (496,)
    obs, rew, term, trunc, info = env.step({HUNTER: a, EVADER: b})
Use env.wrappers.SingleAgentEnv for a standard single-agent Gymnasium env.

reset(seed=s): reseeds the env RNG and plays dungeon s.
reset(options={"map_seed": s}): plays dungeon s but keeps the RNG stream running.
reset() : picks a dungeon from `seeds` using the RNG.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from gymnasium import spaces

from env.generator import DOOR, FLOOR, WALL, generate_dungeon

HUNTER, EVADER = "hunter", "evader"
UP, DOWN, LEFT, RIGHT, STAY = range(5)
MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))

N_CHANNELS = 6   # wall, door, key, exit, other agent, visible
N_SCALARS = 10


@dataclass
class EnvConfig:
    size: int = 21
    n_rooms: int = 5
    n_doors: int = 1
    max_steps: int = 200
    view_radius: int = 4              # 9x9 window
    hunter_opens_doors: bool = True   # Hunter walks through doors without a key
    win_reward: float = 1.0
    step_penalty: float = 0.01
    obstacle_density: float = 0.2     # pillars/crates inside rooms (block movement + sight)
    shaping: float = 0.01             # per tile of path-distance progress
    key_bonus: float = 0.1
    door_bonus: float = 0.1
    hunter_sprint_every: int = 40     # Hunter repeats its move on every 40th step (bonus move; 0 = off)
    capture_radius: int = 0           # caught if Manhattan distance <= r after the move (0 = same cell; also with swap rule)
    hunter_min_dist: int = 8          # Hunter spawns at least this many path tiles from the Evader
    hunter_exit_dist: int | None = 6  # Hunter spawns within this path distance of the exit (None = anywhere far)


@lru_cache(maxsize=4096)
def _dungeon(seed, size, n_rooms, n_doors, obstacle_density, hunter_exit_dist,
             hunter_min_dist):
    return generate_dungeon(seed, size=size, n_rooms=n_rooms, n_doors=n_doors,
                            obstacle_density=obstacle_density,
                            hunter_exit_dist=hunter_exit_dist,
                            hunter_min_dist=hunter_min_dist)


class DungeonPursuitEnv:
    def __init__(self, cfg: EnvConfig | None = None, seeds=None, rng_seed=None):
        self.cfg = cfg or EnvConfig()
        self.seeds = list(seeds) if seeds is not None else list(range(1000))
        self.rng = np.random.default_rng(rng_seed)
        self.H = self.W = self.cfg.size
        R = self.cfg.view_radius
        self.win = 2 * R + 1
        n_obs = N_CHANNELS * self.win * self.win + N_SCALARS
        self.observation_space = spaces.Box(-1.0, 1.0, (n_obs,), np.float32)
        self.action_space = spaces.Discrete(5)
        self.dungeon = None
        self.t = 0

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        map_seed = None
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            map_seed = int(seed)
        if options and "map_seed" in options:
            map_seed = int(options["map_seed"])
        if map_seed is None:
            map_seed = int(self.rng.choice(self.seeds))
        c = self.cfg
        self.dungeon = _dungeon(map_seed, c.size, c.n_rooms, c.n_doors, c.obstacle_density,
                         c.hunter_exit_dist, c.hunter_min_dist)
        d = self.dungeon
        self.grid = d.grid.copy()
        self.pos = {HUNTER: tuple(d.hunter_start), EVADER: tuple(d.evader_start)}
        self.exit_pos = tuple(d.exit_pos)
        self.keys = [tuple(k) for k in d.keys]
        self.doors = [tuple(x) for x in d.doors]
        self.keys_held = 0
        self.t = 0
        self.outcome = ""
        self._cache = {}
        return self._obs_all(), {"map_seed": map_seed}

    # ----------------------------------------------------------- path helpers
    def dist_field(self, targets, can_open_doors):
        """Multi-source BFS path distance to the nearest target; -1 = unreachable."""
        key = (frozenset(targets), bool(can_open_doors))
        f = self._cache.get(key)
        if f is not None:
            return f
        H, W = self.H, self.W
        flat = self.grid.ravel().tolist()
        dist = [-1] * (H * W)
        q = deque()
        for r, c in targets:
            dist[r * W + c] = 0
            q.append(r * W + c)
        while q:
            i = q.popleft()
            r, c = divmod(i, W)
            nd = dist[i] + 1
            for dr, dc in MOVES[:4]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    j = nr * W + nc
                    if dist[j] >= 0:
                        continue
                    t = flat[j]
                    if t == WALL or (t == DOOR and not can_open_doors):
                        continue
                    dist[j] = nd
                    q.append(j)
        f = np.array(dist, dtype=np.int32).reshape(H, W)
        if len(self._cache) > 256:
            self._cache.clear()
        self._cache[key] = f
        return f

    def first_action_toward(self, start, targets, can_open_doors) -> int:
        f = self.dist_field(set(targets), can_open_doors)
        r, c = start
        d = f[r, c]
        if d <= 0:
            return STAY
        for a in range(4):
            nr, nc = r + MOVES[a][0], c + MOVES[a][1]
            if 0 <= nr < self.H and 0 <= nc < self.W and f[nr, nc] == d - 1:
                return a
        return STAY

    def objective_targets(self):
        """Evader goal: the key if a closed door blocks the exit, else the exit."""
        if self.keys_held == 0 and self.keys and self.doors:
            f = self.dist_field({self.exit_pos}, False)
            r, c = self.pos[EVADER]
            if f[r, c] < 0:
                return set(self.keys)
        return {self.exit_pos}

    def random_floor_tile(self):
        tiles = np.argwhere(self.grid == FLOOR)
        r, c = tiles[int(self.rng.integers(len(tiles)))]
        return (int(r), int(c))

    # ----------------------------------------------------------------- vision
    def _los(self, a, b) -> bool:
        """Bresenham line of sight; walls and closed doors block (endpoints excepted)."""
        (r0, c0), (r1, c1) = a, b
        dr, dc = abs(r1 - r0), abs(c1 - c0)
        sr = 1 if r1 > r0 else -1
        sc = 1 if c1 > c0 else -1
        err = dc - dr
        r, c = r0, c0
        while (r, c) != (r1, c1):
            if (r, c) != (r0, c0) and self.grid[r, c] != FLOOR:
                return False
            e2 = 2 * err
            if e2 > -dr:
                err -= dr
                c += sc
            if e2 < dc:
                err += dc
                r += sr
        return True

    def _in_view(self, role, cell) -> bool:
        p = self.pos[role]
        R = self.cfg.view_radius
        if max(abs(cell[0] - p[0]), abs(cell[1] - p[1])) > R:
            return False
        if not (0 <= cell[0] < self.H and 0 <= cell[1] < self.W):
            return False
        return self._los(p, cell)

    def sees_other(self, role) -> bool:
        other = EVADER if role == HUNTER else HUNTER
        return self._in_view(role, self.pos[other])

    def observe(self, role) -> np.ndarray:
        R, n = self.cfg.view_radius, self.win
        pr, pc = self.pos[role]
        other = EVADER if role == HUNTER else HUNTER
        opos = self.pos[other]
        keys, doors = set(self.keys), set(self.doors)
        ch = np.zeros((N_CHANNELS, n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                r, c = pr + i - R, pc + j - R
                if not (0 <= r < self.H and 0 <= c < self.W):
                    ch[0, i, j] = 1.0          # outside the map counts as wall
                    continue
                if not self._los((pr, pc), (r, c)):
                    continue
                ch[5, i, j] = 1.0
                t = self.grid[r, c]
                ch[0, i, j] = t == WALL
                ch[1, i, j] = (r, c) in doors
                ch[2, i, j] = (r, c) in keys
                ch[3, i, j] = (r, c) == self.exit_pos
                ch[4, i, j] = (r, c) == opos

        H, W = self.H, self.W
        s = np.zeros(N_SCALARS, dtype=np.float32)
        s[0], s[1] = pr / (H - 1), pc / (W - 1)
        s[2] = min(self.keys_held, 1)
        s[3] = self.t / self.cfg.max_steps
        visible = self.sees_other(role)
        if role == EVADER:                      # goal knowledge: key or exit
            tg = self.objective_targets()
            f = self.dist_field(tg, self.keys_held > 0)
            d = f[pr, pc]
            a = self.first_action_toward((pr, pc), tg, self.keys_held > 0)
            s[4], s[5] = MOVES[a]               # path-direction compass
            s[6] = 1.0 if d < 0 else min(d, 100) / 100
        elif visible:                           # Hunter only targets what it sees
            f = self.dist_field({opos}, self.cfg.hunter_opens_doors)
            d = f[pr, pc]
            a = self.first_action_toward((pr, pc), {opos}, self.cfg.hunter_opens_doors)
            s[4], s[5] = MOVES[a]
            s[6] = 1.0 if d < 0 else min(d, 100) / 100
        if visible:
            s[7] = 1.0
            s[8], s[9] = (opos[0] - pr) / H, (opos[1] - pc) / W
        return np.clip(np.concatenate([ch.ravel(), s]), -1.0, 1.0).astype(np.float32)

    def _obs_all(self):
        return {HUNTER: self.observe(HUNTER), EVADER: self.observe(EVADER)}

    # ------------------------------------------------------------------- step
    def resolve_move(self, role, action):
        """Returns (new_pos, opened_door_pos_or_None) without changing state."""
        r, c = self.pos[role]
        dr, dc = MOVES[int(action)]
        nr, nc = r + dr, c + dc
        if (dr, dc) == (0, 0) or not (0 <= nr < self.H and 0 <= nc < self.W):
            return (r, c), None
        t = self.grid[nr, nc]
        if t == WALL:
            return (r, c), None
        if t == DOOR:
            can = self.keys_held > 0 if role == EVADER else self.cfg.hunter_opens_doors
            return ((nr, nc), (nr, nc)) if can else ((r, c), None)
        return (nr, nc), None

    def step(self, actions):
        cfg = self.cfg
        h_old, e_old = self.pos[HUNTER], self.pos[EVADER]

        # shaping baselines (cached BFS fields)
        tg_old = self.objective_targets()
        d_e_old = self.dist_field(tg_old, self.keys_held > 0)[e_old]
        d_h_old = self.dist_field({e_old}, cfg.hunter_opens_doors)[h_old]

        h_new, h_open = self.resolve_move(HUNTER, actions[HUNTER])
        e_new, e_open = self.resolve_move(EVADER, actions[EVADER])
        h_path = [h_new]
        k = cfg.hunter_sprint_every
        if k > 0 and (self.t + 1) % k == 0:   # sprint: repeat the Hunter's action once more
            self.pos[HUNTER] = h_new
            h_new2, h_open2 = self.resolve_move(HUNTER, actions[HUNTER])
            self.pos[HUNTER] = h_old
            h_path.append(h_new2)
            h_new = h_new2
            h_open = h_open2 or h_open
        rew = {HUNTER: -cfg.step_penalty, EVADER: -cfg.step_penalty}

        opened = False
        for door, who in ((h_open, HUNTER), (e_open, EVADER)):
            if door is None:
                continue
            if self.grid[door] == DOOR:
                self.grid[door] = FLOOR
                if door in self.doors:
                    self.doors.remove(door)
                opened = True
            if who == EVADER:
                self.keys_held -= 1
                rew[EVADER] += cfg.door_bonus
        if opened:
            self._cache.clear()

        self.pos[HUNTER], self.pos[EVADER] = h_new, e_new
        if e_new in self.keys:
            self.keys.remove(e_new)
            self.keys_held += 1
            rew[EVADER] += cfg.key_bonus

        # path-distance shaping
        d_e_new = self.dist_field(tg_old, True if self.keys_held > 0 else False)[e_new]
        if d_e_old >= 0 and d_e_new >= 0:
            rew[EVADER] += cfg.shaping * (d_e_old - d_e_new)
        d_h_new = self.dist_field({e_new}, cfg.hunter_opens_doors)[h_new]
        if d_h_old >= 0 and d_h_new >= 0:
            rew[HUNTER] += cfg.shaping * (d_h_old - d_h_new)

        self.t += 1
        caught = any(abs(h[0] - e_new[0]) + abs(h[1] - e_new[1]) <= cfg.capture_radius
                     for h in h_path) or (h_new == e_old and e_new == h_old)
        escaped = e_new == self.exit_pos
        term = trunc = False
        if caught:
            term, self.outcome = True, "caught"
            rew[HUNTER] += cfg.win_reward
            rew[EVADER] -= cfg.win_reward
        elif escaped:
            term, self.outcome = True, "escaped"
            rew[EVADER] += cfg.win_reward
            rew[HUNTER] -= cfg.win_reward
        elif self.t >= cfg.max_steps:
            trunc, self.outcome = True, "timeout"
        info = {"outcome": self.outcome or "ongoing", "t": self.t}
        return self._obs_all(), rew, term, trunc, info

    # ------------------------------------------------------------ rendering
    def snapshot(self):
        return dict(grid=self.grid, hunter=self.pos[HUNTER], evader=self.pos[EVADER],
                    exit=self.exit_pos, keys=list(self.keys), doors=list(self.doors),
                    t=self.t, keys_held=self.keys_held, outcome=self.outcome,
                    obstacles=list(self.dungeon.obstacles), seed=self.dungeon.seed,
                    max_steps=self.cfg.max_steps)
