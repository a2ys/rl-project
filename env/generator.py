from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

FLOOR, WALL, DOOR = 0, 1, 2

Pos = Tuple[int, int]

SEED_SPLITS = {
    "train": range(0, 1000),
    "val": range(10_000, 10_200),
    "test": range(20_000, 20_500),
}


@dataclass
class Dungeon:
    grid: np.ndarray
    evader_start: Pos
    hunter_start: Pos
    exit_pos: Pos
    keys: List[Pos]
    doors: List[Pos]
    rooms: List[Tuple[int, int, int, int]]
    seed: int
    obstacles: List[Pos] = field(default_factory=list)


def _bfs(grid: np.ndarray, start: Pos, block_doors: bool) -> np.ndarray:
    h, w = grid.shape
    dist = np.full((h, w), -1, dtype=np.int32)
    dist[start] = 0
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and dist[nr, nc] < 0:
                t = grid[nr, nc]
                if t == WALL or (block_doors and t == DOOR):
                    continue
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def _place_rooms(rng, size, n_rooms, room_size):
    lo, hi = room_size
    rooms = []
    for _ in range(300):
        if len(rooms) >= n_rooms:
            break
        h = int(rng.integers(lo, hi + 1))
        w = int(rng.integers(lo, hi + 1))
        r = int(rng.integers(1, size - h))
        c = int(rng.integers(1, size - w))
        if any(r < orr + oh + 1 and orr < r + h + 1 and
               c < oc + ow + 1 and oc < c + w + 1
               for orr, oc, oh, ow in rooms):
            continue
        rooms.append((r, c, h, w))
    return rooms


def _carve_h(grid, row, c_a, c_b):
    grid[row, min(c_a, c_b):max(c_a, c_b) + 1] = FLOOR


def _carve_v(grid, col, r_a, r_b):
    grid[min(r_a, r_b):max(r_a, r_b) + 1, col] = FLOOR


def _random_tile_in_room(rng, room, grid=None) -> Pos:
    r, c, h, w = room
    if grid is not None:
        cells = [(rr, cc) for rr in range(r, r + h) for cc in range(c, c + w)
                 if grid[rr, cc] == FLOOR]
        return cells[int(rng.integers(len(cells)))]
    return (int(rng.integers(r, r + h)), int(rng.integers(c, c + w)))


def _connected(grid) -> bool:
    cells = np.argwhere(grid != WALL)
    start = (int(cells[0][0]), int(cells[0][1]))
    return int((_bfs(grid, start, block_doors=False) >= 0).sum()) == len(cells)


def _flip(cells, ih, iw, fr, fc):
    return [((ih - 1 - i) if fr else i, (iw - 1 - j) if fc else j) for i, j in cells]


def _template(rng, ih, iw):
    opts = ["corners", "cluster"]
    if ih >= 3 and iw >= 3:
        opts += ["pillars"]
    if max(ih, iw) >= 4:
        opts += ["colonnade"]
    kind = opts[int(rng.integers(len(opts)))]
    if kind == "corners":
        cells = [(0, 0), (ih - 1, iw - 1)]
        if ih >= 3 and iw >= 3:
            cells += [(0, iw - 1), (ih - 1, 0)]
        elif rng.random() < 0.5:
            cells = _flip(cells, ih, iw, False, True)
    elif kind == "cluster":
        cells = _flip([(0, 0), (0, 1), (1, 0)], ih, iw,
                      bool(rng.random() < 0.5), bool(rng.random() < 0.5))
    elif kind == "pillars":
        cells = [(1, 1), (1, iw - 2), (ih - 2, 1), (ih - 2, iw - 2)]
    else:
        if iw >= ih:
            cells = [(ih // 2, j) for j in range(1 if iw % 2 == 0 else 0, iw, 2)]
        else:
            cells = [(i, iw // 2) for i in range(1 if ih % 2 == 0 else 0, ih, 2)]
        cells = [(i, j) for i, j in cells if 0 < i + j < ih + iw - 2 or max(ih, iw) > 4]
    return set(cells)


def _place_obstacles(rng, grid, rooms, density, lanes) -> List[Pos]:
    out: List[Pos] = []
    if density <= 0:
        return out
    for r, c, h, w in rooms:
        ih, iw = h - 2, w - 2
        if min(ih, iw) < 2 or rng.random() >= min(1.0, 4 * density):
            continue
        for i, j in sorted(_template(rng, ih, iw)):
            cell = (r + 1 + i, c + 1 + j)
            if grid[cell] != FLOOR or lanes[cell]:
                continue
            grid[cell] = WALL
            if _connected(grid):
                out.append(cell)
            else:
                grid[cell] = FLOOR
    return out


def _near_obstacles(shape, obstacles):
    m = np.zeros(shape, dtype=bool)
    for r, c in obstacles:
        m[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2] = True
    return m


def _spawn_tile(rng, room, grid, near_obs) -> Pos:
    r, c, h, w = room
    cells = [(rr, cc) for rr in range(r, r + h) for cc in range(c, c + w)
             if grid[rr, cc] == FLOOR]
    ok = [t for t in cells if not near_obs[t]]
    ok = ok or cells
    return ok[int(rng.integers(len(ok)))]


def _straight_corridor_cells(grid, room_mask) -> List[Pos]:
    h, w = grid.shape
    out = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if grid[r, c] != FLOOR or room_mask[r, c]:
                continue
            vert = (grid[r - 1, c] == FLOOR and grid[r + 1, c] == FLOOR
                    and grid[r, c - 1] == WALL and grid[r, c + 1] == WALL)
            horiz = (grid[r, c - 1] == FLOOR and grid[r, c + 1] == FLOOR
                     and grid[r - 1, c] == WALL and grid[r + 1, c] == WALL)
            if vert or horiz:
                out.append((r, c))
    return out


def _try_generate(rng, seed, size, n_rooms, n_doors, room_size,
                  hunter_min_dist, obstacle_density=0.0,
                  hunter_exit_dist=None, hunter_max_dist=30, key_min_dist=6) -> Optional[Dungeon]:
    rooms = _place_rooms(rng, size, n_rooms, room_size)
    if len(rooms) < 3:
        return None
    rooms.sort(key=lambda rm: rm[1] + rm[3] // 2)

    grid = np.ones((size, size), dtype=np.int8)
    room_mask = np.zeros((size, size), dtype=bool)
    for r, c, h, w in rooms:
        grid[r:r + h, c:c + w] = FLOOR
        room_mask[r:r + h, c:c + w] = True

    centers = [(r + h // 2, c + w // 2) for r, c, h, w in rooms]
    lane_grid = np.ones((size, size), dtype=np.int8)
    for (r1, c1), (r2, c2) in zip(centers, centers[1:]):
        if rng.random() < 0.5:
            _carve_h(lane_grid, r1, c1, c2)
            _carve_v(lane_grid, c2, r1, r2)
        else:
            _carve_v(lane_grid, c1, r1, r2)
            _carve_h(lane_grid, r2, c1, c2)
    lanes = lane_grid == FLOOR
    grid[lanes] = FLOOR

    obstacles = _place_obstacles(rng, grid, rooms, obstacle_density, lanes)
    near_obs = _near_obstacles(grid.shape, obstacles)

    si = int(rng.integers(len(rooms)))
    evader = _spawn_tile(rng, rooms[si], grid, near_obs)
    dist_open = _bfs(grid, evader, block_doors=False)
    ei = max(range(len(rooms)), key=lambda i: int(dist_open[centers[i]]))
    if ei == si:
        return None
    exit_pos = _spawn_tile(rng, rooms[ei], grid, near_obs)

    cells = _straight_corridor_cells(grid, room_mask)
    cells = [cells[i] for i in rng.permutation(len(cells))]
    doors: List[Pos] = []
    if n_doors > 0:
        first = None
        for cell in cells:
            grid[cell] = DOOR
            if _bfs(grid, evader, block_doors=True)[exit_pos] < 0:
                first = cell
                break
            grid[cell] = FLOOR
        if first is None:
            return None
        doors.append(first)
        for cell in cells:
            if len(doors) >= n_doors:
                break
            if grid[cell] != FLOOR:
                continue
            if any(abs(cell[0] - d[0]) + abs(cell[1] - d[1]) <= 2 for d in doors):
                continue
            grid[cell] = DOOR
            doors.append(cell)
        if len(doors) < n_doors:
            return None

    keys: List[Pos] = []
    if n_doors > 0:
        region = _bfs(grid, evader, block_doors=True)
        cand = [(int(p[0]), int(p[1])) for p in np.argwhere(region > 0)
                if (int(p[0]), int(p[1])) != exit_pos]
        if len(cand) < n_doors:
            return None
        good = [t for t in cand if region[t] >= key_min_dist]
        if len(good) < n_doors:
            good = sorted(cand, key=lambda t: -region[t])[:max(n_doors, 3)]
        idx = rng.choice(len(good), size=n_doors, replace=False)
        keys = [good[int(i)] for i in idx]

    occupied = {evader, exit_pos, *keys}
    tiles = [(int(p[0]), int(p[1])) for p in np.argwhere(dist_open >= 0)
             if grid[p[0], p[1]] == FLOOR]
    tiles = [t for t in tiles if t not in occupied]
    if not tiles:
        return None
    d_exit = _bfs(grid, exit_pos, block_doors=False)
    band = [t for t in tiles if hunter_min_dist <= dist_open[t] <= hunter_max_dist]
    pool = [t for t in band if hunter_exit_dist is None or 0 <= d_exit[t] <= hunter_exit_dist]
    pool = pool or [t for t in tiles if dist_open[t] >= hunter_min_dist] or         [max(tiles, key=lambda t: int(dist_open[t]))]
    hunter = pool[int(rng.integers(len(pool)))]

    return Dungeon(grid=grid, evader_start=evader, hunter_start=hunter,
                   exit_pos=exit_pos, keys=keys, doors=doors,
                   rooms=rooms, seed=seed, obstacles=obstacles)


def generate_dungeon(seed: int, size: int = 21, n_rooms: int = 5,
                     n_doors: int = 1, room_size: Tuple[int, int] = (3, 6),
                     hunter_min_dist: int = 8,
                     obstacle_density: float = 0.2,
                     hunter_exit_dist: Optional[int] = None,
                     hunter_max_dist: int = 30, key_min_dist: int = 6) -> Dungeon:
    rng = np.random.default_rng(seed)
    for attempt in range(300):
        nd = n_doors if attempt < 150 else 0
        d = _try_generate(rng, seed, size, n_rooms, nd, room_size,
                          hunter_min_dist, obstacle_density, hunter_exit_dist,
                          hunter_max_dist, key_min_dist)
        if d is not None:
            return d
    raise RuntimeError(f"Could not generate a dungeon for seed {seed}")


def validate(d: Dungeon) -> None:
    open_ = _bfs(d.grid, d.evader_start, block_doors=False)
    blocked = _bfs(d.grid, d.evader_start, block_doors=True)
    assert open_[d.exit_pos] >= 0, "exit unreachable"
    assert open_[d.hunter_start] >= 0, "hunter spawn unreachable"
    assert len(d.keys) >= len(d.doors), "not enough keys"
    assert all(blocked[k] >= 0 for k in d.keys), "key behind a door"
    if d.doors:
        assert blocked[d.exit_pos] < 0, "doors don't gate the exit"
    assert d.grid[d.evader_start] == FLOOR
    assert d.grid[d.hunter_start] == FLOOR
    assert d.grid[d.exit_pos] == FLOOR
    assert d.evader_start != d.hunter_start
    assert all(d.grid[o] == WALL for o in d.obstacles)
    assert _connected(d.grid), "map is not connected"


def render_ascii(d: Dungeon) -> str:
    chars = {FLOOR: ".", WALL: "#", DOOR: "D"}
    rows = [[chars[int(t)] for t in row] for row in d.grid]
    for r, c in d.obstacles:
        rows[r][c] = "o"
    for r, c in d.keys:
        rows[r][c] = "K"
    rows[d.exit_pos[0]][d.exit_pos[1]] = "X"
    rows[d.evader_start[0]][d.evader_start[1]] = "E"
    rows[d.hunter_start[0]][d.hunter_start[1]] = "H"
    return "\n".join("".join(r) for r in rows)


if __name__ == "__main__":
    for s in (42, 71, 103, 912):
        d = generate_dungeon(s)
        validate(d)
        print(f"--- seed {s} ---")
        print(render_ascii(d))
        print()

    a, b = generate_dungeon(42), generate_dungeon(42)
    assert np.array_equal(a.grid, b.grid) and a.keys == b.keys, "not deterministic!"
    print("determinism: OK")

    n_fail, layouts = 0, set()
    for s in SEED_SPLITS["train"]:
        try:
            d = generate_dungeon(s)
            validate(d)
            layouts.add(d.grid.tobytes())
        except Exception as e:
            n_fail += 1
            print("seed", s, "FAILED:", e)
    print(f"1000 train seeds: {n_fail} failures, {len(layouts)} distinct layouts")

    assert not (set(SEED_SPLITS["train"]) & set(SEED_SPLITS["test"]))
    assert not (set(SEED_SPLITS["val"]) & set(SEED_SPLITS["test"]))
    print("splits disjoint: OK")
