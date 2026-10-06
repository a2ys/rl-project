"""Static dungeon layer: stone floor, extruded walls, props, baked lighting.

Output of `build_static_map` (cached per map and tile size by the renderer):
  floor   RGB image of the whole map (flagstones, ambient occlusion, lighting)
  strips  per-row RGBA images of everything that can hide a moving sprite
          (wall blocks, pillars, crates); row r's strip starts at image y = r * T
  occ     all strips composited, drawn over the floor for the base scene
  torches (flame_x, flame_y, light_x, light_y) in image pixels

Geometry: tile (r, c) has footprint [c*T, (c+1)*T) x [Hh + r*T, Hh + (r+1)*T). A wall is
a block Hh pixels tall: its top face is the footprint shifted up by Hh, its front face
fills the gap below the top face. Drawing rows top to bottom therefore sorts by depth.
"""
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from visualization import theme as th
from visualization.props import Props, obstacle_kind

TORCH_SPACING = 5.5       # minimum distance between torches, in tiles
TORCH_RADIUS = 4.4        # light radius in tiles
LIGHT_RES = 4             # light map is computed at 1/LIGHT_RES resolution


@dataclass
class StaticMap:
    floor: Image.Image
    strips: list
    occ: Image.Image
    torches: list
    walkable: Image.Image      # L mask (map footprint area) of non-rock tiles, for overlays


def _jitter(c, v):
    return tuple(max(0, min(255, int(x + v))) for x in c)


# ------------------------------------------------------------------ floor
def _draw_flagstone(d, x, y, T, rng):
    v = rng.uniform(-1, 1) * 7
    tint = rng.uniform(-1, 1) * 3
    col = (int(th.FLOOR_BASE[0] + v + tint), int(th.FLOOR_BASE[1] + v), int(th.FLOOR_BASE[2] + v - tint))
    d.rectangle([x, y, x + T - 1, y + T - 1], fill=col)
    if rng.random() < 0.35:                                   # split slab
        if rng.random() < 0.5:
            s = x + int(T * rng.uniform(0.3, 0.7))
            d.line([s, y, s, y + T - 1], fill=th.FLOOR_MORTAR)
        else:
            s = y + int(T * rng.uniform(0.3, 0.7))
            d.line([x, s, x + T - 1, s], fill=th.FLOOR_MORTAR)
    d.line([x, y, x + T - 1, y], fill=th.FLOOR_HILITE)        # bevel: lit top/left, dark bottom/right
    d.line([x, y, x, y + T - 1], fill=_jitter(th.FLOOR_HILITE, -10))
    d.line([x, y + T - 1, x + T - 1, y + T - 1], fill=th.FLOOR_MORTAR)
    d.line([x + T - 1, y, x + T - 1, y + T - 1], fill=th.FLOOR_MORTAR)
    if T >= 16 and rng.random() < 0.3:                        # crack
        px, py = x + rng.uniform(0.1, 0.9) * T, y + rng.uniform(0.1, 0.9) * T
        for _ in range(int(rng.integers(2, 5))):
            nx = min(max(px + rng.uniform(-0.3, 0.3) * T, x + 2), x + T - 3)
            ny = min(max(py + rng.uniform(-0.3, 0.3) * T, y + 2), y + T - 3)
            d.line([px, py, nx, ny], fill=th.CRACK, width=max(1, T // 30))
            px, py = nx, ny
    for _ in range(int(rng.integers(0, 4))):                  # pits and flecks
        fx, fy = x + int(rng.integers(2, T - 2)), y + int(rng.integers(2, T - 2))
        d.point([fx, fy], fill=_jitter(col, rng.choice([-18, 14])))


def _ambient_occlusion(size, rock, obstacles, T, Hh):
    """Soft darkening near wall bases and under obstacles (blurred footprint mask)."""
    mask = Image.new("L", size, 0)
    d = ImageDraw.Draw(mask)
    off = int(0.1 * T)
    for r, c in zip(*np.nonzero(rock)):
        x, y = c * T, Hh + r * T
        d.rectangle([x, y + off, x + T, y + T + off], fill=255)
    for r, c in obstacles:
        x, y = c * T, Hh + r * T
        d.ellipse([x + 0.1 * T, y + 0.55 * T, x + 0.95 * T, y + 1.05 * T], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(0.2 * T))
    return np.asarray(mask, np.float32)[..., None] / 255.0


def _build_floor(grid, rock, obstacles, exit_pos, props, rng, T, Hh):
    H, W = grid.shape
    size = (W * T, H * T + Hh)
    img = Image.new("RGB", size, th.VOID)
    d = ImageDraw.Draw(img)
    for r in range(H):
        for c in range(W):
            if not rock[r, c]:
                _draw_flagstone(d, c * T, Hh + r * T, T, rng)
    er, ec = exit_pos
    img.paste(props.exit_base(), (ec * T, Hh + er * T), props.exit_base())
    arr = np.asarray(img, np.float32)
    arr += rng.normal(0, 2.2, (size[1], size[0], 1))                      # grain
    stains = Image.fromarray(rng.integers(0, 255, (H * 2, W * 2), dtype=np.uint8))
    stains = np.asarray(stains.resize(size, Image.BICUBIC), np.float32)[..., None] / 255.0
    arr *= 1.0 + (stains - 0.5) * 0.22                                    # low-frequency dirt
    arr *= 1.0 - 0.62 * _ambient_occlusion(size, rock, obstacles, T, Hh)
    return arr


# ------------------------------------------------------------------ walls
def _draw_wall(d, x, T, Hh, up_rock, left_rock, right_rock, front_visible, rng):
    """One wall block into a row strip (top face y in [0, T), front face below)."""
    j = rng.uniform(-1, 1) * 9
    top = _jitter(th.WALL_TOP, j)
    d.rectangle([x, 0, x + T - 1, T - 1], fill=top)
    d.line([x, T - 1, x + T - 1, T - 1], fill=th.shade(top, 0.78))        # slab seams
    d.line([x + T - 1, 0, x + T - 1, T - 1], fill=th.shade(top, 0.82))
    if rng.random() < 0.5:
        sx = x + int(T * rng.uniform(0.25, 0.75))
        d.line([sx, 1, sx, T - 2], fill=th.shade(top, 0.86))
    if rng.random() < 0.35:
        sy = int(T * rng.uniform(0.3, 0.7))
        d.line([x + 1, sy, x + T - 2, sy], fill=th.shade(top, 0.9))
    if not up_rock:                                           # lit far edge of the top face
        d.line([x, 0, x + T - 1, 0], fill=th.WALL_TOP_RIM, width=2)
    if not left_rock:
        d.line([x, 0, x, T - 1], fill=th.shade(top, 1.12))
    if not right_rock:
        d.line([x + T - 1, 0, x + T - 1, T - 1], fill=th.shade(top, 0.7))
    if not front_visible:
        return
    y0, y1 = T, T + Hh
    d.rectangle([x, y0, x + T - 1, y1 - 1], fill=th.WALL_FRONT)
    courses = 2 if Hh >= 14 else 1
    ch = Hh / courses
    for i in range(courses):                                  # staggered brick courses
        ya, yb = y0 + int(i * ch), y0 + int((i + 1) * ch)
        bw = max(6, T // 2)
        shift = (bw // 2) * ((i + int(x // T)) % 2)
        bx = x - shift
        while bx < x + T:
            xa, xb = max(x, bx), min(x + T, bx + bw)
            if xb > xa:
                d.rectangle([xa, ya, xb - 1, yb - 1], fill=_jitter(th.WALL_FRONT, rng.uniform(-6, 6)))
                d.line([xa, ya, xb - 1, ya], fill=th.MORTAR)
                if bx >= x:
                    d.line([xa, ya, xa, yb - 1], fill=th.MORTAR)
            bx += bw
    d.line([x, y0, x + T - 1, y0], fill=th.shade(th.WALL_TOP, 0.62), width=1)
    d.line([x, y0 + 1, x + T - 1, y0 + 1], fill=th.shade(th.WALL_FRONT, 1.3))       # edge highlight
    for k in range(max(2, Hh // 3)):                          # base shading toward the floor
        tone = th.mix(th.WALL_FRONT_DARK, th.WALL_FRONT, k / max(2, Hh // 3))
        d.line([x, y1 - 1 - k, x + T - 1, y1 - 1 - k], fill=tone)
    if not left_rock:
        d.line([x, y0, x, y1 - 1], fill=th.shade(th.WALL_FRONT, 1.25))
    if not right_rock:
        d.line([x + T - 1, y0, x + T - 1, y1 - 1], fill=th.shade(th.WALL_FRONT, 0.6))


def _pick_torches(candidates, rng):
    """Greedy, spaced torch placement over wall tiles whose front face is visible."""
    chosen = []
    for i in rng.permutation(len(candidates)):
        r, c = candidates[i]
        if all((r - a) ** 2 + (c - b) ** 2 >= TORCH_SPACING ** 2 for a, b in chosen):
            chosen.append((r, c))
    return chosen


# ---------------------------------------------------------------- lighting
def light_multiplier(size, lights, strength=th.VIGNETTE_STRENGTH):
    """(h, w, 3) float32 brightness multiplier: cool ambient + warm radial lights + vignette."""
    w, h = size
    lw, lh = w // LIGHT_RES + 2, h // LIGHT_RES + 2
    xs = (np.arange(lw) * LIGHT_RES)[None, :].astype(np.float32)
    ys = (np.arange(lh) * LIGHT_RES)[:, None].astype(np.float32)
    m = np.empty((lh, lw, 3), np.float32)
    m[:] = th.AMBIENT
    for lx, ly, rad, col, k in lights:
        dist = np.sqrt((xs - lx) ** 2 + (ys - ly) ** 2) / rad
        m += (np.clip(1 - dist, 0, 1) ** 2 * k)[..., None] * np.array(col, np.float32)
    r2 = (((xs / w - 0.5) * 2) ** 2 + ((ys / h - 0.5) * 2) ** 2) / 2
    m *= (1 - strength * np.clip(r2, 0, 1) ** 1.3)[..., None]
    small = Image.fromarray(np.clip(m * 100, 0, 255).astype(np.uint8))
    big = small.resize((lw * LIGHT_RES, lh * LIGHT_RES), Image.BICUBIC).crop((0, 0, w, h))
    return np.asarray(big, np.float32) / 100.0


# ------------------------------------------------------------------- build
def build_static_map(grid, obstacles, exit_pos, seed, T, Hh):
    H, W = grid.shape
    rng = np.random.default_rng(int(seed or 0) * 7919 + 17)
    obst = {tuple(o) for o in obstacles}
    rock = grid == 1
    for r, c in obst:
        rock[r, c] = False                                    # obstacles stand on floor
    props = Props(T, Hh)
    size = (W * T, H * T + Hh)

    floor = _build_floor(grid, rock, obst, exit_pos, props, rng, T, Hh)

    padded = np.pad(rock, 1, constant_values=True)
    strips = [None] * H
    candidates = []
    visible = np.zeros_like(rock)
    for r in range(H):
        for c in range(W):
            if rock[r, c]:
                near = padded[r:r + 3, c:c + 3]
                visible[r, c] = not near.all()                # buried rock stays void
    for r in range(H):
        has_wall = visible[r].any() or any((r, c) in obst for c in range(W))
        if not has_wall:
            continue
        strip = Image.new("RGBA", (W * T, T + Hh), (0, 0, 0, 0))
        d = ImageDraw.Draw(strip)
        for c in range(W):
            if visible[r, c]:
                up = padded[r, c + 1]
                front = not padded[r + 2, c + 1]
                _draw_wall(d, c * T, T, Hh, up, padded[r + 1, c], padded[r + 1, c + 2], front, rng)
                if front and not grid[min(r + 1, H - 1), c] == 2:
                    candidates.append((r, c))
            elif (r, c) in obst:
                spr = props.obstacle(obstacle_kind(r, c))
                strip.alpha_composite(spr, (c * T + (T - spr.width) // 2,
                                            Hh + T - spr.height - int(0.04 * T)))
        strips[r] = strip

    torches, lights = [], []
    for r, c in _pick_torches(candidates, rng):
        br = props.torch_bracket()
        strips[r].alpha_composite(br, (c * T + (T - br.width) // 2, T + int(Hh * 0.38)))
        fx, fy = c * T + T / 2, r * T + T + Hh * 0.24
        lx, ly = fx, r * T + T + Hh + 0.2 * T
        torches.append((fx, fy, lx, ly))
        lights.append((lx, ly, TORCH_RADIUS * T, th.TORCH_LIGHT, 0.95))
    er, ec = exit_pos
    lights.append(((ec + 0.5) * T, Hh + (er + 0.5) * T, 2.6 * T, th.EXIT_LIGHT, 0.9))

    lm = light_multiplier(size, lights)
    floor_img = Image.fromarray(np.clip(floor * lm, 0, 255).astype(np.uint8), "RGB")
    occ = Image.new("RGBA", size, (0, 0, 0, 0))
    for r, strip in enumerate(strips):
        if strip is None:
            continue
        a = np.asarray(strip, np.float32)
        a[..., :3] += rng.normal(0, 2.4, (T + Hh, W * T, 1))                  # stone grain
        a[..., :3] *= lm[r * T:r * T + T + Hh]
        strips[r] = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGBA")
        occ.alpha_composite(strips[r], (0, r * T))
    walkable = Image.fromarray(np.kron(~rock, np.ones((T, T))).astype(np.uint8) * 255, "L")
    return StaticMap(floor_img, strips, occ, torches, walkable)
