import math

import numpy as np
from PIL import Image, ImageChops, ImageDraw

from visualization import theme as th
from visualization.gfx import add_light
from visualization.props import Props
from visualization.sprites import ARROW_LEVELS, FOOT_Y, Sprites, facing_name, hop
from visualization.tiles import build_static_map

WALL_HEIGHT = 0.55
FLAME_LEVELS = 4


def tile_for_area(width, height, rows, cols):
    return max(8, int(min(width / cols, height / (rows + WALL_HEIGHT))))


class PrettyRenderer:
    def __init__(self, tile=30):
        self._facing, self._last_pos = {}, {}
        self._static_key = self._scene_key = None
        self._known_doors = set()
        self.set_tile(tile)

    def set_tile(self, tile):
        tile = max(8, int(tile))
        if getattr(self, "T", None) == tile:
            return
        self.T, self.Hh = tile, int(round(tile * WALL_HEIGHT))
        self.sprites, self.props = Sprites(self.T), Props(self.T, self.Hh)
        self._static = self._scene = None
        self._static_key = self._scene_key = None

    def map_size(self, rows, cols):
        return cols * self.T, rows * self.T + self.Hh

    def _ensure_static(self, snap):
        grid = snap["grid"]
        obstacles = tuple(sorted(map(tuple, snap.get("obstacles", []))))
        key = (self.T, snap.get("seed"), (grid == 1).tobytes(), obstacles, tuple(snap["exit"]))
        if key != self._static_key:
            self._static = build_static_map(grid, obstacles, tuple(snap["exit"]), snap.get("seed"),
                                            self.T, self.Hh)
            self._static_key = key
            self._scene_key = None
            self._known_doors = set()
            self._facing, self._last_pos = {}, {}
        self._known_doors |= set(map(tuple, snap["doors"]))

    def _build_scene(self, shape, vision, heat, trails):
        T, Hh = self.T, self.Hh
        H, W = shape
        scene = self._static.floor.copy()
        if heat is not None:
            small = np.zeros((H, W, 3), np.uint8)
            alpha = np.zeros((H, W), np.uint8)
            for r, c in zip(*np.nonzero(np.isfinite(heat))):
                small[r, c] = th.heat_color(heat[r, c])
                alpha[r, c] = 150
            size = (W * T, H * T)
            scene.paste(Image.fromarray(small).resize(size, Image.NEAREST), (0, Hh),
                        Image.fromarray(alpha).resize(size, Image.NEAREST))
        for role, cells in (vision or {}).items():
            m = np.zeros((H, W), np.uint8)
            for r, c in cells:
                m[r, c] = 38
            mask = Image.fromarray(m).resize((W * T, H * T), Image.BILINEAR)
            mask = ImageChops.multiply(mask, self._static.walkable)
            scene.paste(Image.new("RGB", mask.size, th.ROLE_COLOR[role]), (0, Hh), mask)
        for role, pts in (trails or {}).items():
            n = len(pts)
            for i, (r, c) in enumerate(pts):
                f = (i + 1) / n
                glow = self.sprites.glow(th.ROLE_COLOR[role], T * (0.08 + 0.14 * f), 0.25 + 0.55 * f)
                add_light(scene, glow, (c + 0.5) * T, Hh + (r + 0.5) * T)
        scene.paste(self._static.occ, (0, 0), self._static.occ)
        return scene

    def _reocclude(self, frame, box, depth, door_list):
        T, Hh = self.T, self.Hh
        x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
        x1, y1 = min(frame.width, int(box[2])), min(frame.height, int(box[3]))
        if x1 <= x0 or y1 <= y0:
            return
        strips = self._static.strips
        for r in range(max(0, depth + 1), len(strips)):
            top = r * T
            if top >= y1:
                break
            ya, yb = max(y0, top), min(y1, top + T + Hh)
            if yb <= ya:
                continue
            if strips[r] is not None:
                part = strips[r].crop((x0, ya - top, x1, yb - top))
                frame.paste(part, (x0, ya), part)
            for dr, dx, spr in door_list:
                if dr == r and dx + T > x0 and dx < x1:
                    part = spr.crop((max(x0, dx) - dx, ya - top, min(x1, dx + T) - dx, yb - top))
                    frame.paste(part, (max(x0, dx), ya), part)

    def render_map(self, snap, hp=None, ep=None, anim=0.0, trails=None, vision=None, heat=None,
                   probs=None, capture_radius=0):
        T, Hh = self.T, self.Hh
        self._ensure_static(snap)
        grid = snap["grid"]
        H, W = grid.shape
        pos = {"hunter": hp if hp is not None else snap["hunter"],
               "evader": ep if ep is not None else snap["evader"]}
        scene_key = (self._static_key, tuple((k, frozenset(v)) for k, v in (vision or {}).items()),
                     None if heat is None else hash(heat.tobytes()),
                     tuple((k, tuple(map(tuple, v))) for k, v in (trails or {}).items()))
        if scene_key != self._scene_key:
            self._scene = self._build_scene((H, W), vision, heat, trails)
            self._scene_key = scene_key
        frame = self._scene.copy()

        def center(p):
            return (p[1] + 0.5) * T, Hh + (p[0] + 0.5) * T

        st = self._static
        for i, (fx, fy, lx, ly) in enumerate(st.torches):
            lvl = int(2 + 2 * math.sin(anim * 9 + i * 2.3) * math.sin(anim * 5.1 + i)) % FLAME_LEVELS
            add_light(frame, self.sprites.glow(th.TORCH_FLAME, 2.3 * T, 0.12 + 0.045 * lvl), lx, ly)
            flame = self.sprites.flame(int(anim * 8 + i))
            frame.paste(flame, (int(fx - flame.width / 2), int(fy - flame.height / 2)), flame)
        ex, ey = center(snap["exit"])
        pulse = int((0.5 + 0.5 * math.sin(anim * 2.6)) * (FLAME_LEVELS - 1))
        add_light(frame, self.sprites.glow(th.EXIT_GREEN, 1.7 * T, 0.28 + 0.09 * pulse), ex, ey)
        add_light(frame, self.sprites.glow(th.EXIT_GREEN, 0.7 * T, 0.5 + 0.1 * pulse), ex, ey - 0.45 * T)
        self._motes(frame, ex, ey, anim)

        doors = set(map(tuple, snap["doors"]))
        door_list = []
        for r, c in sorted(self._known_doors):
            spr = self.props.door(closed=(r, c) in doors)
            door_list.append((r, c * T, spr))
            frame.paste(spr, (c * T, r * T), spr)

        for role, p in pos.items():
            cx, cy = center(p)
            sh = self.sprites.shadow()
            frame.paste(sh, (int(cx - sh.width / 2), int(cy + 0.3 * T - sh.height / 2)), sh)
            add_light(frame, self.sprites.glow(th.ROLE_COLOR[role], 1.0 * T, 0.2), cx, cy)
        if capture_radius:
            ring = self.sprites.capture_ring(capture_radius)
            cx, cy = center(pos["hunter"])
            frame.paste(ring, (int(cx - ring.width / 2), int(cy - ring.height / 2)), ring)

        items = []
        for k in snap["keys"]:
            items.append((k[0], "key", k))
        for role, p in pos.items():
            items.append((math.floor(p[0] + 0.5) + 0.01, role, p))
        items.sort(key=lambda it: it[0])
        for depth, kind, data in items:
            if kind == "key":
                box = self._draw_key(frame, center(data), anim, data)
            else:
                box = self._draw_agent(frame, kind, data, anim, snap["keys_held"] > 0,
                                       (probs or {}).get(kind))
            self._reocclude(frame, box, int(depth), door_list)

        return frame

    def _draw_key(self, frame, c, anim, rc):
        T = self.T
        bob = 0.07 * T * math.sin(anim * 3 + rc[0] + rc[1])
        add_light(frame, self.sprites.glow(th.KEY_GOLD, 0.9 * T, 0.28), c[0], c[1] - 0.1 * T)
        spr = self.sprites.key()
        x, y = int(c[0] - spr.width / 2), int(c[1] - spr.height / 2 - 0.12 * T + bob)
        frame.paste(spr, (x, y), spr)
        return (x - 4, y - 4, x + spr.width + 4, y + spr.height + 0.3 * T)

    def _draw_agent(self, frame, role, p, anim, has_key, probs):
        T, Hh = self.T, self.Hh
        last = self._last_pos.get(role)
        if last is not None and (abs(p[0] - last[0]) + abs(p[1] - last[1])) > 1e-6:
            self._facing[role] = facing_name((p[0] - last[0], p[1] - last[1]))
        self._last_pos[role] = p
        facing = self._facing.get(role, "down")
        spr = self.sprites.agent(role, facing)
        fr, fc = p[0] % 1.0, p[1] % 1.0
        frac = max(fr if fr > 1e-6 else 0, fc if fc > 1e-6 else 0)
        lift = hop(frac) if frac else 0.025 * math.sin(anim * 3.2 + (role == "evader") * 2)
        cx, cy = (p[1] + 0.5) * T, Hh + (p[0] + 0.5) * T
        x = int(cx - spr.width / 2)
        y = int(cy + 0.3 * T - FOOT_Y * T - lift * T)
        frame.paste(spr, (x, y), spr)
        box = [x, y, x + spr.width, y + spr.height]
        if role == "evader" and has_key:
            k = self.sprites.key(0.62)
            kx, ky = int(cx - k.width / 2), int(y - 0.22 * T + 0.05 * T * math.sin(anim * 4))
            add_light(frame, self.sprites.glow(th.KEY_GOLD, 0.6 * T, 0.3), cx, ky + k.height / 2)
            frame.paste(k, (kx, ky), k)
            box[1] = min(box[1], ky)
        if probs is not None:
            self._arrows(frame, cx, cy - 0.2 * T, probs)
            box = [min(box[0], cx - 0.9 * T), min(box[1], cy - 1.1 * T),
                   max(box[2], cx + 0.9 * T), max(box[3], cy + 0.8 * T)]
        return tuple(box)

    def _arrows(self, frame, cx, cy, probs):
        T = self.T
        for direction, (dx, dy), p in zip(("up", "down", "left", "right"),
                                          ((0, -1), (0, 1), (-1, 0), (1, 0)), probs[:4]):
            if p < 0.05:
                continue
            a = self.sprites.arrow(direction, min(ARROW_LEVELS - 1, int(float(p) * ARROW_LEVELS)))
            frame.paste(a, (int(cx + dx * 0.68 * T - a.width / 2), int(cy + dy * 0.68 * T - a.height / 2)), a)
        if len(probs) > 4 and probs[4] >= 0.05:
            ring = self.sprites.stay_ring(min(ARROW_LEVELS - 1, int(float(probs[4]) * ARROW_LEVELS)))
            frame.paste(ring, (int(cx - ring.width / 2), int(cy - ring.height / 2)), ring)

    def _motes(self, frame, ex, ey, anim):
        T = self.T
        d = ImageDraw.Draw(frame)
        for i in range(5):
            t = (anim * 0.45 + i / 5) % 1.0
            x = ex + math.sin(i * 12.9 + t * 3) * 0.28 * T
            y = ey - t * 0.9 * T
            r = max(1, int(T * 0.035 * (1 - t) + 1))
            c = th.mix(th.EXIT_GREEN, (255, 255, 255), 0.5 * (1 - t))
            d.ellipse([x - r, y - r, x + r, y + r], fill=c)

    def render(self, snap, hp=None, ep=None, anim=0.0, trails=None, vision=None, heat=None,
               probs=None, stats=None, hspec="", espec="", footer=(), panel=True,
               capture_radius=0, info=None):
        img = self.render_map(snap, hp, ep, anim, trails, vision, heat, probs, capture_radius)
        if not panel:
            return img
        from visualization import hud
        state = hud.HudState.from_snapshot(snap, hspec, espec, probs, stats, info)
        gap, pad = 28, 20
        side = hud.render_side_panel(380, 2000, state, 1.0)
        side = side.crop((0, 0, side.width, (side.getbbox() or (0, 0, 0, img.height))[3] + 4))
        h = max(img.height, side.height)
        out = Image.new("RGB", (pad + img.width + gap + side.width + pad, h + 2 * pad), th.UI_BG)
        out.paste(img, (pad, pad + (h - img.height) // 2))
        out.paste(side, (pad + img.width + gap, pad), side)
        return out
