import numpy as np
from PIL import Image

from visualization import theme as th
from visualization.gfx import SS, canvas, downscale, outlined

OBSTACLE_KINDS = ("pillar", "crate", "pillar_broken", "crate_stack")


def obstacle_kind(r, c):
    return OBSTACLE_KINDS[((r * 7 + c * 13) // 2 + r * c) % len(OBSTACLE_KINDS)]


class Props:
    def __init__(self, T, Hh):
        self.T, self.Hh = T, Hh
        self._cache = {}

    def get(self, key, builder, *args):
        if key not in self._cache:
            self._cache[key] = builder(*args)
        return self._cache[key]

    def obstacle(self, kind):
        return self.get(("obs", kind), self._draw_obstacle, kind)

    def _draw_obstacle(self, kind):
        if kind.startswith("pillar"):
            return self._pillar(broken=kind.endswith("broken"))
        return self._crate(stacked=kind.endswith("stack"))

    def _pillar(self, broken):
        T = self.T
        w, h = 0.86 * T, (0.85 if broken else 1.3) * T
        im, d = canvas(w, h)
        u = T * SS
        W, Hpx = int(w * SS), int(h * SS)
        stone = th.mix(th.WALL_TOP, (120, 118, 134), 0.5)
        base_h = 0.16 * u
        d.rounded_rectangle([0.02 * u, Hpx - base_h, W - 0.02 * u, Hpx - 2], 6, fill=th.shade(stone, 0.8))
        top = 0.2 * u if not broken else 0.0
        sx0, sx1 = 0.15 * u, W - 0.15 * u
        shaft = np.zeros((Hpx, W, 4), np.uint8)
        xs = np.linspace(-1, 1, int(sx1 - sx0))
        light = np.clip(0.62 + 0.55 * np.cos((xs + 0.55) * 1.2), 0.35, 1.25)
        col = (np.array(stone, np.float32)[None, :] * light[:, None]).clip(0, 255).astype(np.uint8)
        y0 = int(top + (0.1 * u if not broken else 0.25 * u))
        y1 = int(Hpx - base_h)
        shaft[y0:y1, int(sx0):int(sx0) + len(xs), :3] = col[None, :, :]
        shaft[y0:y1, int(sx0):int(sx0) + len(xs), 3] = 255
        im.alpha_composite(Image.fromarray(shaft, "RGBA"))
        for fx in (0.38, 0.62):
            x = sx0 + (sx1 - sx0) * fx
            d.line([x, y0 + 4, x, y1 - 2], fill=(*th.shade(stone, 0.55), 160), width=SS)
        if broken:
            d.polygon([(sx0, y0), (sx0 + 0.18 * u, y0 - 0.1 * u), (sx0 + 0.3 * u, y0 + 0.04 * u),
                       (sx0 + 0.46 * u, y0 - 0.14 * u), (sx1, y0 + 0.02 * u), (sx1, y0 + 0.2 * u),
                       (sx0, y0 + 0.2 * u)], fill=th.shade(stone, 1.15))
            for rx, ry, rr in ((0.06, 0.9, 0.08), (0.8, 0.94, 0.06)):
                d.ellipse([(rx - rr) * u, Hpx - (1 - ry + rr) * u * 0.4 - rr * u,
                           (rx + rr) * u, Hpx - (1 - ry - rr) * u * 0.4], fill=th.shade(stone, 0.9))
        else:
            d.rounded_rectangle([0.04 * u, 0.0, W - 0.04 * u, 0.2 * u], 6, fill=th.shade(stone, 1.2))
            d.rectangle([0.04 * u + 4, 0.14 * u, W - 0.04 * u - 4, 0.2 * u], fill=th.shade(stone, 0.85))
        im = outlined(im, th.OUTLINE, max(2, int(0.03 * u)))
        return downscale(im, w, h)

    def _crate(self, stacked):
        T = self.T
        w, h = 0.84 * T, (1.15 if stacked else 0.78) * T
        im, d = canvas(w, h)
        u = T * SS
        wood, wood_hi, wood_lo = th.WOOD, th.shade(th.WOOD, 1.28), th.shade(th.WOOD, 0.62)

        def box(x0, y0, x1, y1, depth):
            d.rectangle([x0, y0 - depth, x1, y0], fill=wood_hi)
            d.rectangle([x0, y0, x1, y1], fill=wood)
            for i in range(1, 4):
                x = x0 + (x1 - x0) * i / 4
                d.line([x, y0, x, y1], fill=wood_lo, width=SS)
            for yy in (y0, y1 - 0.08 * u):
                d.rectangle([x0, yy, x1, yy + 0.08 * u], fill=th.IRON)
            d.line([x0, y0, x1, y1], fill=wood_lo, width=int(0.045 * u))
            d.line([x0, y1, x1, y0], fill=wood_lo, width=int(0.045 * u))
            d.line([x0, y0 - depth * 0.5, x1, y0 - depth * 0.5], fill=wood, width=SS)

        W, Hpx = int(w * SS), int(h * SS)
        bw = W - 0.1 * u
        if stacked:
            box(0.05 * u, Hpx - 0.04 * u - 0.5 * u, 0.05 * u + bw, Hpx - 0.04 * u, 0.2 * u)
            box(0.22 * u, Hpx - 0.04 * u - 0.5 * u - 0.2 * u - 0.28 * u + 0.12 * u,
                0.22 * u + bw * 0.62, Hpx - 0.04 * u - 0.5 * u - 0.2 * u + 0.12 * u, 0.12 * u)
        else:
            box(0.05 * u, Hpx - 0.04 * u - 0.5 * u, 0.05 * u + bw, Hpx - 0.04 * u, 0.22 * u)
        im = outlined(im, th.OUTLINE, max(2, int(0.03 * u)))
        return downscale(im, w, h)

    def door(self, closed=True):
        return self.get(("door", closed), self._draw_door, closed)

    def _draw_door(self, closed):
        T, Hh = self.T, self.Hh
        im, d = canvas(T, T + Hh)
        u, Hf = T * SS, Hh * SS
        top_h = T * SS
        stone = th.WALL_TOP
        if closed:
            d.rectangle([0, 0, u, top_h], fill=th.shade(th.WOOD, 1.05))
            for i in range(1, 5):
                d.line([0, top_h * i / 5, u, top_h * i / 5], fill=th.shade(th.WOOD, 0.7), width=SS)
            d.rectangle([0, 0, 0.12 * u, top_h], fill=stone)
            d.rectangle([u - 0.12 * u, 0, u, top_h], fill=th.shade(stone, 0.8))
            d.rectangle([0, 0, u, 0.06 * u], fill=th.shade(th.WOOD, 1.35))
            fy = top_h
            d.rectangle([0, fy, u, fy + Hf], fill=th.shade(th.WOOD, 0.78))
            for i in range(1, 6):
                x = u * i / 6
                d.line([x, fy, x, fy + Hf], fill=th.WOOD_DARK, width=SS)
            for frac in (0.22, 0.78):
                d.rectangle([0, fy + Hf * frac - 0.04 * u, u, fy + Hf * frac + 0.04 * u], fill=th.IRON)
            d.rectangle([0, fy, 0.1 * u, fy + Hf], fill=th.shade(stone, 0.7))
            d.rectangle([u - 0.1 * u, fy, u, fy + Hf], fill=th.shade(stone, 0.55))
            cx, cy = u / 2, fy + Hf * 0.5
            r = 0.13 * u
            d.arc([cx - r * 0.7, cy - r * 1.5, cx + r * 0.7, cy - r * 0.2], 180, 360,
                  fill=th.IRON, width=int(0.045 * u))
            d.rounded_rectangle([cx - r, cy - r * 0.5, cx + r, cy + r * 0.9], 6, fill=th.KEY_GOLD,
                                outline=th.shade(th.KEY_GOLD, 0.55), width=SS)
            d.ellipse([cx - r * 0.2, cy, cx + r * 0.2, cy + r * 0.4], fill=th.shade(th.KEY_GOLD, 0.4))
            d.rectangle([0, fy + Hf - 0.05 * u, u, fy + Hf], fill=(0, 0, 0, 80))
        else:
            post = th.shade(stone, 0.9)
            for x0 in (0.02 * u, u - 0.2 * u):
                d.rounded_rectangle([x0, top_h - 0.1 * u, x0 + 0.18 * u, top_h + Hf - 0.02 * u], 8, fill=post)
                d.rectangle([x0, top_h - 0.1 * u, x0 + 0.18 * u, top_h - 0.02 * u], fill=th.shade(stone, 1.25))
            d.line([0.22 * u, top_h + Hf - 0.08 * u, u - 0.22 * u, top_h + Hf - 0.08 * u],
                   fill=(*th.KEY_GOLD, 120), width=int(0.03 * u))
        im = outlined(im, th.OUTLINE, max(2, int(0.02 * u)))
        return downscale(im, T, T + Hh)

    def exit_base(self):
        return self.get(("exit",), self._draw_exit)

    def _draw_exit(self):
        T = self.T
        im, d = canvas(T, T)
        u = T * SS
        g = th.EXIT_GREEN
        d.rounded_rectangle([0.04 * u, 0.04 * u, 0.96 * u, 0.96 * u], 10, fill=th.shade(th.WALL_TOP, 0.75),
                            outline=th.OUTLINE, width=SS)
        steps = 5
        for i in range(steps):
            m = 0.12 + 0.07 * i
            tone = th.mix((70, 74, 96), g, i / (steps - 1) * 0.85)
            d.rectangle([m * u, m * u, (1 - m) * u, (1 - m) * u], fill=th.shade(tone, 0.55 + 0.1 * i))
            d.line([m * u, m * u, (1 - m) * u, m * u], fill=th.shade(tone, 1.2), width=int(0.025 * u))
        d.rectangle([0.43 * u, 0.43 * u, 0.57 * u, 0.57 * u], fill=(220, 255, 235))
        for a in ((0.5, 0.06), (0.5, 0.94)):
            d.rectangle([a[0] * u - 0.025 * u, a[1] * u - 0.025 * u, a[0] * u + 0.025 * u,
                         a[1] * u + 0.025 * u], fill=g)
        return downscale(im, T, T)

    def torch_bracket(self):
        return self.get(("torch",), self._draw_torch)

    def _draw_torch(self):
        T = self.T
        w, h = 0.3 * T, 0.34 * T
        im, d = canvas(w, h)
        u = T * SS
        d.polygon([(0.08 * u, 0.0), (0.22 * u, 0.0), (0.19 * u, 0.2 * u), (0.11 * u, 0.2 * u)],
                  fill=(120, 80, 44))
        d.rectangle([0.12 * u, 0.2 * u, 0.18 * u, 0.3 * u], fill=th.IRON)
        d.rounded_rectangle([0.06 * u, 0.28 * u, 0.24 * u, 0.33 * u], 4, fill=th.IRON)
        im = outlined(im, th.OUTLINE, SS)
        return downscale(im, w, h)
