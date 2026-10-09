import math

from PIL import Image, ImageDraw, ImageFilter

from visualization import theme as th
from visualization.gfx import SS, canvas, downscale, outlined, radial_glow

FACINGS = {(1, 0): "down", (-1, 0): "up", (0, -1): "left", (0, 1): "right"}
AGENT_W, AGENT_H, FOOT_Y = 1.2, 1.6, 1.5
ARROW_LEVELS = 4


def facing_name(delta):
    dr, dc = delta
    if abs(dr) >= abs(dc):
        return "down" if dr >= 0 else "up"
    return "left" if dc < 0 else "right"


class Sprites:
    def __init__(self, T):
        self.T = T
        self._cache = {}

    def get(self, key, builder, *args):
        if key not in self._cache:
            self._cache[key] = builder(*args)
        return self._cache[key]

    def agent(self, role, facing):
        return self.get(("agent", role, facing), self._draw_agent, role, facing)

    def _draw_agent(self, role, facing):
        T = self.T
        im, d = canvas(AGENT_W * T, AGENT_H * T)
        u = T * SS

        k = 1.0 if role == "hunter" else 1.14

        def P(x, y):
            return (x * k + AGENT_W / 2) * u, (y * k + FOOT_Y) * u

        side = facing in ("left", "right")
        s = -1 if facing == "left" else 1
        if role == "hunter":
            self._hunter(d, P, facing, side, s)
        else:
            self._evader(d, P, facing, side, s)
        im = outlined(im, th.OUTLINE, max(2, int(0.035 * u)))
        return downscale(im, AGENT_W * T, AGENT_H * T)

    @staticmethod
    def _hunter(d, P, facing, side, s):
        cloak, dark = th.HUNTER, th.shade(th.HUNTER, 0.55)
        steel, steel_dark = (146, 150, 170), (78, 80, 98)
        boots = (44, 36, 40)
        for lx in ((-0.09, 0.12) if side else (-0.17, 0.17)):
            d.rounded_rectangle([*P(lx - 0.09, -0.17), *P(lx + 0.09, 0.0)], 8, fill=boots)
        hw, hem = (0.25, 0.26) if side else (0.42, 0.35)
        body = [P(-hw, -0.70), P(hw, -0.70), P(hem, -0.10), P(-hem, -0.10)]
        d.polygon(body, fill=cloak)
        d.polygon([P(-hem, -0.24), P(hem, -0.24), P(hem * 0.97, -0.10), P(-hem * 0.97, -0.10)],
                  fill=dark)
        d.polygon([P(-hw, -0.70), P(-hw * 0.35, -0.70), P(-hem * 0.5, -0.10), P(-hem, -0.10)],
                  fill=th.shade(cloak, 1.18))
        if facing == "down":
            d.polygon([P(-0.17, -0.66), P(0.17, -0.66), P(0.13, -0.36), P(-0.13, -0.36)],
                      fill=steel_dark)
        d.rectangle([*P(-hem, -0.34), *P(hem, -0.28)], fill=(52, 36, 36))
        unit = P(1, 0)[0] - P(0, 0)[0]
        if not side:
            for sx in (-1, 1):
                cx, cy = P(sx * 0.42, -0.66)
                d.ellipse([cx - 0.15 * unit, cy - 0.15 * unit, cx + 0.15 * unit, cy + 0.15 * unit], fill=steel)
        r = 0.2 * unit
        hx, hy = P(0.0, -0.86)
        hx += s * r * 0.15 if side else 0
        if side:
            horns = [[P(0.02 * s, -1.0), P(-0.2 * s, -1.1), P(-0.05 * s, -1.34), P(-0.02 * s, -1.12)]]
        else:
            horns = [[P(sx * 0.12, -0.98), P(sx * 0.34, -1.06), P(sx * 0.4, -1.32),
                      P(sx * 0.22, -1.1), P(sx * 0.06, -1.06)] for sx in (-1, 1)]
        for horn in horns:
            d.polygon(horn, fill=th.BONE)
        d.ellipse([hx - r, hy - r, hx + r, hy + r], fill=steel_dark)
        d.pieslice([hx - r, hy - r, hx + r, hy + r], 180, 360, fill=steel)
        if facing != "up":
            for ex in ((-0.07, 0.07) if not side else (0.09 * s,)):
                x, y = P(ex, -0.86)
                x += s * r * 0.15 if side else 0
                d.rounded_rectangle([x - r * 0.2, y - r * 0.09, x + r * 0.2, y + r * 0.14], 4, fill=(255, 168, 70))
        ax = 0.46 * s
        d.line([*P(ax, -0.28), *P(ax + 0.02 * s, -0.98)], fill=th.WOOD_DARK, width=int(0.05 * unit))
        blade_dx = 0.2 * (s if side else 1)
        d.polygon([P(ax, -1.0), P(ax + blade_dx, -1.04), P(ax + blade_dx, -0.78), P(ax, -0.82)],
                  fill=(196, 202, 218))

    @staticmethod
    def _evader(d, P, facing, side, s):
        cloak = th.EVADER
        dark, light = th.shade(cloak, 0.55), th.shade(cloak, 1.25)
        boots = (36, 38, 54)
        for lx in ((-0.05, 0.08) if side else (-0.11, 0.11)):
            d.rounded_rectangle([*P(lx - 0.07, -0.14), *P(lx + 0.07, 0.0)], 6, fill=boots)
        w = 0.2 if side else 0.3
        cape = [P(-0.2, -0.72), P(0.2, -0.72), P(w, -0.06), P(-w, -0.06)]
        d.polygon(cape, fill=cloak)
        d.polygon([P(-w * 0.9, -0.2), P(w * 0.9, -0.2), P(w, -0.06), P(-w, -0.06)], fill=dark)
        d.polygon([P(-0.2, -0.72), P(-0.05, -0.72), P(-0.1, -0.06), P(-w, -0.06)], fill=light)
        if facing == "up":
            d.line([*P(0, -0.7), *P(0, -0.1)], fill=dark, width=SS * 2)
        shift = 0.08 * s if side else 0.0
        hood = [P(-0.22 + shift, -0.7), P(-0.2 + shift, -0.98), P(0.02 + shift * 1.5, -1.2),
                P(0.2 + shift, -0.98), P(0.22 + shift, -0.7)]
        d.polygon(hood, fill=th.shade(cloak, 0.92))
        d.polygon([P(-0.2 + shift, -0.98), P(0.02 + shift * 1.5, -1.2), P(-0.04 + shift, -0.9)],
                  fill=light)
        if facing != "up":
            fx = shift * 1.9
            x0, y0 = P(fx - 0.12, -1.0)
            x1, y1 = P(fx + 0.12, -0.78)
            d.ellipse([x0, y0, x1, y1], fill=(14, 16, 30))
            for ex in ((-0.05, 0.05) if not side else (0.04 * s,)):
                x, y = P(fx + ex, -0.9)
                rr = 0.032 * (P(1, 0)[0] - P(0, 0)[0])
                d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=(210, 245, 255))
        d.polygon([P(-0.22, -0.72), P(0.22, -0.72), P(0.2, -0.62), P(-0.2, -0.62)],
                  fill=(238, 214, 168))

    def key(self, size=1.0):
        return self.get(("key", size), self._draw_key, size)

    def _draw_key(self, size):
        T = self.T
        w = h = 0.95 * T * size
        im, d = canvas(w, h)
        u = T * SS * size
        gold, hi, lo = th.KEY_GOLD, th.shade(th.KEY_GOLD, 1.25), th.shade(th.KEY_GOLD, 0.65)
        cx, cy, r = 0.26 * u, 0.5 * u, 0.17 * u
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=gold)
        d.ellipse([cx - r * 0.5, cy - r * 0.5, cx + r * 0.5, cy + r * 0.5], fill=(0, 0, 0, 0))
        d.rectangle([cx + r * 0.7, cy - 0.04 * u, 0.82 * u, cy + 0.04 * u], fill=gold)
        for tx, tl in ((0.6, 0.15), (0.74, 0.1)):
            d.rectangle([tx * u, cy, tx * u + 0.06 * u, cy + tl * u], fill=gold)
        d.arc([cx - r, cy - r, cx + r, cy + r], 200, 300, fill=hi, width=int(0.035 * u))
        d.line([cx + r, cy + 0.03 * u, 0.8 * u, cy + 0.03 * u], fill=lo, width=int(0.02 * u))
        im = outlined(im, th.OUTLINE, max(2, int(0.03 * u)))
        im = im.rotate(38, resample=Image.BICUBIC, expand=False)
        return downscale(im, w, h)

    def shadow(self, w=0.74, h=0.26, alpha=120):
        return self.get(("shadow", w, h, alpha), self._draw_shadow, w, h, alpha)

    def _draw_shadow(self, w, h, alpha):
        T = self.T
        pad = 0.12 * T
        im = Image.new("RGBA", (int(w * T + 2 * pad), int(h * T + 2 * pad)), (0, 0, 0, 0))
        ImageDraw.Draw(im).ellipse([pad, pad, pad + w * T, pad + h * T], fill=(0, 0, 0, alpha))
        return im.filter(ImageFilter.GaussianBlur(max(1, T * 0.05)))

    def glow(self, color, radius, intensity):
        key = ("glow", color, int(radius), round(intensity, 2))
        return self.get(key, radial_glow, radius, color, intensity)

    def flame(self, frame):
        return self.get(("flame", frame % 4), self._draw_flame, frame % 4)

    def _draw_flame(self, i):
        T = self.T
        w, h = 0.34 * T, 0.5 * T
        im, d = canvas(w, h)
        u = T * SS
        sway = (i - 1.5) * 0.02
        d.polygon([(x * u, y * u * 1.0) for x, y in
                   [(0.17 + sway * 2, 0.0), (0.31, 0.3), (0.28, 0.44), (0.17, 0.5),
                    (0.06, 0.44), (0.03, 0.3)]], fill=(*th.TORCH_FLAME, 255))
        d.polygon([(x * u, y * u) for x, y in
                   [(0.17 + sway, 0.17), (0.25, 0.34), (0.22, 0.45), (0.17, 0.49),
                    (0.12, 0.45), (0.09, 0.34)]], fill=(255, 238, 150, 255))
        return downscale(im, w, h)

    def arrow(self, direction, level):
        return self.get(("arrow", direction, level), self._draw_arrow, direction, level)

    def _draw_arrow(self, direction, level):
        T = self.T
        n = int(T * 0.7)
        im, d = canvas(n, n)
        u = n * SS
        s = (0.16 + 0.12 * level / (ARROW_LEVELS - 1))
        c = 0.5 * u
        d.polygon([(c + s * u, c), (c - s * u * 0.6, c - s * u * 0.8),
                   (c - s * u * 0.6, c + s * u * 0.8)], fill=(255, 250, 240, 255))
        im = outlined(im, th.OUTLINE, SS)
        im = im.rotate({"right": 0, "up": 90, "left": 180, "down": 270}[direction],
                       resample=Image.BICUBIC)
        return downscale(im, n, n)

    def stay_ring(self, level):
        return self.get(("ring", level), self._draw_ring, level)

    def _draw_ring(self, level):
        n = int(self.T * 1.1)
        im, d = canvas(n, n)
        u = n * SS
        d.ellipse([0.08 * u, 0.08 * u, 0.92 * u, 0.92 * u], outline=(255, 250, 240, 235),
                  width=int((0.02 + 0.03 * level) * u))
        return downscale(im, n, n)

    def capture_ring(self, radius_tiles):
        return self.get(("capture", radius_tiles), self._draw_capture, radius_tiles)

    def _draw_capture(self, radius_tiles):
        n = int(2 * (radius_tiles + 0.5) * self.T)
        im, d = canvas(n, n)
        u = n * SS
        d.ellipse([0.02 * u, 0.02 * u, 0.98 * u, 0.98 * u], fill=(*th.HUNTER, 34))
        for a in range(0, 360, 18):
            d.arc([0.02 * u, 0.02 * u, 0.98 * u, 0.98 * u], a, a + 9, fill=(*th.HUNTER, 170),
                  width=max(2, int(0.006 * u * 3)))
        return downscale(im, n, n)


def hop(frac):
    return 0.10 * abs(math.sin(math.pi * frac))
