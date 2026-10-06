"""Live 2.5D viewer window (tkinter + Pillow). Plays episodes in real time.

python -m visualization.live --hunter chase --evader path --seed 42
python -m visualization.live --hunter chase --evader ppo:experiments/evader_vs_chase_s0/model

For PPO agents the panel shows the policy's action probabilities live (U/D/L/R/S,
also drawn as arrows around the agent) and H shows a value heatmap V(s).

Keys: SPACE pause | N/P next/prev map | R restart | S step (paused) | +/- speed
      V vision cones | T trails | H value heatmap (off/evader/hunter) | L legend
      C screenshot | Q quit

The window is resizable: tile size is recomputed from the canvas area (debounced) and the
static map layer is re-cached; per frame only dynamic sprites are composited.
"""
import argparse
import time
import tkinter as tk
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk

from agents.factory import make_agent
from env.dungeon_env import EVADER, HUNTER, DungeonPursuitEnv, EnvConfig
from env.generator import WALL
from visualization import hud
from visualization import theme as th
from visualization.gfx import rounded_panel
from visualization.pretty import PrettyRenderer, tile_for_area

ROLES = (HUNTER, EVADER)
FRAME_BUDGET_MS = 16


def action_probs(agent, env, role):
    """Policy action distribution for a PPO agent, else None."""
    model = getattr(agent, "model", None)
    if model is None:
        return None
    import torch
    obs_t, _ = model.policy.obs_to_tensor(env.observe(role))
    with torch.no_grad():
        return model.policy.get_distribution(obs_t).distribution.probs[0].cpu().numpy()


def value_heatmap(agent, env, role):
    """V(s) of the PPO policy with `role` standing on every walkable tile."""
    import torch
    model = agent.model
    heat = np.full((env.H, env.W), np.nan)
    saved = env.pos[role]
    for r in range(env.H):
        for c in range(env.W):
            if env.grid[r, c] == WALL:
                continue
            env.pos[role] = (r, c)
            obs_t, _ = model.policy.obs_to_tensor(env.observe(role))
            with torch.no_grad():
                heat[r, c] = float(model.policy.predict_values(obs_t)[0, 0])
    env.pos[role] = saved
    lo, hi = np.nanmin(heat), np.nanmax(heat)
    return (heat - lo) / (hi - lo + 1e-9)


def hex_color(c):
    return "#%02x%02x%02x" % tuple(c[:3])


def make_background(w, h):
    """Dark backdrop with a soft vignette and faint grain."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xs / w - 0.5) * 2) ** 2 + ((ys / h - 0.5) * 2) ** 2) / 1.414
    base = np.array(th.UI_BG, np.float32) * (1.25 - 0.9 * r ** 1.5)[..., None]
    base += np.random.default_rng(0).normal(0, 1.2, (h, w, 1))
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


class Viewer:
    RESIZE_DEBOUNCE_MS = 120

    def __init__(self, root, hunter_spec, evader_spec, seed, sps, tile):
        self.root = root
        self.env = DungeonPursuitEnv(EnvConfig())
        self.agents = {HUNTER: make_agent(hunter_spec), EVADER: make_agent(evader_spec)}
        self.specs = {HUNTER: hunter_spec, EVADER: evader_spec}
        self.r = PrettyRenderer(tile)
        self.seed, self.sps = seed, float(sps)
        self.paused, self.show_vision, self.show_trails, self.show_legend = False, True, True, True
        self.heat_mode = 0                       # 0 off, then each PPO role in turn
        self.heat, self.heat_t = None, -99
        self.stats = dict(n=0, caught=0, escaped=0, timeout=0)
        self.canvas = tk.Canvas(root, bd=0, highlightthickness=0, bg=hex_color(th.UI_BG))
        self.canvas.pack(fill="both", expand=True)
        self.backdrop_item = self.canvas.create_image(0, 0, anchor="nw")
        self.map_item = self.canvas.create_image(0, 0, anchor="nw")
        self.photos = {}                         # Tk images: static backdrop and the per-frame map
        self.layout, self._resize_job, self.frame_ms = None, None, 0.0
        self._hud_cache, self._legend_pos = {}, None
        self.canvas.bind("<Configure>", self.on_configure)
        keys = {"space": self.toggle, "n": lambda: self.load(self.seed + 1),
                "p": lambda: self.load(self.seed - 1), "r": lambda: self.load(self.seed),
                "s": self.step_once, "plus": lambda: self.speed(1.4),
                "equal": lambda: self.speed(1.4), "minus": lambda: self.speed(1 / 1.4),
                "v": self.flip_vision, "t": self.flip_trails, "h": self.cycle_heat,
                "l": self.flip_legend, "c": self.screenshot, "q": root.destroy,
                "Escape": root.destroy}
        for k, fn in keys.items():
            root.bind(f"<{k}>" if len(k) > 1 else k, lambda e, f=fn: f())
        self.t0 = time.perf_counter()
        self.load(seed)
        size = self.initial_size(tile)
        root.geometry("%dx%d" % size)
        self.relayout(*size)
        self.tick()

    # ----------------------------------------------------------- layout
    def initial_size(self, tile):
        """Window size that fits the map at roughly `tile` pixels per cell."""
        return int((self.env.W + 0.5) * tile + 420), int((self.env.H + 0.5) * tile + 130)

    def on_configure(self, event):
        """Debounced resize: relayout only after the user stops dragging."""
        if self._resize_job:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(self.RESIZE_DEBOUNCE_MS, self.relayout, event.width, event.height)

    def relayout(self, cw, ch):
        """Recompute tile size and screen regions for a canvas of cw x ch pixels."""
        self._resize_job = None
        cw, ch = max(cw, 640), max(ch, 480)
        ui = max(0.75, min(1.7, ch / 900))
        margin, gap = int(14 * ui), int(18 * ui)
        panel_w = int(min(360 * ui, cw * 0.36))
        footer_h = int((34 + 26 + 6) * ui) + 2 * margin            # key strip + status chips
        area = (margin, margin, cw - margin, ch - footer_h)         # region left for map + info panel
        T = tile_for_area(area[2] - area[0] - panel_w - gap, area[3] - area[1], self.env.H, self.env.W)
        self.r.set_tile(T)
        mw, mh = self.r.map_size(self.env.H, self.env.W)
        group_x = (cw - (mw + gap + panel_w)) // 2                  # map and panel are centred as one group
        map_xy = (max(margin, group_x), area[1] + (area[3] - area[1] - mh) // 2)
        pad, self._layout_id = int(4 * ui), getattr(self, "_layout_id", 0) + 1
        self.layout = dict(
            size=(cw, ch), ui=ui, margin=margin, gap=gap, panel_w=panel_w, area=area, map_xy=map_xy,
            map_size=(mw, mh), panel_xy=(map_xy[0] + mw + gap, map_xy[1]),
            panel_h=area[3] - map_xy[1],
            id=self._layout_id, bg=make_background(cw, ch),
            frame_xy=(map_xy[0] - pad, map_xy[1] - pad),
            map_frame=rounded_panel(mw + 2 * pad, mh + 2 * pad, pad + 1, (6, 5, 10, 255), th.PANEL_EDGE,
                                    max(1, int(ui))))
        self._hud_cache, self._legend_pos = {}, None
        self.photos = {}
        self.canvas.coords(self.map_item, *map_xy)

    # ----------------------------------------------------------- state
    def load(self, seed):
        self.seed = seed
        self.env.reset(seed=seed)
        for a in self.agents.values():
            a.reset()
        self.prev = dict(self.env.pos)
        self.trails = {k: deque([self.env.pos[k]], maxlen=14) for k in ROLES}
        self.done, self.done_at = False, 0.0
        self.last_step = time.perf_counter()
        self.heat, self.heat_t = None, -99
        self.refresh_derived()

    def refresh_derived(self):
        """Everything that only changes when the environment steps."""
        env = self.env
        self.probs = {k: action_probs(self.agents[k], env, k) for k in ROLES}
        self.vision = None
        if self.show_vision:
            R = env.cfg.view_radius
            self.vision = {}
            for k in ROLES:
                pr, pc = env.pos[k]
                self.vision[k] = {(r, c) for r in range(pr - R, pr + R + 1)
                                  for c in range(pc - R, pc + R + 1)
                                  if env._in_view(k, (r, c))}
        mode = self.heat_mode
        if mode and (env.t - self.heat_t >= 12 or self.heat is None):
            self.heat = value_heatmap(self.agents[ROLES[mode - 1]], env, ROLES[mode - 1])
            self.heat_t = env.t
        self.info = hud.env_info(env)

    def step_once(self):
        if self.done:
            return
        self.prev = dict(self.env.pos)
        acts = {k: self.agents[k].act(self.env, k) for k in ROLES}
        _, _, term, trunc, info = self.env.step(acts)
        for k in ROLES:
            self.trails[k].append(self.env.pos[k])
        self.last_step = time.perf_counter()
        if term or trunc:
            self.done, self.done_at = True, self.last_step
            self.stats["n"] += 1
            self.stats[info["outcome"]] += 1
        self.refresh_derived()

    # -------------------------------------------------------- controls
    def toggle(self):
        self.paused = not self.paused

    def speed(self, f):
        self.sps = max(1.0, min(40.0, self.sps * f))

    def flip_vision(self):
        self.show_vision = not self.show_vision
        self.refresh_derived()

    def flip_trails(self):
        self.show_trails = not self.show_trails

    def flip_legend(self):
        self.show_legend = not self.show_legend

    def cycle_heat(self):
        ppo = [i + 1 for i, k in enumerate(ROLES) if hasattr(self.agents[k], "model")]
        order = [0] + ppo
        self.heat_mode = order[(order.index(self.heat_mode) + 1) % len(order)]
        self.heat = None
        if self.heat_mode:
            self.refresh_derived()

    def screenshot(self, path=None):
        Path("experiments").mkdir(exist_ok=True)
        out = path or f"experiments/screenshot_seed{self.seed}_t{self.env.t}.png"
        self.frame().save(out)
        print("saved", out)

    # ----------------------------------------------------------- frame
    def cached(self, name, key, build):
        """HUD pieces change per step, not per frame: rebuild only when their key changes."""
        hit = self._hud_cache.get(name)
        if hit is None or hit[0] != key:
            hit = (key, build())
            self._hud_cache[name] = hit
        return hit[1]

    def hud_state(self):
        tag = {0: "", 1: "heatmap: hunter V(s)", 2: "heatmap: evader V(s)"}[self.heat_mode]
        s = hud.HudState.from_snapshot(self.env.snapshot(), self.specs[HUNTER], self.specs[EVADER],
                                       self.probs, self.stats, self.info)
        s.sps, s.paused, s.heat_tag = self.sps, self.paused, tag
        return s

    def interpolated_positions(self, now):
        """Positions between the previous and current tile (smoothstep eased)."""
        a = 1.0 if (self.paused or self.done) else min(1.0, (now - self.last_step) * self.sps)
        a = a * a * (3 - 2 * a)
        return {k: (self.prev[k][0] + (self.env.pos[k][0] - self.prev[k][0]) * a,
                    self.prev[k][1] + (self.env.pos[k][1] - self.prev[k][1]) * a) for k in ROLES}

    def legend(self):
        """(image, position, inside_map) of the legend, or None when hidden."""
        if not self.show_legend:
            return None
        L, capture = self.layout, getattr(self.env.cfg, "capture_radius", 0)
        img = self.cached("legend", (L["id"], self.show_vision, self.show_trails, self.heat_mode, capture),
                          lambda: hud.legend_panel(L["ui"], self.show_vision, self.show_trails,
                                                   bool(self.heat_mode), capture))
        pos = self.legend_position(img.size)
        mx, my = L["map_xy"]
        mw, mh = L["map_size"]
        inside = pos[0] + img.width > mx and pos[0] < mx + mw and pos[1] + img.height > my and pos[1] < my + mh
        return img, pos, inside

    def backdrop(self, state):
        """Everything except the map: rebuilt (and re-uploaded to Tk) only when HUD content changes."""
        L, ui = self.layout, self.layout["ui"]
        cw, ch = L["size"]
        m, strip_w = L["margin"], cw - 2 * L["margin"]
        flags = dict(vision=self.show_vision, trails=self.show_trails, legend=self.show_legend)
        key = (L["id"], state.key(), tuple(flags.values()), self.heat_mode)

        def build():
            out = L["bg"].copy()
            out.paste(L["map_frame"], L["frame_xy"], L["map_frame"])
            side = hud.render_side_panel(L["panel_w"], L["panel_h"], state, ui)
            out.paste(side, L["panel_xy"], side)
            keys = hud.key_strip(strip_w, flags, ui)
            out.paste(keys, (m, ch - keys.height - m), keys)
            playing = hud.status_strip(strip_w, state, ui)
            out.paste(playing, (m, ch - keys.height - m - playing.height - int(6 * ui)), playing)
            legend = self.legend()
            if legend and not legend[2]:
                out.paste(legend[0], legend[1], legend[0])
            return out
        return self.cached("backdrop", key, build)

    def parts(self):
        """(backdrop, map image): the map image carries the legend when it overlaps the map."""
        now = time.perf_counter()
        pos = self.interpolated_positions(now)
        capture = getattr(self.env.cfg, "capture_radius", 0)
        map_img = self.r.render_map(
            self.env.snapshot(), hp=pos[HUNTER], ep=pos[EVADER], anim=now - self.t0,
            trails={k: list(self.trails[k]) for k in ROLES} if self.show_trails else None,
            vision=self.vision, heat=self.heat if self.heat_mode else None, probs=self.probs,
            capture_radius=capture)
        backdrop = self.backdrop(self.hud_state())
        legend = self.legend()
        if legend and legend[2]:
            mx, my = self.layout["map_xy"]
            map_img.paste(legend[0], (legend[1][0] - mx, legend[1][1] - my), legend[0])
        return backdrop, map_img

    def frame(self):
        """Complete window image (used for screenshots and tests)."""
        backdrop, map_img = self.parts()
        out = backdrop.copy()
        out.paste(map_img, self.layout["map_xy"])
        return out

    def legend_position(self, size):
        """Left of the map when the window has room, else the map corner hiding the fewest tiles."""
        L = self.layout
        key = (self.seed, L["map_xy"], self.r.T, size)
        if self._legend_pos is not None and self._legend_pos[0] == key:
            return self._legend_pos[1]
        mx, my = L["map_xy"]
        mw, mh = L["map_size"]
        if mx - L["gap"] - size[0] >= L["margin"]:
            pos = (mx - L["gap"] - size[0], my)
        else:
            T, Hh = self.r.T, self.r.Hh
            solid = self.env.grid == WALL
            padded = np.pad(solid, 1, constant_values=True)
            shown = np.zeros_like(solid)                   # tiles actually drawn (not buried rock)
            for dr in range(3):
                for dc in range(3):
                    shown |= ~padded[dr:dr + solid.shape[0], dc:dc + solid.shape[1]]
            m = int(L["margin"] * 0.6)
            corners = ((mx + m, my + m), (mx + mw - size[0] - m, my + m),
                       (mx + m, my + mh - size[1] - m), (mx + mw - size[0] - m, my + mh - size[1] - m))

            def covered(corner):
                r0, c0 = (corner[1] - my - Hh) // T, (corner[0] - mx) // T
                r1, c1 = (corner[1] + size[1] - my - Hh) // T + 1, (corner[0] + size[0] - mx) // T + 1
                return shown[max(0, r0):max(0, r1), max(0, c0):max(0, c1)].sum()

            pos = min(corners, key=covered)
        self._legend_pos = (key, pos)
        return pos

    def upload(self, name, item, img, always=False):
        """Send an image to Tk; static images only when they changed, the map every frame."""
        photo, last = self.photos.get(name, (None, None))
        if photo is None or photo.width() != img.width or photo.height() != img.height:
            photo = ImageTk.PhotoImage(img)
            self.canvas.itemconfigure(item, image=photo)
        elif always or last is not img:
            photo.paste(img)                           # reuse the Tk image: no per-frame allocation
        self.photos[name] = (photo, img)

    def tick(self):
        t_start = time.perf_counter()
        if not self.paused:
            if self.done:
                if t_start - self.done_at > 2.5:
                    self.load(self.seed + 1)
            elif t_start - self.last_step >= 1.0 / self.sps:
                self.step_once()
        backdrop, map_img = self.parts()
        self.upload("backdrop", self.backdrop_item, backdrop)
        self.upload("map", self.map_item, map_img, always=True)
        self.frame_ms = (time.perf_counter() - t_start) * 1000
        self.root.after(max(1, FRAME_BUDGET_MS - int(self.frame_ms)), self.tick)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hunter", default="chase")
    ap.add_argument("--evader", default="path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sps", type=float, default=5, help="env steps per second")
    ap.add_argument("--tile", type=int, default=32, help="initial tile size in pixels (window is resizable)")
    a = ap.parse_args()
    root = tk.Tk()
    root.title("Dungeon Pursuit")
    root.configure(bg=hex_color(th.UI_BG))
    root.minsize(760, 560)
    Viewer(root, a.hunter, a.evader, a.seed, a.sps, a.tile)
    root.mainloop()


if __name__ == "__main__":
    main()
