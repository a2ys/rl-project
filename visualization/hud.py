import dataclasses
import re
from dataclasses import dataclass, field
from pathlib import PurePath

from PIL import Image, ImageDraw

from env.generator import SEED_SPLITS
from visualization import theme as th
from visualization.gfx import rounded_panel
from visualization.props import Props
from visualization.sprites import Sprites

HIDDEN_RULE_WORDS = ("reward", "penalty", "bonus", "shaping", "density", "win")
KEY_HELP = (("SPACE", "pause"), ("N P", "next / prev map"), ("R", "restart"), ("S", "step"),
            ("+ -", "speed"), ("V", "vision"), ("T", "trails"), ("H", "heatmap"),
            ("L", "legend"), ("C", "screenshot"), ("Q", "quit"))
KEY_TOGGLES = {"V": "vision", "T": "trails", "L": "legend"}
SPLIT_TEXT = {"train": ("TRAINING MAP", False), "val": ("VALIDATION · UNSEEN", True),
              "test": ("TEST · UNSEEN", True), "custom": ("CUSTOM MAP", False)}
SCRIPTED = {"chase": "Chases the Evader when it can see it, otherwise patrols.",
            "chase_oracle": "Always knows where the Evader is and chases it.",
            "path": "Walks the shortest path to key, door and exit, and steps away "
                    "when the Hunter is within 2 tiles.",
            "random": "Picks a random move every step.",
            "idle": "Never moves."}
RUN_NAMES = {"selfplay": "self-play", "independent": "independent",
             "hunter_vs_path": "trained vs path Evader", "evader_vs_chase": "trained vs chase Hunter",
             "hunter_vs_random": "trained vs random Evader"}
# compass cell (row, col) for UP, DOWN, LEFT, RIGHT, STAY
COMPASS = ((0, 1), (2, 1), (1, 0), (1, 2), (1, 1))


def seed_split(seed):
    for name, rng in SEED_SPLITS.items():
        if isinstance(seed, int) and seed in rng:
            return name
    return "custom"


def rules_from_config(cfg):
    rules = []
    for f in dataclasses.fields(cfg):
        if any(w in f.name for w in HIDDEN_RULE_WORDS):
            continue
        v = getattr(cfg, f.name)
        text = ("yes" if v else "no") if isinstance(v, bool) else f"{v:g}" if isinstance(v, float) else str(v)
        rules.append((f.name.replace("_", " "), text))
    return rules


def describe_agent(spec):
    """(kind line, description or None) for the agent card."""
    kind, _, path = (spec or "").partition(":")
    if not path:
        return f"Scripted · {kind or '-'}", SCRIPTED.get(kind)
    algo = {"dqn": "DQN", "ppogreedy": "PPO greedy"}.get(kind, "PPO")
    run = re.sub(r"_s\d+$", "", re.sub(r"^(check_|dqn_)", "", PurePath(path).parent.name))
    base, cur = run.split("_cur")[0], "_cur" in run
    text = RUN_NAMES.get(base, base.replace("_", " ") or PurePath(path).stem)
    return f"{algo} · {text}{' + curriculum' if cur else ''}", None


@dataclass
class HudState:
    seed: object = "-"
    t: int = 0
    max_steps: int = 200
    hspec: str = ""
    espec: str = ""
    probs: dict = field(default_factory=dict)
    stats: dict = None
    outcome: str = ""
    keys_held: int = 0
    keys_on_map: int = 0
    doors_left: int = 0
    doors_total: int = 0
    dist_hunter: int = None
    dist_evader: int = None
    goal: str = "exit"
    sees: dict = field(default_factory=dict)
    sps: float = None
    paused: bool = False
    heat_tag: str = ""
    rules: list = field(default_factory=list)
    values: dict = field(default_factory=dict)
    events: list = field(default_factory=list)

    @classmethod
    def from_snapshot(cls, snap, hspec="", espec="", probs=None, stats=None, info=None):
        doors = len(snap["doors"])
        s = cls(seed=snap.get("seed", "-"), t=snap["t"], max_steps=snap.get("max_steps", 200),
                hspec=hspec, espec=espec, probs=probs or {}, stats=stats, outcome=snap["outcome"],
                keys_held=snap["keys_held"], keys_on_map=len(snap["keys"]), doors_left=doors,
                doors_total=doors)
        for k, v in (info or {}).items():
            setattr(s, k, v)
        return s

    def key(self):
        probs = {k: None if v is None else tuple(round(float(x), 2) for x in v)
                 for k, v in self.probs.items()}
        values = {k: tuple(round(float(x), 3) for x in v) for k, v in self.values.items()}
        return repr(dataclasses.replace(self, probs=probs, values=values))


def env_info(env):
    hr, er = env.pos["hunter"], env.pos["evader"]
    targets = env.objective_targets()
    return dict(
        dist_hunter=int(env.dist_field({er}, env.cfg.hunter_opens_doors)[hr]),
        dist_evader=int(env.dist_field(targets, env.keys_held > 0)[er]),
        goal="exit" if targets == {env.exit_pos} else "key",
        sees={"hunter": env.sees_other("hunter"), "evader": env.sees_other("evader")},
        doors_total=len(env.dungeon.doors), rules=rules_from_config(env.cfg))


def _px(v, ui):
    return max(1, int(round(v * ui)))


class Pen:
    """Drawing helpers in UI units (scaled by `ui`)."""

    def __init__(self, img, ui):
        self.img, self.d, self.ui = img, ImageDraw.Draw(img), ui

    def p(self, v):
        return _px(v, self.ui)

    def font(self, size, kind="ui"):
        return th.font(self.p(size), kind)

    def text(self, xy, s, size, color=th.TEXT, kind="ui", anchor="ls"):
        self.d.text(xy, s, font=self.font(size, kind), fill=color, anchor=anchor)

    def width(self, s, size, kind="ui"):
        return self.font(size, kind).getlength(s)

    def label(self, xy, s, color=th.TEXT_DIM, size=12.5, right=False):
        """Uppercase tracked label (baseline anchored)."""
        f, track = self.font(size, "label"), self.p(1.3)
        w = sum(f.getlength(ch) for ch in s) + track * (len(s) - 1)
        x, y = (xy[0] - w if right else xy[0]), xy[1]
        for ch in s:
            self.d.text((x, y), ch, font=f, fill=color, anchor="ls")
            x += f.getlength(ch) + track
        return w

    def rule(self, y, x0, x1):
        self.d.line([(x0, y), (x1, y)], fill=th.RULE, width=1)

    def rect(self, box, fill=None, outline=None, width=1):
        self.d.rectangle([int(v) for v in box], fill=fill, outline=outline, width=width)

    def wrap(self, s, size, w, kind="ui"):
        lines, cur = [], ""
        for word in s.split():
            trial = f"{cur} {word}".strip()
            if self.width(trial, size, kind) <= w or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        return lines + ([cur] if cur else [])


def _tiles(n):
    if n is None:
        return "-"
    return "blocked" if n < 0 else f"{n} tile{'s' if n != 1 else ''}"


def situation(s):
    if s.doors_total and s.doors_left == 0:
        return "Door open, Evader runs for the exit", th.EXIT_GREEN
    if s.keys_held:
        return "Evader has the key", th.KEY_GOLD
    if s.doors_left and s.keys_on_map:
        return "Evader is looking for the key", th.EVADER
    return "Evader heads for the exit", th.EXIT_GREEN


def _header(pen, x, y, w, s):
    p = pen.p
    pen.label((x, y), "DUNGEON PURSUIT")
    if s.sps is not None:
        state = "PAUSED" if s.paused else f"{s.sps:.0f} STEPS/S"
        tw = pen.label((x + w, y), state, th.ACCENT if s.paused else th.TEXT_DIM, right=True)
        dot = th.ACCENT if s.paused else th.EXIT_GREEN
        r, cx = p(3), x + w - tw - p(9)
        pen.d.ellipse([cx - r, y - p(7) - r, cx + r, y - p(7) + r], fill=dot)
    y += p(36)
    pen.text((x, y), f"Map {s.seed}", 30, th.TEXT, "display")
    split, unseen = SPLIT_TEXT[seed_split(s.seed)]
    pen.label((x + w, y - p(3)), split, th.ACCENT if unseen else th.TEXT_DIM, right=True)
    y += p(30)

    pen.label((x, y), "EPISODE")
    pen.text((x + w, y), f"{s.t} / {s.max_steps}", 13, th.TEXT, "num", "rs")
    y += p(12)
    th_ = p(4)
    pen.rect((x, y, x + w, y + th_), fill=th.TRACK)
    fill_w = int(w * min(1.0, s.t / max(1, s.max_steps)))
    if fill_w:
        pen.rect((x, y, x + fill_w, y + th_), fill=th.ACCENT)
    next_x = x
    for kind, t in sorted(s.events, key=lambda e: e[1]):
        ex = x + int(w * min(1.0, t / max(1, s.max_steps)))
        col = th.KEY_GOLD if kind == "key" else th.EXIT_GREEN
        pen.rect((ex - p(1), y - p(5), ex + p(1), y + th_ + p(5)), fill=col)
        text = f"{kind} at {t}"
        tw = pen.width(text, 11.5)
        tx = min(max(ex - tw / 2, next_x), x + w - tw)
        pen.text((tx, y + th_ + p(19)), text, 11.5, th.TEXT_DIM)
        next_x = tx + tw + p(12)
    return y + th_ + (p(26) if s.events else p(10))


def _situation(pen, x, y, w, s):
    p = pen.p
    if s.outcome in th.OUTCOME_STYLE:
        title, color, sub = th.OUTCOME_STYLE[s.outcome]
        y += p(30)
        pen.text((x, y), title, 30, color, "display")
        y += p(22)
        pen.text((x, y), f"{sub} at step {s.t}.", 12.5, th.TEXT_DIM)
        y += p(8)
    else:
        text, color = situation(s)
        y += p(22)
        pen.rect((x, y - p(10), x + p(8), y - p(2)), fill=color)
        pen.text((x + p(16), y), text, 15, th.TEXT, "bold")
        y += p(4)

    if s.keys_held:
        key = "carried by Evader"
    elif s.keys_on_map:
        key = "on the floor" if not (s.doors_total and s.doors_left == 0) else "not needed"
    else:
        key = "used" if s.doors_total else "none"
    door = "none" if not s.doors_total else ("open" if s.doors_left == 0 else "locked")
    rows = [("Key", key, None), ("Door", door, None)]
    if s.dist_hunter is not None:
        rows += [("Hunter → Evader", _tiles(s.dist_hunter), None),
                 (f"Evader → {s.goal}", _tiles(s.dist_evader), None)]
    for who, other in (("hunter", "Evader"), ("evader", "Hunter")):
        if who in s.sees:
            rows.append((f"{who.title()} can see {other}", "yes" if s.sees[who] else "no",
                         th.ROLE_COLOR[who] if s.sees[who] else False))
    if s.heat_tag:
        rows.append(("Map overlay", s.heat_tag, None))
    y += p(10)
    for label, val, mark in rows:
        y += p(21)
        pen.text((x, y), label, 12.5, th.TEXT_DIM)
        vx = x + w
        pen.text((vx, y), val, 12.5, th.TEXT, "num", "rs")
        if mark is not None:
            r = p(4)
            mx = vx - pen.width(val, 12.5, "num") - p(10)
            box = (mx - r, y - p(4) - 2 * r + p(1), mx + r, y - p(4) + p(1))
            pen.rect(box, fill=mark if mark else None, outline=th.TEXT_FAINT if not mark else None)
    return y + p(6)


def _compass(pen, x, y, probs, color):
    p = pen.p
    c, gap = p(30), p(2)
    best = max(range(5), key=lambda i: probs[i])
    for i, (r, k) in enumerate(COMPASS):
        cx, cy = x + k * (c + gap), y + r * (c + gap)
        v = float(probs[i])
        pen.rect((cx, cy, cx + c, cy + c), fill=th.mix(th.TRACK, color, min(1.0, v ** 0.75)))
        if i == best:
            pen.rect((cx, cy, cx + c - 1, cy + c - 1), outline=th.TEXT)
        pen.text((cx + c / 2, cy + c / 2), f"{round(v * 100)}", 12, th.TEXT if v >= 0.3 else th.TEXT_DIM,
                 "num", "mm")
    return 3 * c + 2 * gap


def _spark(pen, x, y, w, h, values, color, label):
    p = pen.p
    pen.label((x, y + p(10)), label)
    if values:
        v = values[-1]
        pen.text((x + w, y + p(12)), f"{v:+.2f}", 18, th.TEXT, "num", "rs")
    top, bot = y + p(24), y + h
    if len(values) < 2:
        pen.text((x, bot - p(4)), "collecting…", 11, th.TEXT_FAINT)
        return
    lo, hi = min(values), max(values)
    pad = max(0.05, (hi - lo) * 0.12)
    lo, hi = lo - pad, hi + pad

    def ys(v):
        return bot - (v - lo) / (hi - lo) * (bot - top)
    if lo < 0 < hi:
        zy = ys(0.0)
        for xx in range(int(x), int(x + w), p(6)):
            pen.d.line([(xx, zy), (min(x + w, xx + p(3)), zy)], fill=th.RULE, width=1)
        pen.text((x - p(5), zy), "0", 10, th.TEXT_FAINT, "num", "rm")
    n = len(values)
    pts = [(x + i * w / (n - 1), ys(v)) for i, v in enumerate(values)]
    pen.d.line(pts, fill=color, width=p(2), joint="curve")
    ex, ey = pts[-1]
    r = p(3.5)
    pen.d.ellipse([ex - r - p(2), ey - r - p(2), ex + r + p(2), ey + r + p(2)], fill=th.UI_BG)
    pen.d.ellipse([ex - r, ey - r, ex + r, ey + r], fill=color)


def _agent(pen, x, y, w, role, spec, probs, values):
    p = pen.p
    color = th.ROLE_COLOR[role]
    kind_line, desc = describe_agent(spec)
    y += p(26)
    pen.rect((x, y - p(11), x + p(10), y - p(1)), fill=color)
    pen.text((x + p(18), y), role.title(), 17, th.TEXT, "display")
    pen.text((x + w, y), kind_line, 12, th.TEXT_DIM, "ui", "rs")
    y += p(12)
    if probs is None:
        for line in pen.wrap(desc or "Scripted agent.", 12.5, w):
            y += p(18)
            pen.text((x, y), line, 12.5, th.TEXT_DIM)
        return y + p(4)
    side = _compass(pen, x, y, probs, color)
    algo = "max Q(s, a)" if kind_line.startswith("DQN") else "VALUE V(s)"
    _spark(pen, x + side + p(18), y, w - side - p(18), side - p(14), values or [], color, algo)
    pen.text((x, y + side + p(16)), "move probability, %", 10.5, th.TEXT_FAINT)
    return y + side + p(20)


def _session(pen, x, y, w, stats):
    p = pen.p
    n = stats["n"]
    y += p(26)
    pen.label((x, y), "SESSION")
    pen.text((x + w, y), f"{n} game{'s' if n != 1 else ''}", 12, th.TEXT_DIM, "num", "rs")
    y += p(12)
    segs = [("caught", "Hunter", th.HUNTER), ("escaped", "Evader", th.EVADER),
            ("timeout", "Time out", th.TIMEOUT)]
    bh, gap = p(6), p(2)
    shown = [(k, col) for k, _, col in segs if stats[k]]
    avail = w - gap * max(0, len(shown) - 1)
    bx = x
    if not shown:
        pen.rect((x, y, x + w, y + bh), fill=th.TRACK)
    for i, (k, col) in enumerate(shown):
        sw = avail - (bx - x - gap * i) if i == len(shown) - 1 else int(avail * stats[k] / n)
        pen.rect((bx, y, bx + sw, y + bh), fill=col)
        bx += sw + gap
    y += bh + p(22)
    colw = w / 3
    for i, (k, label, col) in enumerate(segs):
        cx = x + i * colw
        pen.rect((cx, y - p(9), cx + p(8), y - p(1)), fill=col)
        pen.text((cx + p(14), y), label, 12, th.TEXT_DIM)
        pen.text((cx + p(14) + pen.width(label, 12) + p(6), y), str(stats[k]), 13, th.TEXT, "num")
    return y + p(6)


def render_side_panel(width, height, s, ui):
    """One continuous column: header, situation, the two agents, session tally."""
    img = Image.new("RGBA", (width, max(height, 2000)), (0, 0, 0, 0))
    pen = Pen(img, ui)
    p = pen.p
    x, w = p(4), width - p(12)
    y = p(16)
    y = _header(pen, x, y, w, s)
    sections = [lambda yy: _situation(pen, x, yy, w, s),
                lambda yy: _agent(pen, x, yy, w, "hunter", s.hspec, s.probs.get("hunter"),
                                  s.values.get("hunter")),
                lambda yy: _agent(pen, x, yy, w, "evader", s.espec, s.probs.get("evader"),
                                  s.values.get("evader"))]
    if s.stats is not None and s.stats.get("n"):
        sections.append(lambda yy: _session(pen, x, yy, w, s.stats))
    for draw in sections:
        y += p(14)
        pen.rule(y, x, x + w)
        y = draw(y)
    return img.crop((0, 0, width, height))


def key_strip(width, flags, ui):
    """Keyboard shortcuts as one quiet line (wraps if the window is narrow)."""
    fk, fl = th.font(_px(11, ui), "mono"), th.font(_px(11.5, ui))
    gap, inner, line_h = _px(18, ui), _px(7, ui), _px(20, ui)
    items, x, row = [], 0, 0
    for key, label in KEY_HELP:
        iw = fk.getlength(key) + inner + fl.getlength(label)
        if x and x + iw > width:
            x, row = 0, row + 1
        items.append((x, row, key, label))
        x += iw + gap
    img = Image.new("RGBA", (width, line_h * (row + 1)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for x, row, key, label in items:
        on = flags.get(KEY_TOGGLES.get(key, ""), None)
        cy = row * line_h + line_h // 2
        d.text((x, cy), key, font=fk, fill=th.ACCENT if on else th.TEXT_DIM, anchor="lm")
        d.text((x + fk.getlength(key) + inner, cy), label, font=fl,
               fill=th.TEXT_FAINT if on is not False else th.RULE, anchor="lm")
    return img


def legend_panel(ui, show_vision=True, show_trails=True, show_heat=False, capture=0):
    icon = _px(34, ui)
    tile = _px(20, ui)
    sprites, props = Sprites(tile), Props(tile, int(tile * 0.46))
    rows = [("Hunter", sprites.agent("hunter", "down")), ("Evader", sprites.agent("evader", "down")),
            ("Key", sprites.key()), ("Locked door", props.door(True)),
            ("Exit", None), ("Obstacle", props.obstacle("crate"))]
    if show_vision:
        rows.append(("What the agents see", None))
    if show_trails:
        rows.append(("Recent path", None))
    if show_heat:
        rows.append(("Value V(s), low to high", None))
    if capture:
        rows.append(("Capture radius", None))
    row_h = _px(25, ui)
    f = th.font(_px(12, ui))
    pad = _px(12, ui)
    head = _px(26, ui)
    w = pad + icon + _px(10, ui) + max(int(f.getlength(label)) for label, _ in rows) + pad
    h = head + row_h * len(rows) + _px(8, ui)
    img = rounded_panel(w, h, _px(3, ui), th.PANEL_FILL, th.PANEL_EDGE, 1)
    pen = Pen(img, ui)
    pen.label((pad, _px(19, ui)), "LEGEND")
    d = pen.d
    for i, (label, spr) in enumerate(rows):
        cy = head + i * row_h + row_h // 2
        cx = pad + icon // 2
        if label == "Exit":
            spr = props.exit_base()
        if spr is not None:
            if spr.height > row_h:
                k = row_h / spr.height
                spr = spr.resize((max(1, int(spr.width * k)), row_h), Image.LANCZOS)
            img.alpha_composite(spr, (cx - spr.width // 2, cy - spr.height // 2))
        elif label.startswith("What"):
            d.rectangle([cx - _px(12, ui), cy - _px(8, ui), cx + _px(12, ui), cy + _px(8, ui)],
                        fill=(*th.EVADER, 70), outline=(*th.EVADER, 200))
        elif label == "Recent path":
            for k in range(3):
                r = _px(2 + k, ui)
                x0 = cx - _px(10, ui) + k * _px(10, ui)
                d.ellipse([x0 - r, cy - r, x0 + r, cy + r], fill=(*th.HUNTER, 90 + 70 * k))
        elif label.startswith("Value"):
            for k in range(5):
                d.rectangle([cx - _px(15, ui) + k * _px(6, ui), cy - _px(6, ui),
                             cx - _px(10, ui) + k * _px(6, ui), cy + _px(6, ui)], fill=th.heat_color(k / 4))
        elif label == "Capture radius":
            d.ellipse([cx - _px(11, ui), cy - _px(11, ui), cx + _px(11, ui), cy + _px(11, ui)],
                      outline=(*th.HUNTER, 220), width=max(1, _px(2, ui)))
        d.text((pad + icon + _px(10, ui), cy), label, font=f, fill=th.TEXT_DIM, anchor="lm")
    return img
