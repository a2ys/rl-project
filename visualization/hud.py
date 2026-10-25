import dataclasses
from dataclasses import dataclass, field

from PIL import Image, ImageDraw

from env.generator import SEED_SPLITS
from visualization import theme as th
from visualization.gfx import rounded_panel
from visualization.props import Props
from visualization.sprites import Sprites

HIDDEN_RULE_WORDS = ("reward", "penalty", "bonus", "shaping", "density", "win")
KEY_HELP = (("SPACE", "pause"), ("N / P", "next / prev map"), ("R", "restart"), ("S", "step"),
            ("+ / -", "speed"), ("V", "vision"), ("T", "trails"), ("H", "heatmap"),
            ("L", "legend"), ("C", "screenshot"), ("Q", "quit"))
KEY_TOGGLES = {"V": "vision", "T": "trails", "L": "legend"}


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
        return repr(dataclasses.replace(self, probs=probs))


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


def _text(d, xy, s, size, color=th.TEXT, kind="ui", anchor="la"):
    d.text(xy, s, font=th.font(size, kind), fill=color, anchor=anchor)


def _card(w, h, ui):
    return rounded_panel(w, h, _px(14, ui), th.CARD_FILL, th.PANEL_EDGE, 1)


def _chip(img, x, y, label, color, ui, size=11):
    f = th.font(_px(size, ui), "bold")
    tw = int(f.getlength(label))
    w, h = tw + _px(14, ui), _px(size + 9, ui)
    pill = rounded_panel(w, h, h // 2, (*color, 70), (*color, 190), 1)
    img.alpha_composite(pill, (x, y))
    ImageDraw.Draw(img).text((x + w // 2, y + h // 2), label, font=f, fill=th.mix(color, (255, 255, 255), 0.55),
                             anchor="mm")
    return w


def _bar(img, x, y, w, h, frac, color):
    img.alpha_composite(rounded_panel(w, h, h // 2, (*th.BAR_TRACK, 255)), (x, y))
    if frac > 0:
        fw = max(h, int(w * min(1.0, frac)))
        img.alpha_composite(rounded_panel(fw, h, h // 2, (*color, 255)), (x, y))


def _dot(d, x, y, r, color):
    d.ellipse([x - r, y - r, x + r, y + r], fill=color)


def title_card(w, s, ui):
    h = _px(112, ui)
    img = _card(w, h, ui)
    d = ImageDraw.Draw(img)
    m = _px(16, ui)
    _text(d, (m, _px(10, ui)), "DUNGEON PURSUIT", _px(23, ui), th.TEXT, "title")
    _text(d, (m, _px(42, ui)), "Hunter vs Evader  ·  procedural dungeons", _px(12, ui), th.TEXT_DIM)
    x = m
    x += _chip(img, x, _px(64, ui), f"SEED {s.seed}", th.ACCENT, ui) + _px(6, ui)
    _chip(img, x, _px(64, ui), seed_split(s.seed).upper(), th.EVADER, ui)
    _text(d, (w - m, _px(66, ui)), f"step {s.t} / {s.max_steps}", _px(13, ui), th.TEXT, "ui", "ra")
    _bar(img, m, h - _px(18, ui), w - 2 * m, _px(7, ui), s.t / max(1, s.max_steps), th.ACCENT)
    return img


def status_headline(s):
    if s.outcome in th.OUTCOME_STYLE:
        title, color, sub = th.OUTCOME_STYLE[s.outcome]
        return f"{title.title()} · {sub}", color
    if s.doors_total and s.doors_left == 0:
        return "Door open · dash for the exit", th.EXIT_GREEN
    if s.keys_held:
        return "Evader has the key", th.KEY_GOLD
    if s.doors_left and s.keys_on_map:
        return "Evader is searching for the key", th.EVADER
    return "Evader heads for the exit", th.EXIT_GREEN


def _tiles(n):
    if n is None:
        return "-"
    return "blocked" if n < 0 else f"{n} tile{'s' if n != 1 else ''}"


def status_card(w, s, ui):
    if s.keys_held:
        key = "held by Evader"
    elif s.keys_on_map:
        key = "on the floor" if not (s.doors_total and s.doors_left == 0) else "not needed"
    else:
        key = "used" if s.doors_total else "none"
    door = "none" if not s.doors_total else ("open" if s.doors_left == 0 else "locked")
    goal = "key" if s.goal == "key" else "exit"
    rows = [("Key", key, th.KEY_GOLD if s.keys_held else th.TEXT),
            ("Door", door, th.EXIT_GREEN if door == "open" else th.TEXT)]
    if s.dist_hunter is not None:
        rows.append(("Hunter to Evader", _tiles(s.dist_hunter), th.HUNTER))
        rows.append((f"Evader to {goal}", _tiles(s.dist_evader), th.EVADER))
    sight = [(who, other) for who, other in (("hunter", "Evader"), ("evader", "Hunter")) if who in s.sees]
    row = _px(21, ui)
    img = _card(w, _px(58, ui) + row * (len(rows) + len(sight)), ui)
    d = ImageDraw.Draw(img)
    m = _px(16, ui)
    text, color = status_headline(s)
    _dot(d, m + _px(4, ui), _px(24, ui), _px(5, ui), color)
    _text(d, (m + _px(16, ui), _px(24, ui)), text, _px(14, ui), th.TEXT, "bold", "lm")
    y = _px(46, ui)
    for label, val, col in rows:
        _text(d, (m, y), label, _px(12, ui), th.TEXT_DIM)
        _text(d, (w - m, y), val, _px(12, ui), col, "bold", "ra")
        y += row
    for who, other in sight:
        _text(d, (m, y), f"{who.title()} sees {other.lower()}", _px(12, ui), th.TEXT_DIM)
        on = bool(s.sees[who])
        _chip(img, w - m - _px(36, ui), y - _px(2, ui), "YES" if on else "NO",
              th.ROLE_COLOR[who] if on else th.TEXT_FAINT, ui, 10)
        y += row
    return img


def agent_card(w, role, spec, probs, ui):
    color = th.ROLE_COLOR[role]
    ppo = probs is not None
    h = _px(98 if ppo else 56, ui)
    img = _card(w, h, ui)
    d = ImageDraw.Draw(img)
    m = _px(16, ui)
    _dot(d, m + _px(4, ui), _px(21, ui), _px(6, ui), color)
    _text(d, (m + _px(18, ui), _px(21, ui)), role.title(), _px(15, ui), th.TEXT, "bold", "lm")
    badge = ("DQN" if spec.startswith("dqn") else "PPO") if ppo else "SCRIPTED"
    _chip(img, w - m - _px(15, ui) - int(th.font(_px(11, ui), "bold").getlength(badge)), _px(11, ui), badge,
          color if ppo else th.TEXT_FAINT, ui, 10)
    name = spec.split(":", 1)[-1] if spec else "-"
    name = name if len(name) <= 30 else "..." + name[-27:]
    _text(d, (m + _px(18, ui), _px(40, ui)), name, _px(11, ui), th.TEXT_DIM, "ui", "lm")
    if ppo:
        n, gap = 5, _px(8, ui)
        bw = (w - 2 * m - gap * (n - 1)) // n
        base, top = h - _px(18, ui), _px(54, ui)
        for i, lab in enumerate("UDLRS"):
            x = m + i * (bw + gap)
            img.alpha_composite(rounded_panel(bw, base - top, _px(3, ui), (*th.BAR_TRACK, 255)), (x, top))
            fh = max(2, int((base - top) * float(probs[i])))
            img.alpha_composite(rounded_panel(bw, fh, _px(3, ui), (*color, 255)), (x, base - fh))
            _text(d, (x + bw // 2, h - _px(8, ui)), lab, _px(10, ui), th.TEXT_DIM, "ui", "mm")
    return img


def tally_card(w, stats, ui):
    h = _px(86, ui)
    img = _card(w, h, ui)
    d = ImageDraw.Draw(img)
    m = _px(16, ui)
    n = stats["n"] if stats else 0
    _text(d, (m, _px(10, ui)), "SESSION", _px(11, ui), th.TEXT_DIM, "bold")
    _text(d, (w - m, _px(10, ui)), f"{n} episode{'s' if n != 1 else ''}", _px(11, ui), th.TEXT_DIM, "ui", "ra")
    segs = [("caught", "Hunter", th.HUNTER), ("escaped", "Evader", th.EVADER), ("timeout", "Timeout", th.TIMEOUT)]
    bx, by, bw, bh = m, _px(30, ui), w - 2 * m, _px(9, ui)
    img.alpha_composite(rounded_panel(bw, bh, bh // 2, (*th.BAR_TRACK, 255)), (bx, by))
    x = bx
    for i, (k, _, col) in enumerate(segs):
        v = stats[k] if stats else 0
        sw = int(bw * v / n) if n else 0
        if sw > 0:
            img.alpha_composite(rounded_panel(sw, bh, bh // 2 if len(segs) == 1 else _px(2, ui),
                                              (*col, 255)), (x, by))
            x += sw
    colw = (w - 2 * m) // 3
    for i, (k, label, col) in enumerate(segs):
        cx = m + i * colw
        _text(d, (cx, _px(48, ui)), str(stats[k] if stats else 0), _px(20, ui), col, "bold")
        _text(d, (cx, _px(48 + 24, ui)), label, _px(10, ui), th.TEXT_DIM)
    return img


def rules_card(w, rules, ui):
    row = _px(19, ui)
    lines = (len(rules) + 1) // 2
    h = _px(36, ui) + row * lines
    img = _card(w, h, ui)
    d = ImageDraw.Draw(img)
    m = _px(16, ui)
    _text(d, (m, _px(10, ui)), "RULES", _px(11, ui), th.TEXT_DIM, "bold")
    colw = (w - 2 * m) // 2
    for i, (label, val) in enumerate(rules):
        x = m + (i % 2) * colw
        y = _px(30, ui) + (i // 2) * row
        _text(d, (x, y), label, _px(11, ui), th.TEXT_DIM)
        _text(d, (x + colw - _px(22, ui), y), val, _px(11, ui), th.TEXT, "bold", "ra")
    return img


def render_side_panel(width, height, s, ui):
    w = width - _px(24, ui)
    gap = _px(10, ui)
    cards = [title_card(w, s, ui), status_card(w, s, ui),
             agent_card(w, "hunter", s.hspec, s.probs.get("hunter"), ui),
             agent_card(w, "evader", s.espec, s.probs.get("evader"), ui)]
    if s.stats is not None:
        cards.append(tally_card(w, s.stats, ui))
    if s.rules:
        cards.append(rules_card(w, s.rules, ui))
    while len(cards) > 4 and sum(c.height for c in cards) + gap * len(cards) > height - gap:
        cards.pop()
    out = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    y = gap // 2 + _px(4, ui)
    for c in cards:
        out.alpha_composite(c, (_px(12, ui), y))
        y += c.height + gap
    return out


def status_strip(width, s, ui):
    h = _px(26, ui)
    img = Image.new("RGBA", (width, h), (0, 0, 0, 0))
    x = 0
    x += _chip(img, x, 0, "PAUSED" if s.paused else "PLAYING", th.ACCENT if s.paused else th.EXIT_GREEN, ui) + _px(6, ui)
    if s.sps is not None:
        x += _chip(img, x, 0, f"{s.sps:.0f} STEPS/S", th.TEXT_DIM, ui) + _px(6, ui)
    if s.heat_tag:
        _chip(img, x, 0, s.heat_tag.upper(), th.KEY_GOLD, ui)
    return img


def key_strip(width, flags, ui):
    f_key, f_lab = th.font(_px(11, ui), "bold"), th.font(_px(11, ui))
    pad, gap = _px(14, ui), _px(14, ui)
    h = _px(34, ui)
    img = rounded_panel(width, h, _px(12, ui), th.PANEL_FILL, th.PANEL_EDGE, 1)
    d = ImageDraw.Draw(img)
    x = pad
    for key, label in KEY_HELP:
        kw = int(f_key.getlength(key)) + _px(12, ui)
        on = flags.get(KEY_TOGGLES.get(key.strip(), ""), None)
        cap = rounded_panel(kw, _px(20, ui), _px(5, ui), (*th.BAR_TRACK, 255),
                            (*(th.ACCENT if on else th.PANEL_EDGE[:3]), 220), 1)
        img.alpha_composite(cap, (x, (h - cap.height) // 2))
        d.text((x + kw // 2, h // 2), key, font=f_key, fill=th.TEXT, anchor="mm")
        x += kw + _px(6, ui)
        d.text((x, h // 2), label, font=f_lab, fill=th.TEXT_DIM if on is not False else th.TEXT_FAINT, anchor="lm")
        x += int(f_lab.getlength(label)) + gap
    return img


def legend_panel(ui, show_vision=True, show_trails=True, show_heat=False, capture=0):
    icon = _px(38, ui)
    tile = _px(21, ui)
    sprites, props = Sprites(tile), Props(tile, int(tile * 0.46))
    rows = [("Hunter", sprites.agent("hunter", "down")), ("Evader", sprites.agent("evader", "down")),
            ("Key", sprites.key()), ("Locked door", props.door(True)),
            ("Exit", None), ("Obstacle", props.obstacle("crate"))]
    if show_vision:
        rows.append(("Vision cone", None))
    if show_trails:
        rows.append(("Trails", None))
    if show_heat:
        rows.append(("Value heatmap", None))
    if capture:
        rows.append(("Capture radius", None))
    row_h = _px(27, ui)
    f = th.font(_px(12, ui))
    w = icon + _px(34, ui) + max(int(f.getlength(label)) for label, _ in rows)
    h = row_h * len(rows) + _px(14, ui)
    img = rounded_panel(w, h, _px(12, ui), th.PANEL_FILL, th.PANEL_EDGE, 1)
    d = ImageDraw.Draw(img)
    for i, (label, spr) in enumerate(rows):
        cy = _px(7, ui) + i * row_h + row_h // 2
        cx = _px(10, ui) + icon // 2
        if label == "Exit":
            spr = props.exit_base()
        if spr is not None:
            if spr.height > row_h:
                k = row_h / spr.height
                spr = spr.resize((max(1, int(spr.width * k)), row_h), Image.LANCZOS)
            img.alpha_composite(spr, (cx - spr.width // 2, cy - spr.height // 2))
        elif label == "Vision cone":
            img.alpha_composite(rounded_panel(_px(26, ui), _px(18, ui), 5, (*th.EVADER, 90), (*th.EVADER, 200), 1),
                                (cx - _px(13, ui), cy - _px(9, ui)))
        elif label == "Trails":
            for k in range(3):
                _dot(d, cx - _px(10, ui) + k * _px(10, ui), cy, _px(2 + k, ui), (*th.HUNTER, 90 + 70 * k))
        elif label == "Value heatmap":
            for k in range(5):
                d.rectangle([cx - _px(15, ui) + k * _px(6, ui), cy - _px(6, ui),
                             cx - _px(10, ui) + k * _px(6, ui), cy + _px(6, ui)], fill=th.heat_color(k / 4))
        elif label == "Capture radius":
            d.ellipse([cx - _px(11, ui), cy - _px(11, ui), cx + _px(11, ui), cy + _px(11, ui)],
                      outline=(*th.HUNTER, 220), width=max(1, _px(2, ui)))
        d.text((icon + _px(20, ui), cy), label, font=f, fill=th.TEXT, anchor="lm")
    return img
