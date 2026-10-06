"""Single source of truth for colours and fonts used by every visualization module.

The palette is deliberately narrow: cool desaturated stone, one warm torch accent,
and four signal colours (hunter red, evader blue, key gold, exit green).
"""
from PIL import ImageFont

# --- dungeon stone -----------------------------------------------------------
VOID = (10, 9, 16)                  # solid rock never seen from a floor tile
FLOOR_BASE = (84, 82, 100)
FLOOR_MORTAR = (46, 44, 60)
FLOOR_HILITE = (112, 110, 128)
CRACK = (38, 36, 50)
WALL_TOP = (112, 108, 134)
WALL_TOP_RIM = (168, 164, 192)      # lit edge facing the viewer's far side
WALL_FRONT = (70, 66, 90)
WALL_FRONT_DARK = (46, 43, 62)
MORTAR = (34, 31, 46)

# --- lighting ----------------------------------------------------------------
AMBIENT = (0.62, 0.65, 0.84)        # cool moonlit base multiplier
TORCH_LIGHT = (1.00, 0.62, 0.30)    # warm radial light multiplier
TORCH_FLAME = (255, 168, 64)
EXIT_LIGHT = (0.25, 0.95, 0.60)
VIGNETTE_STRENGTH = 0.55

# --- signal colours ----------------------------------------------------------
HUNTER = (206, 62, 58)
EVADER = (88, 168, 236)
KEY_GOLD = (244, 196, 64)
EXIT_GREEN = (78, 226, 148)
WOOD = (150, 98, 54)
WOOD_DARK = (88, 54, 28)
IRON = (74, 76, 92)
BONE = (232, 222, 200)
OUTLINE = (16, 12, 20)
TIMEOUT = (150, 146, 170)

# --- interface ---------------------------------------------------------------
UI_BG = (14, 13, 22)
PANEL_FILL = (20, 18, 30, 205)
PANEL_EDGE = (92, 84, 120, 150)
CARD_FILL = (34, 31, 48, 190)
TEXT = (236, 232, 244)
TEXT_DIM = (150, 146, 172)
TEXT_FAINT = (104, 100, 128)
ACCENT = (255, 178, 84)
BAR_TRACK = (50, 46, 68)
HEAT_STOPS = ((0.0, (28, 22, 78)), (0.5, (190, 56, 98)), (1.0, (255, 214, 96)))

OUTCOME_STYLE = {"caught": ("CAUGHT", HUNTER, "The Hunter got its prey"),
                 "escaped": ("ESCAPED", EXIT_GREEN, "The Evader reached the exit"),
                 "timeout": ("TIME OUT", TIMEOUT, "Nobody won this round")}

ROLE_COLOR = {"hunter": HUNTER, "evader": EVADER}

_FONTS = {}
_FONT_FILES = {"title": ("georgiab.ttf", "segoeuib.ttf", "arialbd.ttf"),
               "ui": ("bahnschrift.ttf", "segoeui.ttf", "arial.ttf"),
               "bold": ("seguisb.ttf", "segoeuib.ttf", "arialbd.ttf"),
               "mono": ("consola.ttf", "cour.ttf")}


def font(size, kind="ui"):
    """Cached TrueType font with a graceful fallback chain (Windows first)."""
    key = (int(size), kind)
    if key not in _FONTS:
        loaded = None
        for name in _FONT_FILES[kind]:
            try:
                loaded = ImageFont.truetype(name, int(size))
                break
            except OSError:
                continue
        if loaded is None:
            try:
                loaded = ImageFont.load_default(int(size))
            except TypeError:
                loaded = ImageFont.load_default()
        _FONTS[key] = loaded
    return _FONTS[key]


def shade(c, f):
    """Scale an RGB colour's brightness."""
    return tuple(max(0, min(255, int(v * f))) for v in c[:3])


def mix(a, b, t):
    """Linear blend of two RGB colours."""
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def heat_color(v):
    """0..1 -> indigo -> ember -> gold (value heatmap ramp)."""
    v = max(0.0, min(1.0, float(v)))
    for (a, ca), (b, cb) in zip(HEAT_STOPS, HEAT_STOPS[1:]):
        if v <= b:
            return mix(ca, cb, (v - a) / (b - a))
    return HEAT_STOPS[-1][1]
