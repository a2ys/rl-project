from PIL import ImageFont

VOID = (10, 9, 16)
FLOOR_BASE = (84, 82, 100)
FLOOR_MORTAR = (46, 44, 60)
FLOOR_HILITE = (112, 110, 128)
CRACK = (38, 36, 50)
WALL_TOP = (112, 108, 134)
WALL_TOP_RIM = (168, 164, 192)
WALL_FRONT = (70, 66, 90)
WALL_FRONT_DARK = (46, 43, 62)
MORTAR = (34, 31, 46)

AMBIENT = (0.62, 0.65, 0.84)
TORCH_LIGHT = (1.00, 0.62, 0.30)
TORCH_FLAME = (255, 168, 64)
EXIT_LIGHT = (0.25, 0.95, 0.60)
VIGNETTE_STRENGTH = 0.55

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

# UI: one quiet instrument column on cool graphite; amber is the only accent (torchlight)
UI_BG = (12, 13, 18)
RULE = (38, 40, 52)
TRACK = (30, 32, 42)
TEXT = (230, 231, 236)
TEXT_DIM = (146, 149, 163)
TEXT_FAINT = (92, 95, 111)
ACCENT = (232, 168, 82)
PANEL_FILL = (12, 13, 18, 228)
PANEL_EDGE = (*RULE, 255)
BAR_TRACK = TRACK
HEAT_STOPS = ((0.0, (28, 22, 78)), (0.5, (190, 56, 98)), (1.0, (255, 214, 96)))

OUTCOME_STYLE = {"caught": ("Caught", HUNTER, "The Hunter caught the Evader"),
                 "escaped": ("Escaped", EXIT_GREEN, "The Evader reached the exit"),
                 "timeout": ("Time out", TIMEOUT, "Nobody won within 200 steps")}

ROLE_COLOR = {"hunter": HUNTER, "evader": EVADER}

_FONTS = {}
# kind -> (file, variable-font instance or None); first file that loads wins
_FONT_FILES = {"display": (("bahnschrift.ttf", "Bold SemiCondensed"), ("arialbd.ttf", None)),
               "label": (("bahnschrift.ttf", "SemiBold Condensed"), ("arialbd.ttf", None)),
               "num": (("bahnschrift.ttf", "SemiBold"), ("arialbd.ttf", None)),
               "ui": (("segoeui.ttf", None), ("arial.ttf", None)),
               "bold": (("seguisb.ttf", None), ("arialbd.ttf", None)),
               "mono": (("CascadiaMono.ttf", None), ("consola.ttf", None), ("cour.ttf", None))}
_FONT_FILES["title"] = _FONT_FILES["display"]


def font(size, kind="ui"):
    key = (int(size), kind)
    if key not in _FONTS:
        loaded = None
        for name, inst in _FONT_FILES[kind]:
            try:
                loaded = ImageFont.truetype(name, int(size))
                if inst:
                    loaded.set_variation_by_name(inst)
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
    return tuple(max(0, min(255, int(v * f))) for v in c[:3])


def mix(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def heat_color(v):
    v = max(0.0, min(1.0, float(v)))
    for (a, ca), (b, cb) in zip(HEAT_STOPS, HEAT_STOPS[1:]):
        if v <= b:
            return mix(ca, cb, (v - a) / (b - a))
    return HEAT_STOPS[-1][1]
