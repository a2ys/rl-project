"""Small Pillow drawing helpers: supersampled canvases, outlines, glows, panels."""
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

SS = 4  # supersampling factor for hand-drawn sprites


def canvas(w, h):
    """Transparent RGBA canvas at SS x resolution plus its draw handle."""
    im = Image.new("RGBA", (int(w * SS), int(h * SS)), (0, 0, 0, 0))
    return im, ImageDraw.Draw(im)


def downscale(im, w, h):
    return im.resize((int(w), int(h)), Image.LANCZOS)


def outlined(im, color, px):
    """Add a solid silhouette outline of `px` supersampled pixels around the opaque area."""
    alpha = im.getchannel("A")
    grown = alpha.filter(ImageFilter.MaxFilter(2 * px + 1))
    base = Image.new("RGBA", im.size, (*color[:3], 0))
    base.putalpha(grown)
    return Image.alpha_composite(base, im)


def radial_glow(radius, color, intensity=1.0):
    """RGB image (black background) with a soft radial falloff, for additive blending."""
    r = max(2, int(radius))
    ax = np.arange(-r, r + 1, dtype=np.float32)
    d = np.sqrt(ax[None, :] ** 2 + ax[:, None] ** 2) / r
    k = np.clip(1.0 - d, 0, 1) ** 2 * intensity
    arr = k[..., None] * np.array(color[:3], np.float32)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def add_light(frame, glow, cx, cy):
    """Additively blend a glow image centred at (cx, cy); clipped to the frame."""
    gw, gh = glow.size
    x0, y0 = int(cx - gw / 2), int(cy - gh / 2)
    bx0, by0 = max(0, x0), max(0, y0)
    bx1, by1 = min(frame.width, x0 + gw), min(frame.height, y0 + gh)
    if bx1 <= bx0 or by1 <= by0:
        return
    region = frame.crop((bx0, by0, bx1, by1))
    g = glow.crop((bx0 - x0, by0 - y0, bx1 - x0, by1 - y0))
    frame.paste(ImageChops.add(region, g), (bx0, by0))


def rounded_panel(w, h, radius, fill, edge=None, edge_w=2):
    """Anti-aliased rounded rectangle (RGBA), drawn supersampled then downscaled."""
    k = 3
    im = Image.new("RGBA", (w * k, h * k), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, w * k - 1, h * k - 1], radius * k, fill=fill,
                        outline=edge, width=edge_w * k if edge else 0)
    return im.resize((w, h), Image.LANCZOS)
