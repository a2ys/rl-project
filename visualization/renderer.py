import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from env.generator import WALL, generate_dungeon

COL = dict(wall=(38, 40, 52), floor=(226, 224, 214), door=(150, 95, 45),
           key=(240, 190, 30), exit=(60, 170, 90), hunter=(210, 55, 55),
           evader=(55, 110, 220), bg=(245, 245, 240), text=(30, 30, 30))


def render_frame(snap, tile=24, header=22):
    grid = snap["grid"]
    H, W = grid.shape
    img = Image.new("RGB", (W * tile, H * tile + header), COL["bg"])
    d = ImageDraw.Draw(img)

    def box(r, c, inset=0):
        x0, y0 = c * tile + inset, header + r * tile + inset
        return [x0, y0, x0 + tile - 1 - 2 * inset, y0 + tile - 1 - 2 * inset]

    for r in range(H):
        for c in range(W):
            d.rectangle(box(r, c), fill=COL["wall"] if grid[r, c] == WALL else COL["floor"])
    er, ec = snap["exit"]
    d.rectangle(box(er, ec, 2), fill=COL["exit"])
    d.text((ec * tile + tile // 2 - 3, header + er * tile + tile // 2 - 5), "X", fill=(255, 255, 255))
    for r, c in snap["doors"]:
        d.rectangle(box(r, c, 1), fill=COL["door"])
    for r, c in snap["keys"]:
        d.ellipse(box(r, c, 6), fill=COL["key"])
    hr, hc = snap["hunter"]
    er_, ec_ = snap["evader"]
    d.ellipse(box(hr, hc, 3), fill=COL["hunter"])
    d.ellipse(box(er_, ec_, 6 if (hr, hc) == (er_, ec_) else 3), fill=COL["evader"])
    d.text((4, 5), f"t={snap['t']}  keys held={snap['keys_held']}  {snap['outcome']}",
           fill=COL["text"])
    return img


def render_gallery(seeds, cols=3, tile=24):
    from visualization.pretty import PrettyRenderer
    rend = PrettyRenderer(tile)
    frames = []
    for s in seeds:
        d = generate_dungeon(s)
        frames.append(rend.render(dict(
            grid=d.grid, hunter=d.hunter_start, evader=d.evader_start,
            exit=d.exit_pos, keys=d.keys, doors=d.doors, obstacles=d.obstacles,
            seed=s, t=0, keys_held=0, outcome=""), panel=False))
    w, h = frames[0].size
    rows = (len(frames) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * (w + 8) + 8, rows * (h + 8) + 8), (14, 12, 20))
    for i, f in enumerate(frames):
        canvas.paste(f, (8 + (i % cols) * (w + 8), 8 + (i // cols) * (h + 8)))
    return canvas


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 71, 103, 912, 7, 8])
    ap.add_argument("--out", default="experiments/gallery.png")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    render_gallery(a.seeds).save(a.out)
    print("saved", a.out)
