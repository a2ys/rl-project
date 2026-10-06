"""Record an episode as a smooth GIF (same 2.5D look as the live viewer).

python -m visualization.play --hunter chase --evader path --seed 42
python -m visualization.play --hunter random --evader random --seed 42 --out experiments/before.gif
python -m visualization.play --hunter chase --evader ppo:experiments/evader_vs_chase_s0/model --seed 42 --out experiments/after.gif
"""
import argparse
from collections import deque
from pathlib import Path

from agents.factory import make_agent
from env.dungeon_env import EVADER, HUNTER, DungeonPursuitEnv, EnvConfig
from visualization.hud import env_info
from visualization.live import action_probs
from visualization.pretty import PrettyRenderer

ROLES = (HUNTER, EVADER)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hunter", default="chase")
    ap.add_argument("--evader", default="path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--smooth", type=int, default=3, help="frames per env step")
    ap.add_argument("--tile", type=int, default=30)
    ap.add_argument("--out", default="experiments/demo.gif")
    a = ap.parse_args()

    env = DungeonPursuitEnv(EnvConfig())
    agents = {HUNTER: make_agent(a.hunter), EVADER: make_agent(a.evader)}
    rend = PrettyRenderer(a.tile)
    env.reset(seed=a.seed)
    for ag in agents.values():
        ag.reset()
    trails = {k: deque([env.pos[k]], maxlen=14) for k in ROLES}
    stats = dict(n=0, caught=0, escaped=0, timeout=0)
    frames, tick = [], 0

    def draw(pos, probs):
        nonlocal tick
        tick += 1
        frames.append(rend.render(
            env.snapshot(), hp=pos[HUNTER], ep=pos[EVADER], anim=tick * 0.08,
            trails={k: list(trails[k]) for k in ROLES}, probs=probs, stats=None,
            hspec=a.hunter, espec=a.evader, info=env_info(env),
            capture_radius=getattr(env.cfg, "capture_radius", 0)))

    probs = {k: action_probs(agents[k], env, k) for k in ROLES}
    draw(dict(env.pos), probs)
    while True:
        prev = dict(env.pos)
        acts = {k: agents[k].act(env, k) for k in ROLES}
        _, _, term, trunc, info = env.step(acts)
        for k in ROLES:
            trails[k].append(env.pos[k])
        for i in range(1, a.smooth + 1):
            f = i / a.smooth
            f = f * f * (3 - 2 * f)
            draw({k: (prev[k][0] + (env.pos[k][0] - prev[k][0]) * f,
                      prev[k][1] + (env.pos[k][1] - prev[k][1]) * f) for k in ROLES}, probs)
        probs = {k: action_probs(agents[k], env, k) for k in ROLES}
        if term or trunc:
            break
    frames += [frames[-1]] * (a.fps * 2)          # hold last frame ~2s

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(a.out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / a.fps), loop=0, optimize=False)
    frames[-1].save(Path(a.out).with_suffix(".png"))
    print(f"outcome={info['outcome']} length={env.t}  saved {a.out}")


if __name__ == "__main__":
    main()
