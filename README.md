# Dungeon Pursuit

Two-agent adversarial reinforcement learning on procedurally generated dungeons.
A **Hunter** tries to catch an **Evader**; the Evader must find a key, open a locked door and reach the exit.
The project provides a Gymnasium-style environment, a dungeon generator, scripted opponents, PPO baselines,
evaluation tools and a 2.5D live viewer.

![live viewer](experiments/live_large.png)

## Game rules

- 21x21 grid, simultaneous moves, 200 steps max. Actions: UP, DOWN, LEFT, RIGHT, STAY (picking up keys and opening doors is automatic).
- Evader wins by reaching the exit. Hunter wins by catching the Evader (same cell, swap, or within `capture_radius`). Otherwise timeout.
- A locked door gates the exit. The Evader needs the key (it is consumed by the door); the Hunter walks through doors freely (`hunter_opens_doors`).
- Both agents see only a 9x9 line-of-sight window; walls, closed doors and obstacles block sight.
- Equal speed alone made the Evader win about 99% of games, so the Hunter has structural advantages, all `EnvConfig` fields:

| field | default | effect |
|---|---|---|
| `hunter_exit_dist` | 6 | guard spawn: Hunter starts within 6 path tiles of the exit (`None` = anywhere far) |
| `hunter_min_dist` | 8 | Hunter starts at least this many path tiles from the Evader |
| `hunter_sprint_every` | 40 | Hunter repeats its move on every 40th step (bonus move; 0 = off) |
| `capture_radius` | 0 | caught if Manhattan distance <= r after the move, also applied to the swap rule (1 is much stronger) |

### Generator (`env/generator.py`)

Deterministic per seed. Rooms are placed with a wall margin, connected by L-shaped corridors, and validated for solvability.

- **Door and key:** the first door sits on a corridor cell that disconnects the exit from the Evader start. Keys are only placed in the region reachable with doors closed, at least `key_min_dist=6` path tiles from the Evader.
- **Obstacles:** each room with a 2x2+ interior is furnished with probability `min(1, 4 * obstacle_density)` using one template: corner crates, L-shaped crate stack, symmetric inset pillars, or colonnade. The outer ring of a room and the corridor lanes crossing it stay clear, and a tile is only accepted if the map stays connected.
- **Spawns:** Evader in a random room on a tile not touching an obstacle. Exit in the room farthest from it (same tile rule). Hunter at path distance [`hunter_min_dist`, 30] from the Evader and within `hunter_exit_dist` of the exit; constraints relax in that order if no tile qualifies.
- Seed splits: train 0-999, val 10000-10199, test 20000-20499. Never tune on test.

## Observation, actions, rewards

- Observation (496 floats): 9x9 line-of-sight window x 6 channels (wall, door, key, exit, other agent, visible) + 10 scalars (position, keys held, time fraction, target direction/distance, other-agent visibility and relative position). Hunter target = Evader when visible; Evader target = key (if a door blocks the exit) else exit.
- Rewards: win +1 / lose -1, step -0.01, path-distance shaping 0.01 per tile of progress, Evader +0.1 for the key and +0.1 for opening a door.
- API: `env.reset(seed=s)` plays dungeon `s`; `env.step({HUNTER: a, EVADER: b})` returns per-role dicts. `env.wrappers.SingleAgentEnv` wraps one role against a fixed opponent as a standard Gymnasium env.

## Project layout

```
env/
  generator.py        dungeon generation, validation, ASCII render, seed splits
  dungeon_env.py      EnvConfig and the two-agent environment (rules, observations, rewards)
  wrappers.py         SingleAgentEnv: one role vs a fixed opponent
agents/
  scripted.py         random, idle, chase (and oracle) Hunter, path Evader
  ppo_agent.py        SB3Agent wrapper for trained PPO models
  factory.py          make_agent("random|chase|chase_oracle|path|ppo:<model>|pposample:<model>")
training/train.py     PPO training for one role against a scripted opponent
evaluation/
  evaluate.py         matchup table on a map split -> experiments/baselines.csv
  generalization.py   train / val / test win rates of one model
visualization/
  live.py             real-time 2.5D viewer window (tkinter + Pillow)
  play.py             record an episode as a GIF
  renderer.py         flat top-down map gallery (PNG)
  heatmap.py          visitation heatmap over many episodes
  plot_training.py    training curves from a run directory
  pretty.py           2.5D renderer: cached static layer, scene overlays, depth-sorted sprites
  tiles.py            static layer: floor, extruded walls, props, baked lighting
  props.py            pillars, crates, door, exit stairway, torches
  sprites.py          agents, key, flame, probability arrows, glows
  gfx.py              Pillow helpers (supersampled canvases, outlines, glows, panels)
  hud.py              info cards, legend, key strip
  theme.py            colours and fonts (single source of truth)
tests/test_env.py     generator, rules, capture radius, sprint, guard spawn, Gymnasium API check
experiments/          trained models, curves, baselines.csv, images and GIFs
```

## Setup

```
python -m venv venv
venv\Scripts\activate        # Windows; use `source venv/bin/activate` elsewhere
pip install -r requirements.txt
```

## Run

```
python -m pytest -q
python -m env.generator                                        # ASCII maps + validation of 1000 train seeds

# live viewer (resizable window)
python -m visualization.live --hunter chase --evader path --seed 42
python -m visualization.live --hunter chase --evader ppo:experiments/evader_vs_chase_s0/model

# renders
python -m visualization.play --hunter chase --evader path --seed 42 --out experiments/demo.gif
python -m visualization.renderer --seeds 42 71 103 912 7 8     # experiments/gallery.png
python -m visualization.heatmap --hunter random --evader random --seed 42 --episodes 200

# evaluation (val split by default)
python -m evaluation.evaluate --n-maps 100
python -m evaluation.evaluate --hunter-model experiments/hunter_vs_path_s0/model --evader-model experiments/evader_vs_chase_s0/model
python -m evaluation.generalization --role hunter --model experiments/hunter_vs_path_s0/model

# training and curves
python -m training.train --role hunter --steps 300000                          # vs path Evader
python -m training.train --role hunter --opponent random --steps 300000
python -m training.train --role evader --steps 600000                          # vs chase Hunter
python -m visualization.plot_training experiments/hunter_vs_path_s0
```

Live viewer keys: `SPACE` pause, `N`/`P` next/previous map, `R` restart, `S` single step (paused), `+`/`-` speed,
`V` vision cones, `T` trails, `H` value heatmap (off / evader / hunter, PPO agents), `L` legend, `C` screenshot, `Q` or `Esc` quit.
For PPO agents the panel shows live action probabilities (also drawn as arrows around the agent).

## Results

Win rates on the val split, 100 maps, current rules (`experiments/baselines.csv`).

| Hunter | Evader | Hunter win | Evader win | Timeout | Mean length |
|---|---|---|---|---|---|
| random | random | 0.03 | 0.00 | 0.97 | 196.9 |
| chase | random | 0.99 | 0.00 | 0.01 | 43.0 |
| random | path | 0.06 | 0.93 | 0.01 | 50.6 |
| chase | path | 0.39 | 0.61 | 0.00 | 44.0 |
| chase_oracle | path | 0.49 | 0.51 | 0.00 | 43.5 |
| PPO hunter (vs path) | path | 0.43 | 0.21 | 0.36 | 114.1 |
| PPO hunter (vs path) | random | 0.00 | 0.00 | 1.00 | 200.0 |
| PPO hunter (vs random) | path | 0.17 | 0.78 | 0.05 | 53.3 |
| PPO hunter (vs random) | random | 0.09 | 0.00 | 0.91 | 191.2 |
| chase | PPO evader (vs chase) | 0.49 | 0.42 | 0.09 | 74.5 |
| random | PPO evader (vs chase) | 0.01 | 0.11 | 0.88 | 185.1 |

Notes: the PPO hunter trained against `path` reaches 0.43 on val (0.40 on test) but only walks toward an Evader it can see, so it times out against a random Evader.
The PPO hunter trained against a random Evader is weak (sparse catch reward, nothing to follow until the Evader is in view).
Models were trained under the current rules; the previous models are kept as `model_old.zip` in each run directory.
