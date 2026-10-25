# Dungeon Pursuit

Two-agent adversarial reinforcement learning on procedurally generated dungeons.
A **Hunter** tries to catch an **Evader**; the Evader must find a key, open a locked door and reach the exit.
The project provides a Gymnasium-style environment, a dungeon generator, scripted opponents, hand-written PPO and DQN,
curriculum and self-play training, evaluation tools and a 2.5D live viewer.

Research question: does curriculum + self-play training produce more robust and generalizable adversarial agents than naive simultaneous learning?
PPO and DQN are implemented from scratch on PyTorch tensors and autograd (no RL libraries).

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

## Algorithms

All in `rl/`, no RL libraries.

- **PPO** (`rl/ppo.py`, `rl/nets.py`): separate actor and critic MLPs (256-256, tanh, orthogonal init), GAE (gamma 0.99, lambda 0.95), clipped surrogate 0.2, entropy bonus 0.01, value coef 0.5,
  4 epochs of minibatch 512 per rollout, update stops early when approx KL > 1.5 x 0.03, linear LR anneal (3e-4 down to 10%), grad clip 0.5, per-minibatch advantage normalisation.
  Timeouts are treated as terminal (no bootstrap) because the time fraction is part of the observation.
- **DQN** (`rl/dqn.py`): double DQN, Huber loss, hard target sync every 2000 gradient steps, float16 replay buffer (200k), linear epsilon decay 1.0 to 0.05 over the first 30% of training.
- `rl/vec_env.py` runs envs in worker processes, `rl/curriculum.py` holds the levels and schedules, `rl/pool.py` the self-play opponent pool.

## Training modes

`training/train.py --mode ...`

- `single`: one PPO learner against a fixed opponent (scripted spec or a frozen `ppo:<model.pt>`).
- `independent`: both roles learn at the same time from the same games.
- `selfplay`: each role learns against a pool of frozen snapshots of the other role. Snapshot every 50k samples, at most 20 kept, the latest is drawn with p=0.5 otherwise uniform, the opponent is resampled every episode. Half the envs train the Hunter, half the Evader.

`--steps` is the number of samples per learner, so all modes give each learner the same amount of data (selfplay plays 2x the env steps of independent).

### Curriculum

`--curriculum off|linear|adaptive`. Map complexity and scripted-opponent weakness grow together; level 3 is the default `EnvConfig` used for all evaluation. `opp_eps` is the probability that a scripted opponent plays a random action.

| level | size | rooms | doors | obstacles | opp_eps |
|---|---|---|---|---|---|
| 0 | 13 | 3 | 0 | 0.0 | 0.75 |
| 1 | 15 | 3 | 1 | 0.1 | 0.5 |
| 2 | 17 | 4 | 1 | 0.1 | 0.25 |
| 3 | 21 | 5 | 1 | 0.2 | 0.0 |

- `adaptive`: advance when the rate over the last 300 episodes reaches the threshold, after at least 50k samples at the level. The rate is the learner win rate (single, threshold 0.6) or the decisive (non-timeout) rate (two learners, 0.8). Fallback: advance anyway after 20% of the budget at one level.
- `linear`: equal time per level over the first 60% of training.
- `off`: always level 3.
- OOD map config (25x25, 7 rooms, 2 doors, 0.25 obstacles) is never trained on, only used for evaluation.

## Evaluation

- Splits: train / val / test seeds plus an OOD config on the test seeds. Never tune on test.
- PPO agents are evaluated by sampling their policy with a seeded generator (reproducible). The argmax policy is worse under partial observability: on 200 maps the single PPO hunter vs path scores 0.475 val / 0.45 test greedy against 0.68 / 0.705 sampled, and greedy times out 30-39% of games (oscillation).
- Agent specs: `ppo:<model.pt>` sampled, `ppogreedy:<model.pt>` argmax, `dqn:<model.pt>`, plus the scripted ones.
- `evaluation.generalization`: train / val / test / OOD win rates of one agent against a fixed opponent, with the train-test gap and OOD drop.
- `evaluation.exploitability`: freeze a trained agent, train a fresh PPO opponent against it, report how many samples it needs to reach 50% win rate (rolling 200 episodes). Fewer means more exploitable. Several frozen models compare in one run.
- `evaluation.crossplay`: matrix of Hunter win rate, Hunter checkpoint i vs Evader checkpoint j from `ckpt/`. Monotone training gives transitive structure; cycling (typical of naive simultaneous learning) gives non-transitive bands.

## Project layout

```
env/
  generator.py        dungeon generation, validation, ASCII render, seed splits
  dungeon_env.py      EnvConfig and the two-agent environment (rules, observations, rewards)
  wrappers.py         SingleAgentEnv: one role vs a fixed opponent
agents/
  scripted.py         random, idle, chase (and oracle) Hunter, path Evader
  learned.py          LearnedAgent: PPO / DQN checkpoint (.pt) or live net as an agent
  factory.py          make_agent("random|idle|chase|chase_oracle|path|ppo:<m>|ppogreedy:<m>|dqn:<m>")
rl/
  nets.py             actor-critic and Q MLPs, checkpoint save/load, categorical sampling
  ppo.py              rollout buffer, GAE, PPO update
  dqn.py              replay buffer, double DQN
  vec_env.py          vectorised env over worker processes
  curriculum.py       levels, OOD config, off / linear / adaptive schedule
  pool.py             frozen policies and the self-play opponent pool
training/
  train.py            PPO: single / independent / selfplay, curriculum, logs, checkpoints
  train_dqn.py        DQN baseline for one role vs a scripted opponent
evaluation/
  evaluate.py         batched matchup table on a map split -> experiments/baselines.csv
  generalization.py   train / val / test / OOD win rates of one agent
  exploitability.py   train a fresh opponent against a frozen agent
  crossplay.py        checkpoint-vs-checkpoint matrix of a run
visualization/
  live.py             real-time 2.5D viewer window (tkinter + Pillow)
  play.py             record an episode as a GIF
  renderer.py         flat top-down map gallery (PNG)
  heatmap.py          visitation heatmap over many episodes
  plot_training.py    training curves from a run directory
  dashboard.py        static HTML dashboard over all runs
  decision_map.py     action preferences of learned agents on one map
  pretty.py           2.5D renderer: cached static layer, scene overlays, depth-sorted sprites
  tiles.py            static layer: floor, extruded walls, props, baked lighting
  props.py            pillars, crates, door, exit stairway, torches
  sprites.py          agents, key, flame, probability arrows, glows
  gfx.py              Pillow helpers (supersampled canvases, outlines, glows, panels)
  hud.py              info cards, legend, key strip
  theme.py            colours and fonts (single source of truth)
tests/
  test_env.py         generator, rules, capture radius, sprint, guard spawn, Gymnasium API check
  test_rl.py          RL stack tests
experiments/
  run_all.py          full experiment matrix runner (resumable)
  legacy_sb3/         old stable-baselines3 runs, kept for reference
  <run>/              hunter.pt, evader.pt, ckpt/, csv logs, config.json, summary.json
```

## Setup

```
python -m venv venv
venv\Scripts\activate        # Windows; use `source venv/bin/activate` elsewhere
pip install -r requirements.txt
```

The default torch wheel is CPU-only, which is what the project is tuned for. For CUDA install the matching build from pytorch.org.

## Run

```
python -m pytest -q
python -m env.generator                                        # ASCII maps + validation of 1000 train seeds

# full experiment matrix (resumable; finished runs are skipped)
python -m experiments.run_all --dry-run                        # commands + time estimate
python -m experiments.run_all                                  # 3 seeds, 2M samples per learner
python -m experiments.run_all --phase baselines --seeds 0 --steps 1000000

# training (runs go to experiments/<name>_s<seed>/, override with --out)
python -m training.train --mode single --role hunter --steps 1000000           # vs path Evader
python -m training.train --mode single --role evader --curriculum adaptive     # vs chase Hunter
python -m training.train --mode independent --steps 2000000
python -m training.train --mode selfplay --curriculum adaptive --steps 2000000
python -m training.train_dqn --role hunter --steps 1000000

# evaluation (val split by default)
python -m evaluation.evaluate --n-maps 100
python -m evaluation.evaluate --hunter-model experiments/hunter_vs_path_s0/hunter.pt --evader-model experiments/evader_vs_chase_s0/evader.pt
python -m evaluation.generalization --role hunter --agent ppo:experiments/selfplay_cur-adaptive_s0/hunter.pt
python -m evaluation.exploitability --frozen-role hunter --steps 500000 \
    --models experiments/independent_s0/hunter.pt experiments/selfplay_s0/hunter.pt --labels independent selfplay
python -m evaluation.crossplay experiments/independent_s0 --max-ckpts 10

# curves, dashboard, decision maps
python -m visualization.plot_training experiments/selfplay_s0
python -m visualization.dashboard                              # experiments/dashboard.html
python -m visualization.decision_map --hunter ppo:experiments/selfplay_s0/hunter.pt --evader ppo:experiments/selfplay_s0/evader.pt --seed 42

# live viewer (resizable window)
python -m visualization.live --hunter chase --evader path --seed 42
python -m visualization.live --hunter ppo:experiments/selfplay_s0/hunter.pt --evader ppo:experiments/selfplay_s0/evader.pt

# renders
python -m visualization.play --hunter chase --evader path --seed 42 --out experiments/demo.gif
python -m visualization.renderer --seeds 42 71 103 912 7 8     # experiments/gallery.png
python -m visualization.heatmap --hunter random --evader random --seed 42 --episodes 200
```

Each run directory holds `hunter.pt` / `evader.pt` (final models), `ckpt/<role>_<samples>.pt`, `config.json`, `episodes.csv`, `updates.csv`, `eval.csv`, `curriculum.csv` and `summary.json`.

Live viewer keys: `SPACE` pause, `N`/`P` next/previous map, `R` restart, `S` single step (paused), `+`/`-` speed,
`V` vision cones, `T` trails, `H` value heatmap (off / evader / hunter, PPO agents), `L` legend, `C` screenshot, `Q` or `Esc` quit.
For learned agents the panel shows live action probabilities (also drawn as arrows around the agent).

## Training time

Measured on an i5-13500H (12 cores / 16 threads), CPU-only torch, 32 envs on 8 worker processes: about 5.5k-8k env steps/s in rollouts (the old SB3 runs did about 475 steps/s).
Minutes per 1M samples per learner, including evaluation: single 5.5, independent 4.5, selfplay 14, DQN about 15 (estimate).

- Full matrix (3 seeds, 2M samples, exploitability 500k): about 11 h.
- One seed at 1M: about 2 h (`python -m experiments.run_all --seeds 0 --steps 1000000 --exploit-steps 300000`).
- The bottleneck is the Python environment simulation (line-of-sight observations, BFS), not the network, so a GPU (`--device cuda`) gives little speedup. More CPU cores and `--workers` help.

## Results

Win rates on the val split, 100 maps, current rules (`experiments/baselines.csv`). The batched evaluator reproduces these scripted numbers exactly.

| Hunter | Evader | Hunter win | Evader win | Timeout | Mean length |
|---|---|---|---|---|---|
| random | random | 0.03 | 0.00 | 0.97 | 196.9 |
| chase | random | 0.99 | 0.00 | 0.01 | 43.0 |
| random | path | 0.06 | 0.93 | 0.01 | 50.6 |
| chase | path | 0.39 | 0.61 | 0.00 | 44.0 |
| chase_oracle | path | 0.49 | 0.51 | 0.00 | 43.5 |

PPO results are being regenerated with the new stack (`experiments.run_all`); the old stable-baselines3 numbers were dropped. Open `experiments/dashboard.html` for the full results once run_all is done.

Preliminary sanity checks (1M samples, seed 0, 200 maps, sampled policy): the single PPO hunter vs path reaches 0.68 val / 0.705 test.
Adversarial runs (independent and selfplay) reached a win rate of about 0.25-0.33 against the scripted reference at 1M samples (greedy evaluation, to be regenerated).
