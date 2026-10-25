from collections import deque
from dataclasses import replace

from env.dungeon_env import EnvConfig

# Map complexity and scripted-opponent skill (1 - opp_eps) grow together.
# The last level is exactly the default EnvConfig used for all evaluation.
LEVELS = [
    dict(size=13, n_rooms=3, n_doors=0, obstacle_density=0.0,
         hunter_exit_dist=4, hunter_min_dist=5, opp_eps=0.75),
    dict(size=15, n_rooms=3, n_doors=1, obstacle_density=0.1,
         hunter_exit_dist=4, hunter_min_dist=6, opp_eps=0.5),
    dict(size=17, n_rooms=4, n_doors=1, obstacle_density=0.1,
         hunter_exit_dist=5, hunter_min_dist=7, opp_eps=0.25),
    dict(size=21, n_rooms=5, n_doors=1, obstacle_density=0.2,
         hunter_exit_dist=6, hunter_min_dist=8, opp_eps=0.0),
]

# Out-of-distribution maps: larger, more rooms, two doors. Never trained on.
OOD = dict(size=25, n_rooms=7, n_doors=2, obstacle_density=0.25,
           hunter_exit_dist=6, hunter_min_dist=8)


def level_cfg(level, base: EnvConfig | None = None):
    over = {k: v for k, v in level.items() if k != "opp_eps"}
    return replace(base or EnvConfig(), **over)


def ood_cfg(base: EnvConfig | None = None):
    return replace(base or EnvConfig(), **OOD)


class Curriculum:
    """Picks the training level.

    off       always the final level
    linear    equal time per level over the first `ramp` fraction of training
    adaptive  advance when the success rate over the last `window` episodes reaches
              `threshold` (after `min_steps` at the level), or after `max_frac` of the
              budget at the level as a fallback
    """

    def __init__(self, mode, total_steps, levels=LEVELS, threshold=0.6, window=300,
                 min_steps=50_000, max_frac=0.2, ramp=0.6):
        assert mode in ("off", "linear", "adaptive")
        self.mode, self.levels, self.total = mode, levels, total_steps
        self.threshold, self.min_steps = threshold, min_steps
        self.max_steps = int(max_frac * total_steps)
        self.ramp = ramp
        self.level = len(levels) - 1 if mode == "off" else 0
        self.level_start = 0
        self.recent = deque(maxlen=window)
        self.history = [(0, self.level)]

    @property
    def final(self):
        return self.level == len(self.levels) - 1

    def observe(self, success):
        self.recent.append(float(success))

    def rate(self):
        return sum(self.recent) / len(self.recent) if self.recent else 0.0

    def update(self, timesteps):
        """Returns True if the level changed."""
        if self.final or self.mode == "off":
            return False
        if self.mode == "linear":
            per = self.ramp * self.total / (len(self.levels) - 1)
            target = min(len(self.levels) - 1, int(timesteps // per))
            advance = target > self.level
        else:
            at_level = timesteps - self.level_start
            full = len(self.recent) == self.recent.maxlen
            advance = (at_level >= self.min_steps and full and self.rate() >= self.threshold) \
                or at_level >= self.max_steps
        if advance:
            self.level += 1
            self.level_start = timesteps
            self.recent.clear()
            self.history.append((timesteps, self.level))
        return advance
