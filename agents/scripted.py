from env.dungeon_env import EVADER, HUNTER, STAY


class Agent:
    def reset(self):
        pass

    def act(self, env, role) -> int:
        raise NotImplementedError


class RandomAgent(Agent):
    def act(self, env, role):
        return int(env.rng.integers(5))


class IdleAgent(Agent):
    def act(self, env, role):
        return STAY


class ChaseHunter(Agent):
    def __init__(self, oracle=False):
        self.oracle = oracle
        self.target = None

    def reset(self):
        self.target = None

    def act(self, env, role=HUNTER):
        me = env.pos[HUNTER]
        if self.oracle or env.sees_other(HUNTER):
            self.target = env.pos[EVADER]
        if self.target is None or me == self.target:
            self.target = env.random_floor_tile()
        return env.first_action_toward(me, {self.target}, env.cfg.hunter_opens_doors)


class PathEvader(Agent):
    def __init__(self, flee_radius=2):
        self.flee_radius = flee_radius

    def act(self, env, role=EVADER):
        me = env.pos[EVADER]
        a = env.first_action_toward(me, env.objective_targets(), env.keys_held > 0)
        if self.flee_radius > 0 and env.sees_other(EVADER):
            h = env.pos[HUNTER]
            if abs(me[0] - h[0]) + abs(me[1] - h[1]) <= self.flee_radius:
                def score(k):
                    p, _ = env.resolve_move(EVADER, k)
                    return (abs(p[0] - h[0]) + abs(p[1] - h[1]), k == a)
                a = max(range(5), key=score)
        return a
