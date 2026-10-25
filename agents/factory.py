from agents.scripted import ChaseHunter, IdleAgent, PathEvader, RandomAgent


def make_agent(spec: str):
    """random | idle | chase | chase_oracle | path |
    ppo:<model.pt> (samples its policy) | ppogreedy:<model.pt> (argmax) | dqn:<model.pt>
    (pposample: is an alias of ppo:)"""
    spec = spec.strip()
    if spec == "random":
        return RandomAgent()
    if spec == "idle":
        return IdleAgent()
    if spec == "chase":
        return ChaseHunter()
    if spec == "chase_oracle":
        return ChaseHunter(oracle=True)
    if spec == "path":
        return PathEvader()
    kind, _, path = spec.partition(":")
    if kind in ("ppo", "pposample", "ppogreedy", "dqn") and path:
        from agents.learned import LearnedAgent
        return LearnedAgent(path, deterministic=kind == "ppogreedy")
    raise ValueError(f"unknown agent spec: {spec}")
