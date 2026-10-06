from agents.scripted import ChaseHunter, IdleAgent, PathEvader, RandomAgent


def make_agent(spec: str):
    """random | idle | chase | chase_oracle | path | ppo:<model path> | pposample:<model path>"""
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
    if spec.startswith("ppo:"):
        from agents.ppo_agent import SB3Agent
        return SB3Agent(spec[4:], deterministic=True)
    if spec.startswith("pposample:"):
        from agents.ppo_agent import SB3Agent
        return SB3Agent(spec[10:], deterministic=False)
    raise ValueError(f"unknown agent spec: {spec}")
