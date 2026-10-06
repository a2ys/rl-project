from agents.scripted import Agent


class SB3Agent(Agent):
    """Wraps a trained stable-baselines3 PPO model (object or path to .zip)."""

    def __init__(self, model_or_path, deterministic=True):
        if isinstance(model_or_path, str):
            from stable_baselines3 import PPO
            self.model = PPO.load(model_or_path, device="cpu")
        else:
            self.model = model_or_path
        self.deterministic = deterministic

    def act(self, env, role):
        action, _ = self.model.predict(env.observe(role),
                                       deterministic=self.deterministic)
        return int(action)
