import gymnasium as gym

class LUNARLANDER_agent():
    def __init__(self):
        self.env = gym.make("LunarLander-v2", render_mode="human")