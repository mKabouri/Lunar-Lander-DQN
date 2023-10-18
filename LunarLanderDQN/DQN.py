import config
import network
import memoryReplay

import gymnasium as gym

class DQN():
    def __init__(self, nb_outputs, memory_replay_capacity=config.MEMORY_REPLAY_CAPACITY):
        """
        
        """
        self.nb_outputs = nb_outputs
        self.env = gym.make("LunarLander-v2", render_mode="human")
        observation, info = self.env.reset()

        self.replay_memory = memoryReplay.ReplayMemory(capacity=memory_replay_capacity)
        self.policy_network = network.PolicyNetwork(nb_observations=len(observation), nb_actions=self.env.action_space.n)

    def select_action(self):
        pass

    def train(self):
        pass

    def save_weights(self):
        pass

    def produce_plot(self):
        pass
