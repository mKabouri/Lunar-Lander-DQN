import config
import network
import memoryReplay

import gymnasium as gym
import numpy as np
import matplotlib as plt
import torch
import torch.optim as optim

env = gym.make("LunarLander-v2", render_mode="human")

class DQN():
    def __init__(self, nb_outputs, memory_replay_capacity=config.MEMORY_REPLAY_CAPACITY, env=env):
        """
        Our DQN implementation of Deep Q-Learning algorithm for Lunar Lander
        """
        self.nb_outputs = nb_outputs
        self.env = env
        observation, _ = self.env.reset()

        self.replay_memory = memoryReplay.ReplayMemory(capacity=memory_replay_capacity)

        self.policy_network = network.PolicyNetwork(nb_observations=len(observation), nb_actions=self.env.action_space.n).to(config.device)
        self.target_network = network.PolicyNetwork(nb_observations=len(observation), nb_actions=self.env.action_space.n).to(config.device)

        self.optimizer = optim.SGD(self.policy_network.parameters(), lr=config.LEARNING_RATE)

        self.steps_done = 0
        self.episode_durations = []


    def select_action(self, state):
        sample = np.random.random()
        eps_threshold = config.EPS_END + (config.EPS_START-config.EPS_END)*np.exp(-self.steps_done/config.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_network(state).max(1)[1].view(1, 1) # take the largest expected reward
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=config.device, dtype=torch.long)

    def train(self):
        pass

    def save_weights(self):
        pass

    def produce_plot(self, show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        # if is_ipython:
        #     if not show_result:
        #         display.display(plt.gcf())
        #         display.clear_output(wait=True)
        #     else:
        #         display.display(plt.gcf())
