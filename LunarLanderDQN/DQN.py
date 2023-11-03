import config
import network
import memoryReplay

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import count

env = gym.make("LunarLander-v2", render_mode="human")

class DQN():
    def __init__(self, memory_replay_capacity=config.MEMORY_REPLAY_CAPACITY, env=env):
        """
        Our DQN implementation of Deep Q-Learning algorithm for Lunar Lander
        """
        self.env = env
        observation, _ = self.env.reset()

        self.replay_memory = memoryReplay.ReplayMemory(capacity=memory_replay_capacity)

        self.policy_network = network.PolicyNetwork(nb_observations=len(observation), nb_actions=self.env.action_space.n).to(config.device)
        self.target_network = network.PolicyNetwork(nb_observations=len(observation), nb_actions=self.env.action_space.n).to(config.device)

        self.loss = None
        self.optimizer = optim.SGD(self.policy_network.parameters(), lr=config.LEARNING_RATE)

        self.steps_done = 0
        self.episode_durations = []

        self.avg_rewards_per_episode = []


    def select_action(self, state):
        sample = np.random.random()
        eps_threshold = config.EPS_END + (config.EPS_START-config.EPS_END)*np.exp(-self.steps_done/config.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_network(state).max(1)[1].view(1, 1) # take the largest expected reward
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=config.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.replay_memory) < config.BATCH_SIZE:
            return
        transitions = self.replay_memory.sample(config.BATCH_SIZE)

        batch = memoryReplay.Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=config.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_network(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(config.BATCH_SIZE, device=config.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * config.DISCOUNT) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 100)
        self.optimizer.step()


    def train(self):
        if torch.cuda.is_available():
            self.num_episodes = 1000
        else:
            self.num_episodes = 50

        for i_episode in range(self.num_episodes):
            # Initialize the environment and get it's state
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=config.device).unsqueeze(0)
            for t in count():
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = env.step(action.item())
                if t == 0:
                    self.avg_rewards_per_episode.append(reward)
                else:
                    self.avg_rewards_per_episode[i_episode] += reward 
                
                reward = torch.tensor([reward], device=config.device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=config.device).unsqueeze(0)

                # Store the transition in memory
                self.replay_memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_network.state_dict()
                policy_net_state_dict = self.policy_network.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*config.TAU + target_net_state_dict[key]*(1-config.TAU)
                self.target_network.load_state_dict(target_net_state_dict)

                if done:
                    self.episode_durations.append(t + 1)
                    self.produce_plot()
                    self.avg_rewards_per_episode[i_episode] /= t+1
                    break

        print('Complete')
        self.plot_avg_rewards_curve()
        self.produce_plot(show_result=True)
        plt.ioff()
        plt.show()
        self.save_weights()

    def save_weights(self):
        self.target_network.save("./weights/target_network_save.pt")
        self.policy_network.save("./weights/policy_network_save.pt")

    def plot_avg_rewards_curve(self):
        plt.plot(np.arange(self.num_episodes), self.avg_rewards_per_episode)

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

        plt.pause(0.001)


if __name__ == '__main__':
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    lunar_lander_agent = DQN()

    lunar_lander_agent.train()