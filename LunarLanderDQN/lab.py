import gymnasium as gym

env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset()

for _ in range(100):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.render()
    if truncated or terminated:
        observation, info = env.reset()



print("\nExperiments:")
print("#####################################")
print(observation)
print("reward_range: ", env.metadata)
print("Action space: ", env.action_space)
print("Observation space (low): ", env.observation_space.low)
print("Observation space (high): ", env.observation_space.high)
print("Observation space (dtype): ", env.observation_space.dtype)
# Get the number of discrete actions
num_actions = env.action_space.n
print(type(num_actions))
print(type(env.observation_space))
# List the possible actions and their indices
possible_actions = list(range(num_actions))
print("Number of discrete actions:", num_actions)
print("Possible actions and their indices:", possible_actions)
