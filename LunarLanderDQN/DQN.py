import numpy as np
from collections import namedtuple, deque

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    """
    Inspired from:
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#replay-memory

    And it remainds me Linus Torvalds quote:
    "Bad programmers worry about the code. Good programmers worry about data structures and their relationships."
    """
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return np.random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN():
    def __init__(self, nb_outputs, memory_replay_capacity):
        self.nb_outputs = nb_outputs

        self.replay_memory = ReplayMemory(capacity=memory_replay_capacity)
