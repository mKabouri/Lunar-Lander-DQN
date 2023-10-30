import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE=64
MEMORY_REPLAY_CAPACITY=1024 # To have 16 batches
DISCOUNT=0.9
LEARNING_RATE=1e-3
#############################
# For greedy sample selection
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
#############################
