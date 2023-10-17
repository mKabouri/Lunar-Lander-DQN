import torch
import torch.nn as nn
import torch.functional as F

class Network(nn.Module):
    def __init__(self, nb_observations, nb_actions):
        super().__init__()

    def forward(self, input):
        pass


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")