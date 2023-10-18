import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    """
    Our Brain
    """
    def __init__(self, nb_observations, nb_actions):
        super().__init__()
        self.DQN_network = nn.Sequential([
            nn.Linear(nb_observations, 128),  
            F.relu(),
            nn.Linear(128, 128),
            F.relu(),
            nn.Linear(128, nb_actions)
        ])

    def forward(self, input):
        return self.DQN_network(input)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")