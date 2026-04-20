from torch import nn, Tensor
import torch


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.log_std = nn.Parameter(torch.full((23,), -1.0))
        self.arch = nn.Sequential(
            nn.Linear(168, 256), # imu readings + qpos + qvel
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 23) # mean values of action for each actuator to be published
        )
        
        nn.init.zeros_(self.arch[-1].weight)
        nn.init.zeros_(self.arch[-1].bias)
        
    def forward(self, x: Tensor):
        return torch.tanh(self.arch(x))
    

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.arch = nn.Sequential(
            nn.Linear(168, 256), # imu readings + qpos + qvel
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # value
        )
        
    def forward(self, x: Tensor):
        return self.arch(x)
    