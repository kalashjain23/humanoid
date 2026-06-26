from torch import nn, Tensor
import torch


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        
        # one learnable log-std per action dim so the policy controls its own exploration
        # initial std = exp(-1.0) ~ 0.37, then it learns to widen or narrow
        self.log_std = nn.Parameter(torch.full((13,), -1.0))
        self.arch = nn.Sequential(
            nn.Linear(137, 256), # imu readings + qpos + qvel (legs + waist)
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 13) # mean values of action for each actuator to be published
        )
        
        # zero-init last layer so initial policy outputs near zero (home pose delta)
        nn.init.zeros_(self.arch[-1].weight)
        nn.init.zeros_(self.arch[-1].bias)
        
    def forward(self, x: Tensor):
        # tanh bounds actions to [-1, 1], safe for delta commands
        return torch.tanh(self.arch(x))
    

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.arch = nn.Sequential(
            nn.Linear(137, 256), # imu readings + qpos + qvel
            nn.ReLU(),
            nn.Linear(256, 256),
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
    