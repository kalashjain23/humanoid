from torch import nn, Tensor
import torch


class Actor(nn.Module):
    def __init__(self):
        super().__init__()

        # learnable log-std per action dim; init std = exp(-1.0) ~ 0.37
        self.log_std = nn.Parameter(torch.full((13,), -1.0))
        self.arch = nn.Sequential(
            nn.Linear(137, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 13)
        )

        # zero-init last layer so the initial policy outputs near-zero home pose deltas
        nn.init.zeros_(self.arch[-1].weight)
        nn.init.zeros_(self.arch[-1].bias)

    def forward(self, x: Tensor):
        return torch.tanh(self.arch(x))
    

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.arch = nn.Sequential(
            nn.Linear(137, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x: Tensor):
        return self.arch(x)
    