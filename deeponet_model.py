import torch
import torch.nn as nn

class DeepONet(nn.Module):
    def __init__(self, n_sensors, hidden_dim, p):
        super(DeepONet, self).__init__()
        self.branch = nn.Sequential(
            nn.Linear(n_sensors, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, p)
        )
        self.trunk = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, p)
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, f_obs, grid_coords):
        B = self.branch(f_obs) # [b, p]
        T = self.trunk(grid_coords) # [b, n, p]
        return torch.einsum("bp,bnp->bn", B, T) + self.bias