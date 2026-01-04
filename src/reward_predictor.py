"""
Reward predictor
[z_t, h_t] → (r_mean, r_std)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardPredictor(nn.Module):
    """Predicts reward from model state"""
    def __init__(self, hidden_dim=512, z_dim=1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + z_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(512, 1)
        self.std_head = nn.Linear(512, 1)
    
    def forward(self, h_t, z_t):
        """
        Forward pass
        """
        if z_t.dim() == 3:
            z_t = z_t.flatten(start_dim=1)
        state = torch.cat([h_t, z_t], dim=-1)
        features = self.mlp(state)
        mean = self.mean_head(features).squeeze(-1)  # (batch,)
        std = F.softplus(self.std_head(features)).squeeze(-1) + 1e-5
        return mean, std
    