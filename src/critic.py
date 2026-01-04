"""
Critic network
"""
import torch
import torch.nn as nn

class CriticNetwork(nn.Module):
    """
    Critic network - estimates value V(s_t)
    """
    def __init__(self, hidden_dim=512, z_dim=1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim+z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, h_t, z_t):
        """
        Args:
            h_t: (batch, hidden_dim)
            z_t: (batch, 32, 32) or (batch, z_dim)
        Returns:
            value: (batch, 1) or (batch,) - expected future return
        """
        if z_t.dim() == 3:
            z_t = z_t.flatten(start_dim=1)
        state = torch.concat([z_t, h_t], dim=-1)
        critic_value = self.mlp(state)
        return critic_value
    