"""
Actor network
Input: [h_t, z_t] concatenated
Output: Action distribution π(a_t | s_t)
- For discrete actions (Atari): Categorical over 18 actions
- For continuous: Gaussian (mean, std)
"""
import torch
import torch.nn as nn

class ActorNetwork(nn.Module):
    """
    Actor network
    """
    def __init__(self, hidden_dim=512, z_dim=1024, action_dim=18):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim+z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, h_t, z_t):
        """
        Forward pass
        """
        if z_t.dim() == 3:
            z_t = z_t.flatten(start_dim=1)
        state = torch.concat([h_t, z_t], dim=-1)
        action_logits = self.mlp(state)
        return action_logits
    