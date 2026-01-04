"""
Continue predictor
Whether the episode is yet to continue or terminated
"""
import torch
import torch.nn as nn

class ContinuePredictor(nn.Module):
    """Predicts episode continuation"""
    def __init__(self, hidden_dim=512, z_dim=1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + z_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)  # Binary: continue or not
        )
    
    def forward(self, h_t, z_t):
        """
        Forward pass
        """
        if z_t.dim() == 3:
            z_t = z_t.flatten(start_dim=1)
        state = torch.cat([h_t, z_t], dim=-1)
        logits = self.mlp(state).squeeze(-1)  # (batch,)
        return logits
    