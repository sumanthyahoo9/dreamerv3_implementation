"""
The Recurrent Neural Network model
The paper implements a Gated Recurrent Unit network
"""
import torch
import torch.nn as nn

class RecurrentModel(nn.Module):
    """
    Recurrent Neural Net, GRU
    """
    def __init__(self, z_dim=1024, action_dim=18, hidden_dim=512, mlp_hidden=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(z_dim + action_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
        )
        # GRU cell
        self.gru = nn.GRUCell(mlp_hidden, hidden_dim)
    
    def forward(self, z_prev, a_prev, h_prev):
        """
        Forward pass
        The GRU takes in the actions, h_t and representations as the inputs
        """
        if z_prev.dim() == 3:
            z_prev = z_prev.flatten(start_dim=1)
        inputs = torch.concat([a_prev, z_prev], dim=-1)
        mlp_out = self.mlp(inputs)
        h_next = self.gru(mlp_out, h_prev)
        return h_next
    
    