"""
The decoder module for the DreamerV3
Reconstruct image and x_t from [z_t, h_t]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    """
    Decoder module
    """
    def __init__(self, hidden_dim=512, n_channels=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv_tr1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv_tr2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.mlp = nn.Linear(self.hidden_dim + 32*32, 128*10*10)
        self.mean_head = nn.ConvTranspose2d(32, n_channels, kernel_size=6, stride=2, padding=0)
        self.std_head = nn.ConvTranspose2d(32, n_channels, kernel_size=6, stride=2, padding=0)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
    
    def forward(self, z_t, h_t):
        """
        Forward pass
        """
        batch,_ = h_t.shape
        z_t = self.flatten(z_t) # Shape is [batch, 1024]
        out = torch.concat([z_t, h_t], dim=-1) # Shape is [batch, 1024 + 512]
        out = self.mlp(out)
        out = out.reshape(batch, 128, 10, 10)
        out = self.relu(self.conv_tr1(out))
        out = self.relu(self.conv_tr2(out))
        # print(f"The shape of the reconstructed state BEFORE the mean calculation is {out.shape}")
        mean = self.mean_head(out)
        std = self.std_head(out)
        mean = F.sigmoid(mean)
        std = F.softplus(std) + 1e-5
        return mean, std





