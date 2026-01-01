"""
The main encoder for the DreamerV3 model
From the paper, the encoder is defined as a module that:
x_t --> z_t
x_t can be a vector OR an image input [H, W, C]
This is concatenated into a unofirm representation
"""
# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encoder as in the DreamerV3
    """
    def __init__(self, img_channels=1, img_size=84, proprio_dim=0, num_categories=32, num_classes=32):
        super().__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        self.proprio_dim = proprio_dim
        self.num_categories = num_categories
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(self.img_channels, 32, stride=2, kernel_size=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, stride=2, kernel_size=4, padding=1)
        self.conv3 = nn.Conv2d(64, 128, stride=2, kernel_size=4, padding=1)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(12800 + self.proprio_dim, 32*32)
    
    def forward(self, x_img, x_proprio=None):
        """
        Forward pass
        """
        x_out = self.relu(self.conv1(x_img))
        x_out = self.relu(self.conv2(x_out))
        x_out = self.relu(self.conv3(x_out))
        x_out = self.flatten(x_out)
        if x_proprio is not None:
            x_out = torch.concat((x_out, x_proprio), dim=-1)
        logits = self.mlp(x_out)
        z_logits = logits.reshape(-1, 32, 32)
        return z_logits
    
    def sample(self, logits):
        """
        Sample from the trained encoder
        """
        probs = F.softmax(logits, dim=-1)  # (batch, 32, 32)
        # 2. Sample indices from categorical distribution
        # For each of 32 categories, sample 1 class
        dist = torch.distributions.Categorical(probs=probs)  # What goes here?
        indices = dist.sample()
        # 3. Convert indices to one-hot
        one_hot = F.one_hot(indices, num_classes=self.num_classes).float()
        assert one_hot.shape == logits.shape
        return one_hot
        

