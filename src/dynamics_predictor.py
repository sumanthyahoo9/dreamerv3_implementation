"""
h_t → MLP → logits for p(z_t | h_t)
"""
import torch.nn as nn

class DynamicsPredictor(nn.Module):
    """
    The dynamics predictor module
    """
    def __init__(self, hidden_dim=512, num_categories=32, num_classes=32):
        super().__init__()
        self.num_categories = num_categories
        self.num_classes = num_classes
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_categories*num_classes)
        )
    
    def forward(self, h_t):
        """
        Forward pass
        """
        logits = self.mlp(h_t) # Shape is [batch_size, 1024]
        return logits.reshape(-1, self.num_categories, self.num_classes)
    