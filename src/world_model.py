"""
The World model here combines the 4 modules so far
"""
import torch.nn as nn
from src.encoder import Encoder
from src.decoder import Decoder
from src.rnn_model import RecurrentModel
from src.dynamics_predictor import DynamicsPredictor

class WorldModel(nn.Module):
    """
    The world model wrapper
    """
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.dynamics_predictor = DynamicsPredictor()
        self.rnn = RecurrentModel()
        self.sample = self.encoder.sample
    
    def observe(self, x_t, a_t, h_prev):
        """
        Training mode, with real observations
        """
        # Encoder path
        z_logits_post = self.encoder(x_t)
        z_t = self.sample(z_logits_post)
        # Dynamics path, prior
        z_logits_prior = self.dynamics_predictor(h_prev)
        # Update the RNN
        h_t = self.rnn(z_t, a_t, h_prev)
        # Reconstruct
        x_recon_mean, x_recon_std = self.decoder(z_t, h_t)
        return z_t, h_t, z_logits_post, z_logits_prior, x_recon_mean, x_recon_std
    
    def imagine(self, a_t, h_prev):
        """
        Inference, ONLY dynamics!, no encoder
        """
        z_logits = self.dynamics_predictor(h_prev)
        z_t = self.sample(z_logits)
        h_t = self.rnn(z_t, a_t, h_prev)
        return z_t, h_t
    
    def forward(self):
        """
        Forward pass
        """
        pass