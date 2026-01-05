"""
The World model here combines the 4 modules so far
"""
import torch.nn as nn
from src.encoder import Encoder
from src.decoder import Decoder
from src.rnn_model import RecurrentModel
from src.reward_predictor import RewardPredictor
from src.continue_predictor import ContinuePredictor
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
        self.reward_predictor = RewardPredictor()      # ← Add
        self.continue_predictor = ContinuePredictor()  # ← Add
        self.sample = self.encoder.sample
    
    def observe(self, x_t, a_t, h_prev):
        """
        Training mode, with real observations
        Returns all necessary outputs for loss computation
        """
        # Encoder path (posterior)
        z_logits_post = self.encoder(x_t)
        z_t = self.sample(z_logits_post)
        
        # Dynamics path (prior)
        z_logits_prior = self.dynamics_predictor(h_prev)
        
        # Update RNN
        h_t = self.rnn(z_t, a_t, h_prev)
        
        # Reconstruct observation
        x_recon_mean, x_recon_std = self.decoder(z_t, h_t)
        
        # Predict reward
        r_pred_mean, r_pred_std = self.reward_predictor(h_t, z_t)
        
        # Predict continue
        c_pred_logits = self.continue_predictor(h_t, z_t)
        
        return {
            'z_t': z_t,
            'h_t': h_t,
            'z_logits_post': z_logits_post,
            'z_logits_prior': z_logits_prior,
            'x_recon_mean': x_recon_mean,
            'x_recon_std': x_recon_std,
            'r_pred_mean': r_pred_mean,
            'r_pred_std': r_pred_std,
            'c_pred_logits': c_pred_logits
        }
    
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