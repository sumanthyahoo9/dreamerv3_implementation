"""
The three types of loss functions we see in DreamerV3
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DreamerLosses(nn.Module):
    """
    Defining the various loss functions
    """
    def __init__(self):
        super().__init__()
    
    def forward(self):
        """
        Forward pass
        """
        pass
    
    def prediction_loss(self, x_recon_mean, x_recon_std, x_target,
                        r_target, r_pred_mean, r_pred_std,
                        c_pred_logits, c_target):
        """
        L_pred: reconstruction + reward + continue
        L_pred = -ln p(x_t | z, h) - ln p(r_t | z, h) - ln p(c_t | z, h)
        -ln N(x | μ, σ) = 0.5 * ((x - μ)/σ)^2 + ln(σ) + const
        """
        reconstruction = 0.5 * ((x_target - x_recon_mean)/x_recon_std) ** 2 + torch.log(x_recon_std)
        reconstruction = reconstruction.mean()
        reward = 0.5 * ((r_target - r_pred_mean) / r_pred_std) ** 2 + torch.log(r_pred_std)
        reward = reward.mean()
        continue_loss = F.binary_cross_entropy_with_logits(c_pred_logits, c_target)
        return reconstruction + reward + continue_loss
    
    def dynamics_loss(self, z_logits_post, z_logits_prior):
        """
        Measures how good the latent representations are/Measures the difference between two distributions over z(t)
        L_dyn: KL[sg(posterior) || prior] 
        """
        post_probs = F.softmax(z_logits_post.detach(), dim=-1)
        prior_probs = F.softmax(z_logits_prior, dim=-1)

        # KL-divergence
        kl = (post_probs * (torch.log(post_probs) - torch.log(prior_probs))).sum(dim=-1)
        kl = kl.mean()
        kl = torch.clamp(kl, min=1.0)
        return kl
    
    def representation_loss(self, z_logits_post, z_logits_prior):
        """
        L_rep: KL[posterior || sg(prior)]
        """
        post_probs = F.softmax(z_logits_post, dim=-1)
        prior_probs = F.softmax(z_logits_prior.detach(), dim=-1)

        # KL-divergence
        kl = (post_probs * (torch.log(post_probs + 1e-8) - torch.log(prior_probs + 1e-8))).sum(dim=-1)
        kl = kl.mean()
        kl = torch.clamp(kl, min=1.0)
        return kl
