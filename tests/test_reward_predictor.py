"""
Unit test for the Reward predictor
"""
import torch
from src.reward_predictor import RewardPredictor

def test_reward_predictor():
    """
    The unit test
    """
    reward_predictor_model = RewardPredictor()
    h_t = torch.randn((4, 512), requires_grad=True)
    z_t = torch.randn((4, 32, 32), requires_grad=True)
    mean, std = reward_predictor_model(h_t, z_t)
    assert mean.shape == (4,)
    assert std.shape == (4,)
    loss = mean.sum() + std.sum()
    loss.backward()
    assert h_t.grad is not None
    assert z_t.grad is not None