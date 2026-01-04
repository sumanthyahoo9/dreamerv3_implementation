"""
Unit test for the Continue predictor
"""
import torch
from src.continue_predictor import ContinuePredictor

def test_reward_predictor():
    """
    The unit test
    """
    continue_predictor_model = ContinuePredictor()
    h_t = torch.randn((4, 512), requires_grad=True)
    z_t = torch.randn((4, 32, 32), requires_grad=True)
    logits = continue_predictor_model(h_t, z_t)
    assert logits.shape == (4,)
    loss = logits.sum()
    loss.backward()
    assert h_t.grad is not None
    assert z_t.grad is not None