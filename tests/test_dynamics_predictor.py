"""
Unit test for the Dynamics Predictor
"""
import torch
from src.dynamics_predictor import DynamicsPredictor

def test_dynamics_predictor():
    """
    Unit test
    """
    dynamics_predictor = DynamicsPredictor()
    h_t = torch.randn((4, 512), requires_grad=True)
    logits = dynamics_predictor(h_t)
    assert logits.shape == (4, 32, 32)
    loss = logits.sum()
    loss.backward()
    assert h_t.grad is not None