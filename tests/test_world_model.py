"""
Unit-tests for the world model
"""
import torch
from src.world_model import WorldModel

def test_world_model_imagine():
    """
    Test the world model's inference
    """
    a_t = torch.randn((4, 18), requires_grad=True)
    h_prev = torch.randn((4, 512), requires_grad=True)
    world_model = WorldModel()
    z_t, h_t = world_model.imagine(a_t, h_prev)
    assert z_t.shape == (4, 32, 32)
    assert h_t.shape == (4, 512)
    loss = z_t.sum() + h_t.sum()
    loss.backward()
    # Assert that the grads are None
    assert a_t.grad is not None
    assert h_prev.grad is not None

def test_world_model_observe():
    """
    Test the world model's training
    """
    world_model = WorldModel()
    x_t = torch.randint(0, 255, (4, 1, 84, 84), requires_grad=True, dtype=torch.float32)
    a_t = torch.randn((4, 18), requires_grad=True)
    h_prev = torch.randn((4, 512), requires_grad=True)
    z_t, h_t, z_logits_post, z_logits_prior, x_recon_mean, x_recon_std = world_model.observe(x_t, a_t, h_prev)
    loss = z_t.sum() + h_t.sum() + z_logits_post.sum() + z_logits_prior.sum() + x_recon_mean.sum() + x_recon_std.sum()
    loss.backward()
    assert a_t.grad is not None
    assert x_t.grad is not None
    assert h_prev.grad is not None
    



