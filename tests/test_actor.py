"""
Unit test for the actor network
"""
import torch
from src.actor import ActorNetwork

def test():
    """
    Unit test for the Actor Network
    """
    actor = ActorNetwork()
    z_t = torch.randn((4, 32, 32), requires_grad=True)
    h_t = torch.randn((4, 512), requires_grad=True)
    action_logits = actor(h_t, z_t)
    assert action_logits.shape == (4, 18)
    loss = action_logits.sum()
    loss.backward()
    assert z_t.grad is not None
    assert h_t.grad is not None
    