"""
Test the critic network
"""
import torch
from src.critic import CriticNetwork

def test_critic_network():
    """
    Unit test for the critic network
    """
    critic_network = CriticNetwork()
    h_t = torch.randn((4, 512), requires_grad=True)
    z_t = torch.randn((4, 32, 32), requires_grad=True)
    critic_value = critic_network(h_t, z_t)
    assert critic_value.shape == (4, 1)
    loss = critic_value.sum() # Or calculate them as needed!
    loss.backward()
    assert h_t.grad is not None
    assert z_t.grad is not None


