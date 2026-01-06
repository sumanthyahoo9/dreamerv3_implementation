"""
The unit tests for the imagination module needs to be written
"""
import torch
from src.imagination import imagine_rollout
from src.world_model import WorldModel
from src.actor import ActorNetwork
from src.critic import CriticNetwork

def test_imagine_rollout():
    """
    Unit test for the imagination
    """
    world_model = WorldModel()
    actor = ActorNetwork()
    critic = CriticNetwork()
    start_z = torch.randn((4, 32, 32), requires_grad=True)
    start_h = torch.randn((4, 512), requires_grad=True)
    horizon=10
    trajectory = imagine_rollout(
        world_model,
        actor,
        critic,
        start_z,
        start_h,
        horizon
    )
    # Check all keys exist
    assert 'z' in trajectory
    assert 'h' in trajectory
    assert 'actions' in trajectory
    assert 'rewards' in trajectory
    assert 'values' in trajectory
    assert 'dones' in trajectory
    
    # Check shapes (horizon, batch, ...)
    assert trajectory['z'].shape == (horizon, 4, 32, 32)
    assert trajectory['h'].shape == (horizon, 4, 512)
    assert trajectory['actions'].shape == (horizon, 4)
    assert trajectory['rewards'].shape == (horizon, 4)
    assert trajectory['values'].shape == (horizon, 4,1)
    assert trajectory['dones'].shape == (horizon, 4)
    
    # Check values are reasonable
    assert torch.all(trajectory['dones'] >= 0)  # Probabilities
    assert torch.all(trajectory['dones'] <= 1)


