# tests/test_replay_buffer.py
import torch
from src.replay_buffer import ReplayBuffer

def test_replay_buffer_add_episode():
    """Test adding episodes to buffer"""
    buffer = ReplayBuffer(capacity=100, seq_len=10)
    
    # Create fake episode (20 steps)
    obs = torch.randn(20, 4)  # CartPole obs dim
    actions = torch.randint(0, 2, (20,))
    rewards = torch.randn(20)
    dones = torch.zeros(20)
    dones[-1] = 1  # Episode ends at last step
    
    buffer.add_episode(obs, actions, rewards, dones)
    
    assert len(buffer) == 1
    
def test_replay_buffer_sample_sequences():
    """Test sampling subsequences"""
    buffer = ReplayBuffer(capacity=100, seq_len=10)
    
    # Add multiple episodes
    for _ in range(5):
        obs = torch.randn(50, 4)
        actions = torch.randint(0, 2, (50,))
        rewards = torch.randn(50)
        dones = torch.zeros(50)
        buffer.add_episode(obs, actions, rewards, dones)
    
    # Sample batch
    batch = buffer.sample_sequences(batch_size=4)
    
    # Check shapes
    assert batch['observations'].shape == (4, 10, 4)  # (batch, seq_len, obs_dim)
    assert batch['actions'].shape == (4, 10)
    assert batch['rewards'].shape == (4, 10)
    assert batch['dones'].shape == (4, 10)
    
def test_replay_buffer_skip_short_episodes():
    """Test that short episodes are skipped"""
    buffer = ReplayBuffer(capacity=100, seq_len=50)
    
    # Add short episode (should be skipped)
    obs_short = torch.randn(30, 4)
    actions_short = torch.randint(0, 2, (30,))
    rewards_short = torch.randn(30)
    dones_short = torch.zeros(30)
    buffer.add_episode(obs_short, actions_short, rewards_short, dones_short)
    
    assert len(buffer) == 0  # Should not add short episode
    
    # Add long episode (should be added)
    obs_long = torch.randn(100, 4)
    actions_long = torch.randint(0, 2, (100,))
    rewards_long = torch.randn(100)
    dones_long = torch.zeros(100)
    buffer.add_episode(obs_long, actions_long, rewards_long, dones_long)
    
    assert len(buffer) == 1  # Should add long episode