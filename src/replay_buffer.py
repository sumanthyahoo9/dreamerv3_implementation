# src/replay_buffer.py
import random
from collections import deque
import numpy as np
import torch

class ReplayBuffer:
    """
    Stores episodes as sequences for training world model
    """
    def __init__(self, capacity=10000, seq_len=50):
        """
        Args:
            capacity: Max number of episodes to store
            seq_len: Length of subsequences to sample
        """
        self.capacity = capacity
        self.seq_len = seq_len
        self.episodes = deque(maxlen=capacity)  # Circular buffer
    
    def add_episode(self, observations, actions, rewards, dones):
        """
        Add complete episode to buffer
        
        Args:
            observations: (T, obs_dim) - T timesteps
            actions: (T,) - actions taken
            rewards: (T,) - rewards received
            dones: (T,) - episode termination flags
        """
        # Convert to numpy if torch tensors
        if isinstance(observations, torch.Tensor):
            observations = observations.cpu().numpy()
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.cpu().numpy()
        if isinstance(dones, torch.Tensor):
            dones = dones.cpu().numpy()
        
        # Only add if episode is long enough
        if len(observations) >= self.seq_len:
            episode = {
                'obs': observations,
                'actions': actions,
                'rewards': rewards,
                'dones': dones
            }
            self.episodes.append(episode)
    
    def sample_sequences(self, batch_size):
        """
        Sample random subsequences for training
        
        Returns:
            Dict with batched sequences:
            {
                'observations': (batch, seq_len, obs_dim),
                'actions': (batch, seq_len),
                'rewards': (batch, seq_len),
                'dones': (batch, seq_len)
            }
        """
        # Lists to collect samples
        obs_list = []
        actions_list = []
        rewards_list = []
        dones_list = []
        
        for _ in range(batch_size):
            # 1. Pick random episode
            episode_idx = random.randint(0, len(self.episodes) - 1)
            episode = self.episodes[episode_idx]
            
            # 2. Pick random start within episode
            max_start = len(episode['obs']) - self.seq_len
            start_idx = random.randint(0, max_start)
            
            # 3. Extract subsequence
            obs_list.append(episode['obs'][start_idx:start_idx + self.seq_len])
            actions_list.append(episode['actions'][start_idx:start_idx + self.seq_len])
            rewards_list.append(episode['rewards'][start_idx:start_idx + self.seq_len])
            dones_list.append(episode['dones'][start_idx:start_idx + self.seq_len])
        
        # 4. Stack into batch and convert to torch
        return {
            'observations': torch.FloatTensor(np.stack(obs_list)),
            'actions': torch.LongTensor(np.stack(actions_list)),
            'rewards': torch.FloatTensor(np.stack(rewards_list)),
            'dones': torch.FloatTensor(np.stack(dones_list))
        }
    
    def __len__(self):
        return len(self.episodes)