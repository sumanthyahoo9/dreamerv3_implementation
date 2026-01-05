"""
Primary training script
"""
import torch
import gymnasium as gym
from src.world_model import WorldModel
from src.actor import ActorNetwork
from src.critic import CriticNetwork
from src.loss_functions import DreamerLosses

def train():
    """
    Training script
    """
    # 1. Setup
    env = gym.make('ALE/Breakout-v5')  # This can also be CartPole
    world_model = WorldModel()
    actor = ActorNetwork()
    critic = CriticNetwork()
    losses = DreamerLosses()
    
    # 2. Optimizers
    world_model_optimizer = torch.optim.Adam(world_model.parameters(), lr=3e-4)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=8e-5)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=8e-5)
    
    # 3. Training loop
    # TODO: Implement!