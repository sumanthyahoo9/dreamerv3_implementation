"""
Infernece/Imagination
"""
from typing import Dict
import torch
import torch.nn.functional as F

def imagine_rollout(world_model, actor, critic, start_z, start_h, horizon=15) -> Dict[str, torch.Tensor]:
    """
    Rollout trajectory in world model, pure imagination
    """
    trajectory = {
        'z': [],
        'h': [],
        'actions': [],
        'rewards': [],
        'values': [],
        'dones': []
    }
    z_t, h_t = start_z, start_h
    # Disable gradient tracking during inference
    with torch.no_grad():
        for t in range(horizon):
            # 1. Actor chooses action from (z_t, h_t)
            action_logits = actor(h_t, z_t)
            action_dist = torch.distributions.Categorical(logits=action_logits)
            action = action_dist.sample()  # (batch,)
            
            # 2. Predict next z using dynamics (imagination!)
            z_next_logits = world_model.dynamics_predictor(h_t)
            z_next = world_model.sample(z_next_logits)  # (batch, 32, 32)
            
            # 3. Predict reward
            reward_mean, _ = world_model.reward_predictor(h_t, z_t)
            # Use mean as predicted reward (or sample from distribution)
            reward = reward_mean  # (batch,)
            
            # 4. Predict done (continue probability)
            done_logits = world_model.continue_predictor(h_t, z_t)
            done_prob = torch.sigmoid(done_logits)  # P(continue)
            done = 1.0 - done_prob  # P(done) = 1 - P(continue)
            
            # 5. Get value estimate from critic
            value = critic(h_t, z_t)  # (batch,)
            
            # 6. Update h using RNN
            # Convert action to one-hot for RNN
            action_onehot = F.one_hot(action, num_classes=actor.action_dim).float()
            h_next = world_model.rnn(z_t, action_onehot, h_t)
            
            # 7. Store in trajectory
            trajectory['z'].append(z_t)
            trajectory['h'].append(h_t)
            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)
            trajectory['values'].append(value)
            trajectory['dones'].append(done)
            
            # Move to next timestep
            z_t = z_next
            h_t = h_next
    # Stack into tensors
    trajectory = {
        'z': torch.stack(trajectory['z']),           # (horizon, batch, 32, 32)
        'h': torch.stack(trajectory['h']),           # (horizon, batch, 512)
        'actions': torch.stack(trajectory['actions']), # (horizon, batch)
        'rewards': torch.stack(trajectory['rewards']), # (horizon, batch)
        'values': torch.stack(trajectory['values']),   # (horizon, batch)
        'dones': torch.stack(trajectory['dones'])      # (horizon, batch)
        }
    return trajectory