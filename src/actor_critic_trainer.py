"""
Actor-Critic Trainer
"""
import torch
import torch.nn.functional as F

def compute_returns(rewards, values, dones, gamma=0.99, lambda_=0.95):
    """
    Compute λ-returns for value targets
    """
    horizon, _ = rewards.shape
    returns = torch.zeros_like(rewards)
    # Work backwards
    for t in reversed(range(horizon)):
        if t == horizon-1:
            returns[t] = rewards[t]
        else:
            # lambda-return formula
            bootstrap = (1 - lambda_) * values[t+1] + lambda_ * returns[t+1]
            returns[t] = rewards[t] + gamma * (1 - dones[t]) * bootstrap
    return returns

def compute_actor_loss(actor, trajectory):
    """
    Compute the actor loss
    """
    # 1. Get actions, z, h from trajectory
    actions, z, h = trajectory.get("actions"), trajectory.get("z"), trajectory.get("h")
    rewards, values, dones = trajectory.get("rewards"), trajectory.get("values"), trajectory.get("dones")
    # 2. Compute returns (call compute_returns)
    returns = compute_returns(rewards, values, dones)
    # 3. Compute advantages = returns - values
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalize
    # 4. Get action log probs from actor
    horizon, _ = actions.shape
    log_probs = []
    for t in range(horizon):
        # Forward pass through the actor
        action_logits = actor(h[t], z[t])
        # Get the log prob of taken action
        dist = torch.distributions.Categorical(logits=action_logits)
        log_prob = dist.log_prob(actions[t]) # (batch, )
        log_probs.append(log_prob)
    log_probs = torch.stack(log_probs)
    # 5. Loss = -(log_probs * advantages).mean()
    loss = -(log_probs * advantages.detach()).mean()  # Detach advantages!
    return loss

def compute_critic_loss(critic, trajectory):
    """
    Compute value regression loss
    Returns the scalar loss
    """
    z = trajectory['z']       # (horizon, batch, 32, 32)
    h = trajectory['h']       # (horizon, batch, 512)
    rewards = trajectory['rewards']
    values = trajectory['values']
    dones = trajectory['dones']
    horizon, _, _ = h.shape

    # Compute the target returns
    returns = compute_returns(rewards, values, dones)
    # Get the critic predictions
    predicted_values = []
    for t in range(horizon):
        value = critic(h[t], z[t])
        predicted_values.append(value)
    predicted_values = torch.stack(predicted_values)
    # MSE loss
    loss = F.mse_loss(predicted_values, returns)
    return loss



