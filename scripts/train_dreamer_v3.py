"""
DreamerV3 Training Script
Complete training loop for CartPole environment
"""
import torch
import torch.nn.functional as F
import gymnasium as gym
from src.world_model import WorldModel
from src.actor import ActorNetwork
from src.critic import CriticNetwork
from src.loss_functions import DreamerLosses
from src.replay_buffer import ReplayBuffer
from src.imagination import imagine_rollout
from src.actor_critic_trainer import (
    compute_actor_loss,
    compute_critic_loss,
)


def train():
    """Main training loop for DreamerV3"""
    
    # ============ HYPERPARAMETERS ============
    num_episodes = 1000
    max_steps_per_episode = 500
    min_buffer_size = 50          # Start training after this many episodes
    batch_size = 16
    seq_len = 50
    world_model_updates = 10      # WM updates per episode
    actor_critic_updates = 5      # AC updates per episode
    imagination_horizon = 15
    
    # Learning rates
    wm_lr = 3e-4
    actor_lr = 8e-5
    critic_lr = 8e-5
    
    # ============ ENVIRONMENT SETUP ============
    env = gym.make('CartPole-v1')
    action_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    
    print("Environment: CartPole-v1")
    print(f"Action dim: {action_dim}, Obs dim: {obs_dim}")
    
    # ============ MODEL INITIALIZATION ============
    replay_buffer = ReplayBuffer(capacity=10000, seq_len=seq_len)
    
    world_model = WorldModel()
    actor = ActorNetwork(action_dim=action_dim)
    critic = CriticNetwork()
    losses = DreamerLosses()
    
    # ============ OPTIMIZERS ============
    wm_optimizer = torch.optim.Adam(world_model.parameters(), lr=wm_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    
    print("\nStarting training...\n")
    
    # ============ MAIN TRAINING LOOP ============
    for episode in range(num_episodes):
        
        # ========================================
        # PHASE 1: COLLECT EXPERIENCE
        # ========================================
        obs, _ = env.reset()
        obs = torch.FloatTensor(obs).unsqueeze(0)  # (1, obs_dim)
        h_t = torch.zeros(1, 512)  # Initial hidden state
        
        episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'dones': []
        }
        
        episode_reward = 0
        
        for step in range(max_steps_per_episode):
            # Get action from actor
            with torch.no_grad():
                # Encode observation (CartPole: use as proprio)
                z_logits = world_model.encoder(None, obs)
                z_t = world_model.sample(z_logits)
                
                # Actor samples action
                action_logits = actor(h_t, z_t)
                action_dist = torch.distributions.Categorical(logits=action_logits)
                action = action_dist.sample()
            
            # Step environment
            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            # Store transition
            episode_data['observations'].append(obs.squeeze(0))
            episode_data['actions'].append(action)
            episode_data['rewards'].append(torch.tensor([reward]))
            episode_data['dones'].append(torch.tensor([float(done)]))
            
            # Update hidden state
            with torch.no_grad():
                action_onehot = F.one_hot(action, num_classes=action_dim).float()
                h_t = world_model.rnn(z_t, action_onehot, h_t)
            
            episode_reward += reward
            obs = torch.FloatTensor(next_obs).unsqueeze(0)
            
            if done:
                break
        
        # Add episode to replay buffer
        if len(episode_data['observations']) >= seq_len:
            replay_buffer.add_episode(
                torch.stack(episode_data['observations']),
                torch.stack(episode_data['actions']),
                torch.stack(episode_data['rewards']),
                torch.stack(episode_data['dones'])
            )
        
        print(f"Episode {episode:4d} | Reward: {episode_reward:6.1f} | Steps: {step+1:3d} | Buffer: {len(replay_buffer):4d}")
        
        # ========================================
        # PHASE 2: TRAIN WORLD MODEL
        # ========================================
        if len(replay_buffer) >= min_buffer_size:
            wm_losses = []
            
            for _ in range(world_model_updates):
                # Sample batch of sequences
                batch = replay_buffer.sample_sequences(batch_size)
                
                # Use first timestep (simplified - full version would use sequences)
                obs_batch = batch['observations'][:, 0, :]   # (batch, obs_dim)
                actions_batch = batch['actions'][:, 0]       # (batch,)
                rewards_batch = batch['rewards'][:, 0]       # (batch,)
                dones_batch = batch['dones'][:, 0]           # (batch,)
                
                h_prev = torch.zeros(batch_size, 512)
                
                # Forward through world model
                outputs = world_model.observe(
                    obs_batch,      # Observations
                    actions_batch,  # Actions
                    h_prev          # Previous hidden state
                )
                
                # Compute world model losses
                l_pred = losses.prediction_loss(
                    obs_batch,                      # x_target
                    outputs['x_recon_mean'],        # x_recon_mean
                    outputs['x_recon_std'],         # x_recon_std
                    rewards_batch,                  # r_target
                    outputs['r_pred_mean'],         # r_pred_mean
                    outputs['r_pred_std'],          # r_pred_std
                    dones_batch,                    # c_target
                    outputs['c_pred_logits']        # c_pred_logits
                )
                
                l_dyn = losses.dynamics_loss(
                    outputs['z_logits_post'],       # Encoder (posterior)
                    outputs['z_logits_prior']       # Dynamics (prior)
                )
                
                l_rep = losses.representation_loss(
                    outputs['z_logits_post'],       # Encoder
                    outputs['z_logits_prior']       # Dynamics
                )
                
                # Total loss
                wm_loss = l_pred + l_dyn + l_rep
                wm_losses.append(wm_loss.item())
                
                # Update world model
                wm_optimizer.zero_grad()
                wm_loss.backward()
                torch.nn.utils.clip_grad_norm_(world_model.parameters(), max_norm=100.0)
                wm_optimizer.step()
            
            # Log world model training
            if episode % 10 == 0:
                avg_wm_loss = sum(wm_losses) / len(wm_losses)
                print(f"  → WM Loss: {avg_wm_loss:.4f}")
        
        # ========================================
        # PHASE 3: TRAIN ACTOR-CRITIC (IMAGINATION)
        # ========================================
        if len(replay_buffer) >= min_buffer_size:
            actor_losses = []
            critic_losses = []
            
            for _ in range(actor_critic_updates):
                # Sample starting states
                batch = replay_buffer.sample_sequences(batch_size)
                obs_start = batch['observations'][:, 0, :]
                h_start = torch.zeros(batch_size, 512)
                
                # Encode starting states
                with torch.no_grad():
                    z_logits_start = world_model.encoder(None, obs_start)
                    z_start = world_model.sample(z_logits_start)
                
                # Imagine trajectory in world model
                trajectory = imagine_rollout(
                    world_model,
                    actor,
                    critic,
                    z_start,
                    h_start,
                    horizon=imagination_horizon
                )
                
                # Train actor (policy)
                actor_loss = compute_actor_loss(actor, trajectory)
                actor_losses.append(actor_loss.item())
                
                actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=100.0)
                actor_optimizer.step()
                
                # Train critic (value function)
                critic_loss = compute_critic_loss(critic, trajectory)
                critic_losses.append(critic_loss.item())
                
                critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=100.0)
                critic_optimizer.step()
            
            # Log actor-critic training
            if episode % 10 == 0:
                avg_actor_loss = sum(actor_losses) / len(actor_losses)
                avg_critic_loss = sum(critic_losses) / len(critic_losses)
                print(f"  → Actor Loss: {avg_actor_loss:.4f} | Critic Loss: {avg_critic_loss:.4f}")
        
        # ========================================
        # CHECKPOINT SAVING
        # ========================================
        if episode % 100 == 0 and episode > 0:
            checkpoint_path = f'checkpoints/checkpoint_episode_{episode}.pt'
            torch.save({
                'episode': episode,
                'world_model': world_model.state_dict(),
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'wm_optimizer': wm_optimizer.state_dict(),
                'actor_optimizer': actor_optimizer.state_dict(),
                'critic_optimizer': critic_optimizer.state_dict(),
            }, checkpoint_path)
            print(f"\n✓ Checkpoint saved: {checkpoint_path}\n")
    
    # ========================================
    # TRAINING COMPLETE
    # ========================================
    env.close()
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    # Save final model
    torch.save({
        'world_model': world_model.state_dict(),
        'actor': actor.state_dict(),
        'critic': critic.state_dict(),
    }, 'final_model.pt')
    print("✓ Final model saved: final_model.pt")


if __name__ == '__main__':
    # Create checkpoints directory
    import os
    os.makedirs('checkpoints', exist_ok=True)
    
    # Run training
    train()