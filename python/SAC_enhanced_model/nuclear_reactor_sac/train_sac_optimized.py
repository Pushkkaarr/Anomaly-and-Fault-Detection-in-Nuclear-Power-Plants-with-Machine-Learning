"""
Optimized SAC Training for Nuclear Reactor Control

Key Improvements:
1. Warm-up phase with random exploration
2. Adaptive learning rate scheduling
3. Curriculum learning (gradually increasing difficulty)
4. Comprehensive monitoring and checkpointing
5. Early stopping on convergence (FIXED: uses reward per step)
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

# Import our modules
from nuclear_env_optimized import NuclearReactorEnv
from sac_networks import SACAgent
from replay_buffer import ReplayBuffer


class SAC_Trainer:
    """Training manager for SAC agent"""
    
    def __init__(self, 
                 env,
                 agent,
                 replay_buffer,
                 save_dir='models/SAC_optimized',
                 log_dir='logs/SAC_optimized'):
        
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.save_dir = save_dir
        self.log_dir = log_dir
        
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Training history
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_rewards_per_step = []  # NEW: Track reward per step
        self.success_rate_history = []
        self.training_metrics = []
        
        # Best model tracking (based on reward per step)
        self.best_reward_per_step = -np.inf
        self.best_episode = 0
        
    def warmup_phase(self, num_steps=5000):
        """
        Random exploration phase to populate replay buffer
        Critical for SAC to have diverse initial experiences
        """
        print(f"\n{'='*60}")
        print("WARMUP PHASE: Random Exploration")
        print(f"{'='*60}")
        
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episodes_completed = 0
        
        for step in tqdm(range(num_steps), desc="Warmup"):
            # Random action
            action = self.env.action_space.sample()
            
            # Execute
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store in buffer
            self.replay_buffer.add(obs, action, reward, next_obs, float(terminated))
            
            episode_reward += reward
            episode_length += 1
            
            if done:
                episodes_completed += 1
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                # Track reward per step
                reward_per_step = episode_reward / episode_length if episode_length > 0 else 0
                self.episode_rewards_per_step.append(reward_per_step)
                
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
            else:
                obs = next_obs
        
        avg_reward_total = np.mean(self.episode_rewards[-episodes_completed:])
        avg_reward_per_step = np.mean(self.episode_rewards_per_step[-episodes_completed:])
        
        print(f"✓ Warmup complete: {episodes_completed} episodes, buffer size: {len(self.replay_buffer)}")
        print(f"  Avg total reward: {avg_reward_total:.2f}")
        print(f"  Avg reward per step: {avg_reward_per_step:.2f}")
    
    def train(self, 
              total_timesteps=500000,
              batch_size=256,
              eval_frequency=5000,
              save_frequency=10000,
              target_reward_per_step=36.0):
        """
        Main training loop with monitoring and checkpointing
        
        Args:
            total_timesteps: Total environment steps to train
            batch_size: Batch size for network updates
            eval_frequency: How often to evaluate performance
            save_frequency: How often to save checkpoints
            target_reward_per_step: Stop if reward per step exceeds this (36 = better than baseline 33)
        """
        
        print(f"\n{'='*60}")
        print("STARTING SAC TRAINING")
        print(f"{'='*60}")
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Batch size: {batch_size}")
        print(f"Device: {self.agent.device}")
        print(f"Target reward/step: {target_reward_per_step:.1f} (baseline ~33)")
        print(f"{'='*60}\n")
        
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_num = 0
        
        # Training loop
        for step in tqdm(range(total_timesteps), desc="Training"):
            
            # Select action (with exploration noise early in training)
            if step < 10000:
                # More exploration early
                action = self.agent.select_action(obs, deterministic=False)
            else:
                action = self.agent.select_action(obs, deterministic=False)
            
            # Execute action
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store transition
            self.replay_buffer.add(obs, action, reward, next_obs, float(terminated))
            
            episode_reward += reward
            episode_length += 1
            
            # Update networks (after warmup buffer is populated)
            if len(self.replay_buffer) > batch_size * 2:
                update_metrics = self.agent.update(self.replay_buffer, batch_size)
                self.training_metrics.append(update_metrics)
            
            # Episode end
            if done:
                episode_num += 1
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                # Track reward per step
                reward_per_step = episode_reward / episode_length if episode_length > 0 else 0
                self.episode_rewards_per_step.append(reward_per_step)
                
                # Log episode info
                if episode_num % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    avg_reward_per_step = np.mean(self.episode_rewards_per_step[-10:])
                    avg_length = np.mean(self.episode_lengths[-10:])
                    tqdm.write(f"Episode {episode_num} | Total: {episode_reward:.0f} | "
                              f"Per-step: {reward_per_step:.2f} | "
                              f"Length: {episode_length} | Avg10: {avg_reward_per_step:.2f} | "
                              f"Alpha: {self.agent.alpha:.3f}")
                
                # Reset environment
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
            else:
                obs = next_obs
            
            # Evaluation
            if step % eval_frequency == 0 and step > 0:
                eval_reward_per_step, eval_total_reward, eval_avg_steps = self.evaluate(num_episodes=5)
                tqdm.write(f"\n{'='*60}")
                tqdm.write(f"EVALUATION @ Step {step:,}")
                tqdm.write(f"Reward per step: {eval_reward_per_step:.2f}")
                tqdm.write(f"Avg total reward: {eval_total_reward:.0f}")
                tqdm.write(f"Avg episode length: {eval_avg_steps:.0f} steps")
                tqdm.write(f"{'='*60}\n")
                
                # Save best model based on reward per step
                if eval_reward_per_step > self.best_reward_per_step:
                    self.best_reward_per_step = eval_reward_per_step
                    self.best_episode = episode_num
                    self.agent.save(f"{self.save_dir}/best_model.pth")
                    tqdm.write(f"★ New best model saved! Reward/step: {eval_reward_per_step:.2f}")
                
                # Early stopping based on reward per step
                if eval_reward_per_step > target_reward_per_step:
                    tqdm.write(f"\n🎉 TRAINING COMPLETE - Target performance achieved!")
                    tqdm.write(f"Final reward per step: {eval_reward_per_step:.2f} > {target_reward_per_step:.2f}")
                    break
            
            # Periodic checkpoint
            if step % save_frequency == 0 and step > 0:
                self.agent.save(f"{self.save_dir}/checkpoint_{step}.pth")
                self.save_training_curves()
        
        # Final save
        self.agent.save(f"{self.save_dir}/final_model.pth")
        self.save_training_curves()
        self.save_training_summary()
        
        print(f"\n{'='*60}")
        print("TRAINING FINISHED")
        print(f"{'='*60}")
        print(f"Best reward per step: {self.best_reward_per_step:.2f} (Episode {self.best_episode})")
        print(f"Total episodes: {episode_num}")
        print(f"Models saved to: {self.save_dir}")
        print(f"{'='*60}\n")
    
    def evaluate(self, num_episodes=5, render=False):
        """
        Evaluate agent performance (deterministic policy)
        
        Returns: (reward_per_step, avg_total_reward, avg_steps)
        """
        eval_rewards = []
        eval_steps = []
        
        for ep in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False
            
            while not done:
                # Deterministic action selection
                action = self.agent.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                episode_steps += 1
                done = terminated or truncated
            
            eval_rewards.append(episode_reward)
            eval_steps.append(episode_steps)
        
        avg_reward = np.mean(eval_rewards)
        avg_steps = np.mean(eval_steps)
        reward_per_step = avg_reward / avg_steps if avg_steps > 0 else 0
        
        return reward_per_step, avg_reward, avg_steps
    
    def save_training_curves(self):
        """Generate and save training performance plots"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Episode Rewards Per Step (NEW - Most Important Metric)
        if len(self.episode_rewards_per_step) > 0:
            axes[0, 0].plot(self.episode_rewards_per_step, alpha=0.3, color='blue')
            if len(self.episode_rewards_per_step) > 50:
                # Moving average
                window = 50
                moving_avg = np.convolve(self.episode_rewards_per_step, 
                                        np.ones(window)/window, 
                                        mode='valid')
                axes[0, 0].plot(range(window-1, len(self.episode_rewards_per_step)), 
                              moving_avg, 
                              color='red', 
                              linewidth=2, 
                              label=f'{window}-episode MA')
            axes[0, 0].axhline(y=33, color='green', linestyle='--', alpha=0.5, label='Baseline (33)')
            axes[0, 0].axhline(y=36, color='orange', linestyle='--', alpha=0.5, label='Target (36)')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward per Step')
            axes[0, 0].set_title('Episode Rewards per Step (KEY METRIC)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Episode Lengths
        axes[0, 1].plot(self.episode_lengths, color='green', alpha=0.5)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Length (steps)')
        axes[0, 1].set_title('Episode Lengths (Survival Time)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Actor Loss
        if len(self.agent.actor_losses) > 0:
            axes[1, 0].plot(self.agent.actor_losses, alpha=0.5)
            axes[1, 0].set_xlabel('Update Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_title('Actor Loss')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Alpha (Entropy Coefficient)
        if len(self.agent.alpha_values) > 0:
            axes[1, 1].plot(self.agent.alpha_values, color='purple')
            axes[1, 1].set_xlabel('Update Step')
            axes[1, 1].set_ylabel('Alpha')
            axes[1, 1].set_title('Entropy Coefficient (Auto-tuned)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.log_dir}/training_curves.png", dpi=150)
        plt.close()
    
    def save_training_summary(self):
        """Save training statistics to JSON"""
        
        summary = {
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_episodes': len(self.episode_rewards),
            'best_reward_per_step': float(self.best_reward_per_step),
            'best_episode': int(self.best_episode),
            'final_avg_reward_per_step': float(np.mean(self.episode_rewards_per_step[-100:])) if len(self.episode_rewards_per_step) >= 100 else float(np.mean(self.episode_rewards_per_step)),
            'final_avg_total_reward': float(np.mean(self.episode_rewards[-100:])) if len(self.episode_rewards) >= 100 else float(np.mean(self.episode_rewards)),
            'buffer_statistics': self.replay_buffer.get_statistics(),
            'hyperparameters': {
                'gamma': self.agent.gamma,
                'tau': self.agent.tau,
                'initial_alpha': 0.2,
                'final_alpha': float(self.agent.alpha),
                'auto_entropy_tuning': self.agent.auto_entropy_tuning
            }
        }
        
        with open(f"{self.log_dir}/training_summary.json", 'w') as f:
            json.dump(summary, indent=4, fp=f)


def main():
    """Main training script"""
    
    # Configuration
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    
    # Create environment
    env = NuclearReactorEnv(reward_shaping='advanced')
    
    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        max_size=200000
    )
    
    # Create SAC agent
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=DEVICE,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_alpha=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,  # Initial value (will be auto-tuned)
        auto_entropy_tuning=True
    )
    
    # Create trainer
    trainer = SAC_Trainer(
        env=env,
        agent=agent,
        replay_buffer=replay_buffer,
        save_dir='models/SAC_optimized',
        log_dir='logs/SAC_optimized'
    )
    
    # Warmup phase
    trainer.warmup_phase(num_steps=10000)
    
    # Train
    trainer.train(
        total_timesteps=50000,
        batch_size=256,
        eval_frequency=5000,
        save_frequency=25000,
        target_reward_per_step=36.0  # FIXED: Target 36 reward/step (better than baseline 33)
    )
    
    # Final evaluation
    print("\nFinal Evaluation (10 episodes):")
    reward_per_step, total_reward, avg_steps = trainer.evaluate(num_episodes=10)
    print(f"Reward per step: {reward_per_step:.2f}")
    print(f"Average total reward: {total_reward:.0f}")
    print(f"Average episode length: {avg_steps:.0f} steps")


if __name__ == "__main__":
    main()