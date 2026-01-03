"""
Benchmark Comparison: Original vs Optimized SAC

Compares performance between:
- Your original implementation (Stable-Baselines3 SAC)
- The new optimized implementation

Key metrics:
- Reward convergence speed
- Final performance
- Stability
- Safety violations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from stable_baselines3 import SAC as SB3_SAC
import torch

from nuclear_env import NuclearReactorEnv as OriginalEnv
from nuclear_env_optimized import NuclearReactorEnv as OptimizedEnv
from sac_networks import SACAgent


class PerformanceComparator:
    """Compare original vs optimized implementations"""
    
    def __init__(self, original_model_path, optimized_model_path):
        
        # Load original model (Stable-Baselines3)
        self.original_env = OriginalEnv()
        print("Loading original SB3 model...")
        try:
            self.original_model = SB3_SAC.load(original_model_path, env=self.original_env)
            self.has_original = True
        except Exception as e:
            print(f"Warning: Could not load original model: {e}")
            self.has_original = False
        
        # Load optimized model
        self.optimized_env = OptimizedEnv(reward_shaping='advanced')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        state_dim = self.optimized_env.observation_space.shape[0]
        action_dim = self.optimized_env.action_space.shape[0]
        
        self.optimized_agent = SACAgent(state_dim, action_dim, device)
        print("Loading optimized model...")
        self.optimized_agent.load(optimized_model_path)
    
    def run_episode_original(self, deterministic=True):
        """Run episode with original model"""
        if not self.has_original:
            return None
        
        obs, _ = self.original_env.reset()
        done = False
        
        data = {
            'time': [],
            'power': [],
            'fuel_temp': [],
            'reward': []
        }
        
        total_reward = 0
        step = 0
        max_steps = 1000
        
        while not done and step < max_steps:
            action, _ = self.original_model.predict(obs, deterministic=deterministic)
            next_obs, reward, terminated, truncated, _ = self.original_env.step(action)
            done = terminated or truncated
            
            data['time'].append(self.original_env.t)
            data['power'].append(obs[0])
            data['fuel_temp'].append(obs[2])
            data['reward'].append(reward)
            
            total_reward += reward
            obs = next_obs
            step += 1
        
        data['total_reward'] = total_reward
        data['episode_length'] = step
        
        return data
    
    def run_episode_optimized(self, deterministic=True):
        """Run episode with optimized model"""
        
        obs, _ = self.optimized_env.reset()
        done = False
        
        data = {
            'time': [],
            'power': [],
            'fuel_temp': [],
            'reward': []
        }
        
        total_reward = 0
        step = 0
        max_steps = 1000
        
        while not done and step < max_steps:
            action = self.optimized_agent.select_action(obs, deterministic=deterministic)
            next_obs, reward, terminated, truncated, _ = self.optimized_env.step(action)
            done = terminated or truncated
            
            data['time'].append(self.optimized_env.t)
            data['power'].append(obs[0])
            data['fuel_temp'].append(obs[2])
            data['reward'].append(reward)
            
            total_reward += reward
            obs = next_obs
            step += 1
        
        data['total_reward'] = total_reward
        data['episode_length'] = step
        
        return data
    
    def compare_performance(self, num_episodes=20):
        """
        Run both models multiple times and compare
        """
        
        print(f"\n{'='*70}")
        print("PERFORMANCE COMPARISON")
        print(f"{'='*70}\n")
        
        original_rewards = []
        optimized_rewards = []
        
        original_lengths = []
        optimized_lengths = []
        
        # Run episodes
        for ep in range(num_episodes):
            print(f"Episode {ep+1}/{num_episodes}...", end=' ')
            
            # Original
            if self.has_original:
                orig_data = self.run_episode_original(deterministic=True)
                original_rewards.append(orig_data['total_reward'])
                original_lengths.append(orig_data['episode_length'])
            
            # Optimized
            opt_data = self.run_episode_optimized(deterministic=True)
            optimized_rewards.append(opt_data['total_reward'])
            optimized_lengths.append(opt_data['episode_length'])
            
            print("Done")
        
        # Compute statistics
        results = {
            'Original': {
                'mean_reward': np.mean(original_rewards) if original_rewards else 0,
                'std_reward': np.std(original_rewards) if original_rewards else 0,
                'min_reward': np.min(original_rewards) if original_rewards else 0,
                'max_reward': np.max(original_rewards) if original_rewards else 0,
                'mean_length': np.mean(original_lengths) if original_lengths else 0
            },
            'Optimized': {
                'mean_reward': np.mean(optimized_rewards),
                'std_reward': np.std(optimized_rewards),
                'min_reward': np.min(optimized_rewards),
                'max_reward': np.max(optimized_rewards),
                'mean_length': np.mean(optimized_lengths)
            }
        }
        
        # Print comparison
        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")
        
        print(f"\n{'Metric':<30} {'Original':<20} {'Optimized':<20}")
        print("-" * 70)
        
        if self.has_original:
            print(f"{'Mean Reward':<30} {results['Original']['mean_reward']:>18.2f} {results['Optimized']['mean_reward']:>18.2f}")
            print(f"{'Std Dev':<30} {results['Original']['std_reward']:>18.2f} {results['Optimized']['std_reward']:>18.2f}")
            print(f"{'Min Reward':<30} {results['Original']['min_reward']:>18.2f} {results['Optimized']['min_reward']:>18.2f}")
            print(f"{'Max Reward':<30} {results['Original']['max_reward']:>18.2f} {results['Optimized']['max_reward']:>18.2f}")
            print(f"{'Mean Episode Length':<30} {results['Original']['mean_length']:>18.1f} {results['Optimized']['mean_length']:>18.1f}")
            
            # Improvement
            improvement = ((results['Optimized']['mean_reward'] - results['Original']['mean_reward']) / 
                          abs(results['Original']['mean_reward']) * 100)
            print(f"\n{'Improvement':<30} {improvement:>18.1f}%")
        else:
            print(f"{'Mean Reward':<30} {'N/A':<20} {results['Optimized']['mean_reward']:>18.2f}")
            print(f"{'Std Dev':<30} {'N/A':<20} {results['Optimized']['std_reward']:>18.2f}")
            print(f"{'Mean Episode Length':<30} {'N/A':<20} {results['Optimized']['mean_length']:>18.1f}")
        
        print(f"{'='*70}\n")
        
        return results, (original_rewards, optimized_rewards)
    
    def visualize_comparison(self, original_data, optimized_data, save_path='comparison.png'):
        """
        Side-by-side visualization of original vs optimized
        """
        
        if original_data is None:
            print("Skipping comparison visualization (no original model)")
            return
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # ================================================================
        # Power Comparison
        # ================================================================
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(original_data['time'], original_data['power'], 
                color='blue', linewidth=2, alpha=0.7, label='Original')
        ax1.axhline(y=1.0, color='black', linestyle='--', linewidth=1)
        ax1.fill_between(original_data['time'], 0.99, 1.01, alpha=0.2, color='green')
        ax1.set_ylabel('Power', fontweight='bold')
        ax1.set_title('ORIGINAL MODEL - Power', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(optimized_data['time'], optimized_data['power'], 
                color='red', linewidth=2, alpha=0.7, label='Optimized')
        ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1)
        ax2.fill_between(optimized_data['time'], 0.99, 1.01, alpha=0.2, color='green')
        ax2.set_ylabel('Power', fontweight='bold')
        ax2.set_title('OPTIMIZED MODEL - Power', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # ================================================================
        # Temperature Comparison
        # ================================================================
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(original_data['time'], original_data['fuel_temp'], 
                color='blue', linewidth=2, alpha=0.7)
        ax3.axhline(y=1200, color='red', linestyle='--', linewidth=1.5, label='Safety Limit')
        ax3.set_ylabel('Fuel Temperature (°C)', fontweight='bold')
        ax3.set_title('ORIGINAL MODEL - Temperature', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(optimized_data['time'], optimized_data['fuel_temp'], 
                color='red', linewidth=2, alpha=0.7)
        ax4.axhline(y=1200, color='red', linestyle='--', linewidth=1.5, label='Safety Limit')
        ax4.set_ylabel('Fuel Temperature (°C)', fontweight='bold')
        ax4.set_title('OPTIMIZED MODEL - Temperature', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # ================================================================
        # Reward Comparison
        # ================================================================
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(original_data['time'], original_data['reward'], 
                color='blue', linewidth=1.5, alpha=0.7)
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax5.set_xlabel('Time (seconds)', fontweight='bold')
        ax5.set_ylabel('Reward', fontweight='bold')
        ax5.set_title(f"ORIGINAL - Total: {original_data['total_reward']:.1f}", fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(optimized_data['time'], optimized_data['reward'], 
                color='red', linewidth=1.5, alpha=0.7)
        ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax6.set_xlabel('Time (seconds)', fontweight='bold')
        ax6.set_ylabel('Reward', fontweight='bold')
        ax6.set_title(f"OPTIMIZED - Total: {optimized_data['total_reward']:.1f}", fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Comparison visualization saved to: {save_path}")


def main():
    """Main comparison script"""
    
    # Paths to models
    ORIGINAL_MODEL = "models/SAC/sac_nuclear_final.zip"
    OPTIMIZED_MODEL = "models/SAC_optimized/best_model.pth"
    
    # Create comparator
    comparator = PerformanceComparator(ORIGINAL_MODEL, OPTIMIZED_MODEL)
    
    # Run quantitative comparison
    results, rewards = comparator.compare_performance(num_episodes=20)
    
    # Run single episodes for visualization
    if comparator.has_original:
        print("\nGenerating comparison visualization...")
        original_data = comparator.run_episode_original(deterministic=True)
        optimized_data = comparator.run_episode_optimized(deterministic=True)
        
        comparator.visualize_comparison(original_data, optimized_data, 
                                       save_path='original_vs_optimized.png')
    
    # Create reward distribution plot
    if comparator.has_original:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        ax.hist(rewards[0], bins=15, alpha=0.5, label='Original', color='blue')
        ax.hist(rewards[1], bins=15, alpha=0.5, label='Optimized', color='red')
        ax.axvline(x=results['Original']['mean_reward'], color='blue', 
                  linestyle='--', linewidth=2, label='Original Mean')
        ax.axvline(x=results['Optimized']['mean_reward'], color='red', 
                  linestyle='--', linewidth=2, label='Optimized Mean')
        
        ax.set_xlabel('Episode Reward', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Reward Distribution Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('reward_distribution.png', dpi=150)
        plt.close()
        
        print("✓ Reward distribution saved to: reward_distribution.png")
    
    print("\n✓ All comparisons complete!")


if __name__ == "__main__":
    main()