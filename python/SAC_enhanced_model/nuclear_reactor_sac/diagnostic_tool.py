"""
Diagnostic Tool for SAC Training Issues

Quick checks for common problems:
- Reward function verification
- Network gradient flow
- Buffer quality
- Action distribution
- Q-value estimates
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from nuclear_env_optimized import NuclearReactorEnv
from sac_networks import SACAgent
from replay_buffer import ReplayBuffer


class SACDiagnostics:
    """Comprehensive diagnostic tool"""
    
    def __init__(self, model_path=None):
        self.env = NuclearReactorEnv(reward_shaping='advanced')
        
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.agent = SACAgent(state_dim, action_dim, device)
        
        if model_path:
            print(f"Loading model: {model_path}")
            self.agent.load(model_path)
    
    def test_reward_function(self, num_samples=100):
        """
        Test reward function across different states
        Verifies that reward increases as error decreases
        """
        
        print(f"\n{'='*60}")
        print("REWARD FUNCTION TEST")
        print(f"{'='*60}\n")
        
        power_errors = np.linspace(0, 0.2, num_samples)
        rewards = []
        
        for error in power_errors:
            # Create a mock stable state with this power error
            power = 1.0 + error
            fuel_temp = 1100.0  # Safe temperature
            coolant_temp = 290.0
            power_rate = 0.001  # Very stable
            temp_rate = 0.1
            action = np.array([0.0, 0.0])  # No control action
            
            # Call the reward function directly WITHOUT running physics
            reward = self.env._calculate_reward(
                power, fuel_temp, coolant_temp, 
                power_rate, temp_rate, action
            )
            
            rewards.append(reward)
        
        # Plot reward landscape
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        ax.plot(power_errors * 100, rewards, linewidth=2, color='blue')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=1, color='green', linestyle='--', alpha=0.5, label='±1% Target')
        ax.axvline(x=2, color='yellow', linestyle='--', alpha=0.5, label='±2% Good')
        ax.axvline(x=5, color='orange', linestyle='--', alpha=0.5, label='±5% Acceptable')
        
        ax.set_xlabel('Power Error (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Reward', fontsize=12, fontweight='bold')
        ax.set_title('Reward Function Landscape', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('reward_landscape.png', dpi=150)
        plt.close()
        
        # Analysis
        print("Reward at key tolerances:")
        test_points = [0.01, 0.02, 0.05, 0.10, 0.15]
        for error in test_points:
            idx = np.argmin(np.abs(power_errors - error))
            print(f"  ±{error*100:.0f}%: {rewards[idx]:>8.2f}")
        
        # Check monotonicity (reward should decrease with error)
        # Allow small tolerance for numerical noise
        differences = np.diff(rewards)
        non_monotonic_count = np.sum(differences > 0.1)  # Small positive changes OK
        
        if non_monotonic_count < 5:
            print("\n✓ Reward function is monotonic (GOOD)")
        else:
            print(f"\n✗ WARNING: Reward function not monotonic! ({non_monotonic_count} violations)")
        
        print("\n✓ Reward landscape saved to: reward_landscape.png")
    
    def test_action_distribution(self, num_samples=1000):
        """
        Sample actions from policy and verify they're well-distributed
        """
        
        print(f"\n{'='*60}")
        print("ACTION DISTRIBUTION TEST")
        print(f"{'='*60}\n")
        
        obs, _ = self.env.reset()
        
        actions_rod = []
        actions_flow = []
        
        # Sample actions
        for _ in range(num_samples):
            action = self.agent.select_action(obs, deterministic=False)
            actions_rod.append(action[0])
            actions_flow.append(action[1])
        
        actions_rod = np.array(actions_rod)
        actions_flow = np.array(actions_flow)
        
        # Statistics
        print(f"Control Rod Actions:")
        print(f"  Mean: {np.mean(actions_rod):.4f}")
        print(f"  Std:  {np.std(actions_rod):.4f}")
        print(f"  Range: [{np.min(actions_rod):.4f}, {np.max(actions_rod):.4f}]")
        
        print(f"\nCoolant Flow Actions:")
        print(f"  Mean: {np.mean(actions_flow):.4f}")
        print(f"  Std:  {np.std(actions_flow):.4f}")
        print(f"  Range: [{np.min(actions_flow):.4f}, {np.max(actions_flow):.4f}]")
        
        # Plot distributions
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].hist(actions_rod, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Action Value', fontweight='bold')
        axes[0].set_ylabel('Frequency', fontweight='bold')
        axes[0].set_title('Control Rod Action Distribution', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(actions_flow, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Action Value', fontweight='bold')
        axes[1].set_ylabel('Frequency', fontweight='bold')
        axes[1].set_title('Coolant Flow Action Distribution', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('action_distributions.png', dpi=150)
        plt.close()
        
        # Checks
        checks = []
        
        if abs(np.mean(actions_rod)) < 0.1:
            checks.append("✓ Rod actions centered near zero")
        else:
            checks.append("✗ Rod actions biased (mean far from 0)")
        
        if abs(np.mean(actions_flow)) < 0.1:
            checks.append("✓ Flow actions centered near zero")
        else:
            checks.append("✗ Flow actions biased (mean far from 0)")
        
        if np.std(actions_rod) > 0.1:
            checks.append("✓ Rod actions have sufficient variance")
        else:
            checks.append("✗ Rod actions too deterministic")
        
        if np.std(actions_flow) > 0.1:
            checks.append("✓ Flow actions have sufficient variance")
        else:
            checks.append("✗ Flow actions too deterministic")
        
        print("\nAction Distribution Checks:")
        for check in checks:
            print(f"  {check}")
        
        print("\n✓ Action distributions saved to: action_distributions.png")
    
    def test_q_value_estimates(self, num_episodes=10):
        """
        Check if Q-values are reasonable and correlate with returns
        """
        
        print(f"\n{'='*60}")
        print("Q-VALUE ESTIMATION TEST")
        print(f"{'='*60}\n")
        
        q_values_list = []
        returns_list = []
        
        for ep in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            
            episode_q_values = []
            episode_rewards = []
            
            while not done:
                # Get action and Q-value
                action = self.agent.select_action(obs, deterministic=True)
                
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.agent.device)
                    action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.agent.device)
                    q_value = self.agent.critic1(state_tensor, action_tensor).item()
                
                episode_q_values.append(q_value)
                
                # Execute
                obs, reward, terminated, truncated, _ = self.env.step(action)
                episode_rewards.append(reward)
                done = terminated or truncated
            
            # Compute discounted returns
            gamma = 0.99
            returns = []
            R = 0
            for r in reversed(episode_rewards):
                R = r + gamma * R
                returns.insert(0, R)
            
            q_values_list.extend(episode_q_values)
            returns_list.extend(returns)
        
        q_values_list = np.array(q_values_list)
        returns_list = np.array(returns_list)
        
        # Correlation
        correlation = np.corrcoef(q_values_list, returns_list)[0, 1]
        
        print(f"Q-value vs Return Correlation: {correlation:.4f}")
        
        if correlation > 0.7:
            print("✓ Strong correlation (GOOD - Q-values are accurate)")
        elif correlation > 0.4:
            print("⚠ Moderate correlation (ACCEPTABLE - could be better)")
        else:
            print("✗ Weak correlation (BAD - Q-values unreliable)")
        
        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        ax.scatter(returns_list, q_values_list, alpha=0.3, s=10)
        
        # Fit line
        z = np.polyfit(returns_list, q_values_list, 1)
        p = np.poly1d(z)
        x_line = np.linspace(returns_list.min(), returns_list.max(), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f'Fit (corr={correlation:.3f})')
        
        ax.set_xlabel('Actual Return', fontsize=12, fontweight='bold')
        ax.set_ylabel('Q-Value Estimate', fontsize=12, fontweight='bold')
        ax.set_title('Q-Value Accuracy', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('q_value_accuracy.png', dpi=150)
        plt.close()
        
        print("\n✓ Q-value analysis saved to: q_value_accuracy.png")
    
    def test_gradient_flow(self):
        """
        Check if gradients are flowing properly through networks
        """
        
        print(f"\n{'='*60}")
        print("GRADIENT FLOW TEST")
        print(f"{'='*60}\n")
        
        # Create dummy data
        states = torch.randn(32, 8).to(self.agent.device)
        actions = torch.randn(32, 2).to(self.agent.device)
        
        # Forward pass
        self.agent.actor.train()
        new_actions, log_probs, _ = self.agent.actor.sample(states)
        
        q1 = self.agent.critic1(states, new_actions)
        q2 = self.agent.critic2(states, new_actions)
        
        # Backward pass
        actor_loss = (self.agent.alpha * log_probs - torch.min(q1, q2)).mean()
        actor_loss.backward()
        
        # Check gradients
        actor_grads = []
        for name, param in self.agent.actor.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                actor_grads.append((name, grad_norm))
        
        print("Actor Gradient Norms:")
        for name, grad_norm in actor_grads:
            status = "✓" if grad_norm > 1e-7 else "✗"
            print(f"  {status} {name:40s}: {grad_norm:.6f}")
        
        # Check for vanishing/exploding gradients
        grad_norms = [g for _, g in actor_grads]
        
        if all(g > 1e-7 for g in grad_norms):
            print("\n✓ All gradients flowing (GOOD)")
        elif any(g < 1e-7 for g in grad_norms):
            print("\n⚠ WARNING: Some gradients vanishing!")
        
        if any(g > 10.0 for g in grad_norms):
            print("⚠ WARNING: Some gradients exploding!")
        
        # Zero gradients
        self.agent.actor.zero_grad()
    
    def run_full_diagnostics(self):
        """Run all diagnostic tests"""
        
        print(f"\n{'='*60}")
        print("RUNNING FULL SAC DIAGNOSTICS")
        print(f"{'='*60}")
        
        self.test_reward_function()
        self.test_action_distribution()
        self.test_q_value_estimates()
        self.test_gradient_flow()
        
        print(f"\n{'='*60}")
        print("DIAGNOSTICS COMPLETE")
        print(f"{'='*60}")
        print("\nGenerated files:")
        print("  - reward_landscape.png")
        print("  - action_distributions.png")
        print("  - q_value_accuracy.png")
        print("\nReview these plots to identify issues.")


def main():
    """Main diagnostic script"""
    
    # import sys
    
    # if len(sys.argv) > 1:
    #     model_path = sys.argv[1]
    #     print(f"Running diagnostics on trained model: {model_path}")
    # else:
    #     model_path = None
    #     print("Running diagnostics on untrained model")

    model_path = "models/SAC_optimized/best_model.pth"
    print(f"Running diagnostics on trained model: {model_path}")
    diagnostics = SACDiagnostics(model_path)
    diagnostics.run_full_diagnostics()


if __name__ == "__main__":
    main()