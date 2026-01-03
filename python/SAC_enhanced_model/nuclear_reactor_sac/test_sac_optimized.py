"""
SAC Agent Testing and Visualization

Tests the trained SAC agent and generates comprehensive visualizations
of reactor control performance
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from nuclear_env_optimized import NuclearReactorEnv
from sac_networks import SACAgent


class ReactorTester:
    """Test and visualize trained SAC agent"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Load environment
        self.env = NuclearReactorEnv(reward_shaping='advanced')
        
        # Load trained agent
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        self.agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.device
        )
        
        print(f"Loading model from: {model_path}")
        self.agent.load(model_path)
        print(f"✓ Model loaded successfully")
    
    def run_episode(self, deterministic=True, max_steps=1000):
        """
        Run single episode and collect data
        
        Returns:
            Dictionary containing time series data
        """
        obs, _ = self.env.reset()
        done = False
        
        # Data collectors
        data = {
            'time': [],
            'power': [],
            'precursors': [],
            'fuel_temp': [],
            'coolant_temp': [],
            'pressure': [],
            'power_rate': [],
            'temp_rate': [],
            'control_rod': [],
            'coolant_flow': [],
            'reward': [],
            'q_value': []
        }
        
        step = 0
        total_reward = 0
        
        while not done and step < max_steps:
            # Get action
            action = self.agent.select_action(obs, deterministic=deterministic)
            
            # Estimate Q-value
            with torch.no_grad():
                state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
                q_value = self.agent.critic1(state_tensor, action_tensor).item()
            
            # Execute
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Record data
            data['time'].append(self.env.t)
            data['power'].append(obs[0])
            data['precursors'].append(obs[1])
            data['fuel_temp'].append(obs[2])
            data['coolant_temp'].append(obs[3])
            data['pressure'].append(obs[4])
            data['power_rate'].append(obs[5])
            data['temp_rate'].append(obs[6])
            data['control_rod'].append(action[0])
            data['coolant_flow'].append(action[1])
            data['reward'].append(reward)
            data['q_value'].append(q_value)
            
            total_reward += reward
            obs = next_obs
            step += 1
        
        # Episode statistics
        data['total_reward'] = total_reward
        data['episode_length'] = step
        data['success'] = not terminated
        data['termination_reason'] = info.get('reason', 'complete')
        
        return data
    
    def run_stress_test(self, num_episodes=10):
        """
        Run multiple episodes to assess robustness
        
        Returns:
            Statistics dictionary
        """
        print(f"\n{'='*60}")
        print(f"Running stress test: {num_episodes} episodes")
        print(f"{'='*60}\n")
        
        results = []
        
        for ep in range(num_episodes):
            data = self.run_episode(deterministic=True)
            results.append(data)
            
            print(f"Episode {ep+1}/{num_episodes} | "
                  f"Reward: {data['total_reward']:.1f} | "
                  f"Length: {data['episode_length']} | "
                  f"Status: {data['termination_reason']}")
        
        # Aggregate statistics
        stats = {
            'mean_reward': np.mean([r['total_reward'] for r in results]),
            'std_reward': np.std([r['total_reward'] for r in results]),
            'mean_length': np.mean([r['episode_length'] for r in results]),
            'success_rate': np.mean([r['success'] for r in results]) * 100,
            'max_reward': np.max([r['total_reward'] for r in results]),
            'min_reward': np.min([r['total_reward'] for r in results])
        }
        
        print(f"\n{'='*60}")
        print("STRESS TEST RESULTS")
        print(f"{'='*60}")
        print(f"Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"Reward Range: [{stats['min_reward']:.2f}, {stats['max_reward']:.2f}]")
        print(f"Success Rate: {stats['success_rate']:.1f}%")
        print(f"Mean Episode Length: {stats['mean_length']:.1f} steps")
        print(f"{'='*60}\n")
        
        return stats, results
    
    def visualize_episode(self, data, save_path='reactor_performance.png'):
        """
        Create comprehensive visualization of reactor control
        """
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        time = data['time']
        
        # ================================================================
        # Plot 1: Reactor Power
        # ================================================================
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(time, data['power'], color='#e74c3c', linewidth=2, label='Reactor Power')
        ax1.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Target (100%)')
        ax1.fill_between(time, 0.99, 1.01, alpha=0.2, color='green', label='±1% Tolerance')
        ax1.set_ylabel('Normalized Power', fontsize=11, fontweight='bold')
        ax1.set_title('REACTOR POWER CONTROL', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, max(time)])
        
        # ================================================================
        # Plot 2: Temperatures
        # ================================================================
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(time, data['fuel_temp'], color='#e67e22', linewidth=2, label='Fuel Temp')
        ax2.plot(time, data['coolant_temp'], color='#3498db', linewidth=2, label='Coolant Temp')
        ax2.axhline(y=1200, color='red', linestyle='--', linewidth=1.5, label='Safety Limit')
        ax2.set_ylabel('Temperature (°C)', fontsize=10, fontweight='bold')
        ax2.set_title('THERMAL DYNAMICS', fontsize=11, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # ================================================================
        # Plot 3: Pressure
        # ================================================================
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(time, data['pressure'], color='#9b59b6', linewidth=2)
        ax3.set_ylabel('Pressure (MPa)', fontsize=10, fontweight='bold')
        ax3.set_title('COOLANT PRESSURE', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # ================================================================
        # Plot 4: Control Actions
        # ================================================================
        ax4 = fig.add_subplot(gs[2, :])
        ax4.plot(time, data['control_rod'], color='#2ecc71', linewidth=2, label='Control Rod')
        ax4.plot(time, data['coolant_flow'], color='#1abc9c', linewidth=2, label='Coolant Flow')
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax4.set_ylabel('Control Signal', fontsize=10, fontweight='bold')
        ax4.set_title('SAC CONTROL ACTIONS', fontsize=11, fontweight='bold')
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([-1.1, 1.1])
        
        # ================================================================
        # Plot 5: Instantaneous Reward
        # ================================================================
        ax5 = fig.add_subplot(gs[3, 0])
        rewards = data['reward']
        ax5.plot(time, rewards, color='#34495e', linewidth=1.5, alpha=0.7)
        ax5.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax5.set_xlabel('Time (seconds)', fontsize=10, fontweight='bold')
        ax5.set_ylabel('Reward', fontsize=10, fontweight='bold')
        ax5.set_title('INSTANTANEOUS REWARD', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # ================================================================
        # Plot 6: Q-Value Estimate
        # ================================================================
        ax6 = fig.add_subplot(gs[3, 1])
        ax6.plot(time, data['q_value'], color='#e74c3c', linewidth=1.5, alpha=0.7)
        ax6.set_xlabel('Time (seconds)', fontsize=10, fontweight='bold')
        ax6.set_ylabel('Q-Value', fontsize=10, fontweight='bold')
        ax6.set_title('VALUE FUNCTION ESTIMATE', fontsize=11, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # ================================================================
        # Add summary statistics
        # ================================================================
        summary_text = (
            f"Episode Summary:\n"
            f"Total Reward: {data['total_reward']:.1f}\n"
            f"Duration: {data['episode_length']} steps ({max(time):.1f}s)\n"
            f"Max Fuel Temp: {max(data['fuel_temp']):.1f}°C\n"
            f"Power Std Dev: {np.std(data['power']):.4f}\n"
            f"Status: {data['termination_reason']}"
        )
        
        fig.text(0.02, 0.02, summary_text, fontsize=9, 
                family='monospace', verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Visualization saved to: {save_path}")
    
    def analyze_control_quality(self, data):
        """
        Compute control performance metrics
        
        Metrics:
        - RMSE (Power tracking)
        - ISE (Integral Squared Error)
        - Control effort
        - Settling time
        """
        
        power = np.array(data['power'])
        time = np.array(data['time'])
        
        # Power tracking error
        error = power - 1.0
        rmse = np.sqrt(np.mean(error ** 2))
        mae = np.mean(np.abs(error))
        
        # Integral squared error (ISE)
        ise = np.trapz(error ** 2, time)
        
        # Control effort (total variation)
        rod_actions = np.array(data['control_rod'])
        flow_actions = np.array(data['coolant_flow'])
        
        control_effort = np.sum(np.abs(np.diff(rod_actions))) + \
                        np.sum(np.abs(np.diff(flow_actions)))
        
        # Settling time (time to stay within ±1% of setpoint)
        tolerance = 0.01
        settled_mask = np.abs(error) < tolerance
        
        # Find first time it settles
        settling_time = None
        for i in range(len(settled_mask) - 50):  # Need 50 consecutive steps
            if np.all(settled_mask[i:i+50]):
                settling_time = time[i]
                break
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'ISE': ise,
            'Control_Effort': control_effort,
            'Settling_Time': settling_time,
            'Max_Power_Error': np.max(np.abs(error)),
            'Power_Std': np.std(power)
        }
        
        print(f"\n{'='*60}")
        print("CONTROL QUALITY METRICS")
        print(f"{'='*60}")
        for key, value in metrics.items():
            if value is not None:
                print(f"{key:20s}: {value:.6f}")
            else:
                print(f"{key:20s}: Not achieved")
        print(f"{'='*60}\n")
        
        return metrics


def main():
    """Main testing script"""
    
    # Path to trained model
    MODEL_PATH = 'models/SAC_optimized/best_model.pth'
    
    # Create tester
    tester = ReactorTester(MODEL_PATH)
    
    # Run single episode for detailed visualization
    print("\nRunning single episode for visualization...")
    data = tester.run_episode(deterministic=True)
    
    # Visualize
    tester.visualize_episode(data, save_path='reactor_performance_optimized.png')
    
    # Analyze control quality
    metrics = tester.analyze_control_quality(data)
    
    # Run stress test
    stats, all_results = tester.run_stress_test(num_episodes=20)
    
    # Compare best vs worst episode
    best_ep = max(all_results, key=lambda x: x['total_reward'])
    worst_ep = min(all_results, key=lambda x: x['total_reward'])
    
    print("\nGenerating comparison visualization...")
    tester.visualize_episode(best_ep, save_path='best_episode.png')
    tester.visualize_episode(worst_ep, save_path='worst_episode.png')
    
    print("\n✓ All tests complete!")


if __name__ == "__main__":
    main()