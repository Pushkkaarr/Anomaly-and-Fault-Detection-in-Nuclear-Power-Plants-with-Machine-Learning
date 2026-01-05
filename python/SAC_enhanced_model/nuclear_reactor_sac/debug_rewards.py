from nuclear_env_optimized import NuclearReactorEnv
import numpy as np

env = NuclearReactorEnv(reward_shaping='advanced')
obs, _ = env.reset()

episode_reward = 0
step_rewards = []

for step in range(100):  # Just 100 steps
    action = np.zeros(2)  # No control
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Step {step}: Reward = {reward:.2f}, Power = {obs[0]:.4f}, Temp = {obs[2]:.1f}")
    
    episode_reward += reward
    step_rewards.append(reward)
    
    if terminated or truncated:
        print(f"\nEpisode ended: {info.get('reason', 'unknown')}")
        break

print(f"\nTotal reward: {episode_reward:.2f}")
print(f"Average per step: {episode_reward / len(step_rewards):.2f}")
print(f"Max single step: {max(step_rewards):.2f}")
print(f"Min single step: {min(step_rewards):.2f}")