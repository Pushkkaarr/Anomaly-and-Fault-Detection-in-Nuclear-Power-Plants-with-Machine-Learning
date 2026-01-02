import gymnasium as gym
from stable_baselines3 import SAC
from nuclear_env import NuclearReactorEnv
import numpy as np
import matplotlib.pyplot as plt

# --- 1. LOAD THE TRAINED BRAIN ---
model_path = "models/SAC/sac_nuclear_70000_steps.zip"
print(f"Loading model from: {model_path}")

env = NuclearReactorEnv()
model = SAC.load(model_path, env=env)

# --- 2. RUN A STRESS TEST ---
# We will run 1 episode (100 seconds) and plot the results
obs, _ = env.reset()
done = False

# Data logging for plots
time_log = []
power_log = []
temp_log = []
action_rod_log = []
action_flow_log = []

print("Running Simulation with AI Control...")

while not done:
    # Ask the AI what to do
    # deterministic=True means "Do your best move", don't explore anymore
    action, _ = model.predict(obs, deterministic=True)
    
    # Execute the action in physics engine
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Log data
    time_log.append(env.t)
    power_log.append(obs[0])   # Power
    temp_log.append(obs[2])    # Fuel Temp
    action_rod_log.append(action[0])
    action_flow_log.append(action[1])
    
    done = terminated or truncated

# --- 3. VISUALIZE THE RESULTS ---
print("Simulation Complete. Generating Graph...")

plt.figure(figsize=(12, 8))

# Plot 1: Reactor Power
plt.subplot(3, 1, 1)
plt.plot(time_log, power_log, color='red', label='Reactor Power')
plt.axhline(y=1.0, color='black', linestyle='--', label='Target (100%)')
plt.title("AI Performance: Reactor Stability")
plt.ylabel("Normalized Power")
plt.legend()
plt.grid(True)

# Plot 2: Fuel Temperature
plt.subplot(3, 1, 2)
plt.plot(time_log, temp_log, color='orange', label='Fuel Temp')
plt.axhline(y=1200, color='red', linestyle='--', label='Safety Limit')
plt.ylabel("Temp (C)")
plt.legend()
plt.grid(True)

# Plot 3: AI Actions (The "Hands" of the AI)
plt.subplot(3, 1, 3)
plt.plot(time_log, action_rod_log, color='blue', label='Rod Speed Input')
plt.plot(time_log, action_flow_log, color='green', label='Flow Change Input')
plt.ylabel("Control Signal")
plt.xlabel("Time (seconds)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show() # This will pop up the graph