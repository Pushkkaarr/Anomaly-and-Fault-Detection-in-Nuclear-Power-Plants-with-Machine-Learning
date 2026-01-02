import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nuclear_env import NuclearReactorEnv 

# --- CONFIG ---
models_dir = "models/SAC"
log_dir = "logs"
# UPDATE THIS TO YOUR LATEST ZIP FILE
old_model_path = f"{models_dir}/sac_nuclear_80000_steps.zip" 

# --- 1. LOAD THE LAZY BRAIN ---
env = NuclearReactorEnv()
print(f"Loading Lazy Model from: {old_model_path}")
model = SAC.load(old_model_path, env=env)

# --- 2. CHANGE THE SETTINGS (THE KICK) ---
# We increase entropy to make it explore again.
# 0.05 was too safe. 0.1 will force it to try new things.
model.ent_coef = 0.1 

# We also lower the learning rate slightly so it doesn't forget everything
model.learning_rate = 0.0001

print("-------------------------------------------------")
print("RETRAINING: Forcing the AI to fix the offset...")
print("Goal: Push Score from 140 -> 800+")
print("-------------------------------------------------")

# --- 3. TRAIN FOR ANOTHER 100k STEPS ---
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=models_dir, name_prefix="sac_retrained")
model.learn(total_timesteps=100000, callback=checkpoint_callback)

model.save(f"{models_dir}/sac_nuclear_final_polished")
print("DONE. The AI should be perfect now.")