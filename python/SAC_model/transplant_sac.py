import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import sys

# Setup paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nuclear_env import NuclearReactorEnv 

models_dir = "models/SAC"
log_dir = "logs"
# UPDATE THIS to the exact name of your latest zip file (e.g., 80000 or 100000)
old_model_path = f"{models_dir}/sac_nuclear_80000_steps.zip" 

# --- 1. CREATE A FRESH, CURIOUS AGENT ---
env = NuclearReactorEnv()
print("Creating a NEW model with High Curiosity (ent_coef=0.2)...")

# We initialize a NEW model instead of loading the old one.
# ent_coef=0.2 is 4x higher than before. It WILL explore.
new_model = SAC(
    "MlpPolicy", 
    env, 
    verbose=1, 
    tensorboard_log=log_dir, 
    ent_coef=0.2,   # <--- HARD CODED CURIOSITY
    learning_rate=0.0003
)

# --- 2. PERFORM THE TRANSPLANT ---
print(f"Transplanting learned weights from: {old_model_path}")
new_model.set_parameters(old_model_path)

print("-------------------------------------------------")
print("TRANSPLANT COMPLETE. Starting aggressive retraining.")
print("Watch the 'ent_coef' in logs -> It MUST say 0.2 now.")
print("-------------------------------------------------")

# --- 3. TRAIN ---
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=models_dir, name_prefix="sac_transplanted")
new_model.learn(total_timesteps=100000, callback=checkpoint_callback)

new_model.save(f"{models_dir}/sac_nuclear_fixed")