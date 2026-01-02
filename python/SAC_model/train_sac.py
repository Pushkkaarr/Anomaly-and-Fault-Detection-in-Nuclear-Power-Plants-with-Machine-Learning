import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import sys

# Ensure the parent `python/` directory is on sys.path so we can import
# `nuclear_env` when running this script from the `SAC_model` folder.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nuclear_env import NuclearReactorEnv  # Import your physics environment

# --- 1. SETUP LOGGING ---
# Create a folder to save the trained brains
models_dir = "models/SAC"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# --- 2. INITIALIZE ENVIRONMENT ---
# Load the 'Video Game' we created earlier
env = NuclearReactorEnv()

# --- 3. DEFINE THE AI MODEL (SAC) ---
# MlpPolicy: Multi-Layer Perceptron (Standard Deep Neural Network)
# verbose=1: Print progress to the console
model = SAC(
    "MlpPolicy", 
    env, 
    verbose=1, 
    tensorboard_log=log_dir, # Remove this line if you didn't install tensorboard
    learning_rate=0.0003,
    buffer_size=50000, # Memory size (Replay Buffer)
    batch_size=256,    # How many memories to study at once
    
    # --- CRITICAL CHANGE: SAFETY PATCH ---
    # Changed from 'auto' to 0.05.
    # 'auto' makes the AI try crazy things to learn fast (Bad for reactors).
    # 0.05 makes the AI conservative and careful (Good for stability).
    ent_coef=0.05    
)

print("-------------------------------------------------")
print("STARTING SAC TRAINING: The Agent is learning to operate...")
print("Goal: Keep Power at 1.0 and Temp < 1200")
print("Settings: Conservative Mode (ent_coef=0.05), Long Run (200k steps)")
print("-------------------------------------------------")

# --- 4. START TRAINING ---
# --- CRITICAL CHANGE: INCREASED DURATION ---
# Increased from 50,000 to 200,000.
# This gives the AI enough time to stop panicking and start refining.
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=models_dir, name_prefix="sac_nuclear")

model.learn(total_timesteps=200000, callback=checkpoint_callback)

# --- 5. SAVE FINAL MODEL ---
model.save(f"{models_dir}/sac_nuclear_final")
print("DONE! Model saved to models/SAC/sac_nuclear_final.zip")