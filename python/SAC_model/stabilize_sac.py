import gymnasium as gym
from stable_baselines3 import SAC
import os
import sys

# Setup paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nuclear_env import NuclearReactorEnv 

models_dir = "models/SAC"
# UPDATE THIS to your latest zip file (e.g., 60000)
latest_model_path = f"{models_dir}/sac_nuclear_60000_steps.zip" 

print(f"Loading Model: {latest_model_path}")
env = NuclearReactorEnv()
model = SAC.load(latest_model_path, env=env)

# --- THE LOCK IN ---
# We force the AI to stop experimenting.
print("Freezing the brain (Zero Curiosity)...")
model.ent_coef = 0.00001  # Virtually zero entropy
model.learning_rate = 0.0001 # Slow learning rate

# Train for just a short time to let it "settle"
print("Stabilizing for 20,000 steps...")
model.learn(total_timesteps=20000)

model.save(f"{models_dir}/sac_nuclear_STABILIZED")
print("DONE. You now have a stable model saved as 'sac_nuclear_STABILIZED'.")