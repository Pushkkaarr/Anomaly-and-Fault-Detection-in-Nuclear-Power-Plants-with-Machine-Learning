import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# --- 1. PHYSICS CONSTANTS (PHWR 2000MWt) ---
BETA = 0.0065
LAMBDA = 0.08
GEN_TIME = 0.001
Cp_f = 300           # Fuel Specific Heat of Uranium Dioxide UO2 used in reactors
Cp_c = 4200          # Coolant Specific Heat of Heavy Water
UA = 2.5e6           # Heat Transfer Coeff
W_nominal = 8000     # Nominal Flow (kg/s) 
Tin = 260            # Inlet Temp (C)------ 
P_nominal = 10.0     # Nominal Pressure (MPa)---------
ALPHA_F = -1.5e-5    # Doppler Feedback Coeff (Stabilizes the Reactor!)

# --- 2. THE STABILIZED PHYSICS ENGINE ---
def reactor_dynamics(t, y, scenario_type, fault_time):
    P, C, Tf, Tc = y
    
    # A. Control Inputs
    rho_control = 0.0 
    if scenario_type == 'Scram' and t > fault_time:
        rho_control = -0.05 # Scram Rods drop
        
    W = W_nominal
    if scenario_type == 'LOFA' and t > fault_time:
        W = W_nominal * np.exp(-(t - fault_time)/5.0) # Pump coastdown
        
    # B. Apply Physics Feedback (The Missing Link)
    # Reactivity = Control Rods + Doppler Feedback (Temp correction)
    Tf0 = 1095.0 # Reference Temp approx
    rho_total = rho_control + ALPHA_F * (Tf - Tf0)
    
    # C. Differential Equations
    dP_dt = ((rho_total - BETA) / GEN_TIME) * P + LAMBDA * C
    dC_dt = (BETA / GEN_TIME) * P - LAMBDA * C
    
    Power_Watts = P * 2000e6 #2000e6 is thermal power in watts
    Q_trans = UA * (Tf - Tc)
    
    dTf_dt = (Power_Watts - Q_trans) / (40000 * Cp_f)  #40000 kg of fuel mass used in calculation
    
    if W > 0:
        Heat_Removal = 2 * W * Cp_c * (Tc - Tin)
    else:
        Heat_Removal = 0 
        
    dTc_dt = (Q_trans - Heat_Removal) / (5000 * Cp_c) #5000 kg of coolant mass used in calculation
    
    return [dP_dt, dC_dt, dTf_dt, dTc_dt]

# --- 3. ROBUST DATA GENERATION LOOP ---
def generate_dataset_v2(num_episodes=100):
    all_data = []
    print(f"Generating {num_episodes} episodes with stabilized physics...")
    
    for episode in range(num_episodes):
        # 1. Setup Scenario
        scenario_type = np.random.choice(['Normal', 'Scram', 'LOFA'], p=[0.4, 0.3, 0.3])
        fault_time = np.random.uniform(10, 40)
        t_eval = np.linspace(0, 60, 600) # 60 seconds
        
        # 2. Initial Conditions (Equilibrium)
        P0 = 1.0 
        C0 = (BETA / (GEN_TIME * LAMBDA)) * P0
        Q_eq = P0 * 2000e6
        Tc0 = Tin + Q_eq / (2 * W_nominal * Cp_c)
        Tf0 = Tc0 + Q_eq / UA
        y0 = [P0, C0, Tf0, Tc0]
        
        # 3. Run Simulation
        try:
            sol = solve_ivp(
                fun=lambda t, y: reactor_dynamics(t, y, scenario_type, fault_time),
                t_span=[0, 60],
                t_eval=t_eval,
                y0=y0,
                method='LSODA'
            )
        except:
            print(f"Episode {episode} failed. Skipping.")
            continue
            
        # 4. Add Sensor Noise (Post-Processing)
        n_steps = len(sol.t)
        noise_P = np.random.normal(0, 0.01, n_steps)   
        noise_T = np.random.normal(0, 2.0, n_steps)    
        noise_Flow = np.random.normal(0, 10.0, n_steps)
        
        for i in range(n_steps):
            # Physical values
            val_P = sol.y[0][i]
            val_Tf = sol.y[2][i]
            val_Tc = sol.y[3][i]
            
            # Reconstruct Inputs
            current_flow = W_nominal
            if scenario_type == 'LOFA' and sol.t[i] > fault_time:
                 current_flow = W_nominal * np.exp(-(sol.t[i] - fault_time)/5.0)
            
            # Calculate Pressure
            T_avg = (val_Tc + Tin) / 2
            val_Press = P_nominal * (T_avg / 285.0) 
            
            # Labeling
            label = 0
            if sol.t[i] > fault_time:
                if scenario_type == 'Scram': label = 1
                if scenario_type == 'LOFA': label = 2
            
            row = {
                'Time': sol.t[i],
                'Power': max(0, val_P + noise_P[i]),          
                'Fuel_Temp': val_Tf + noise_T[i],             
                'Coolant_Temp': val_Tc + noise_T[i],          
                'Pressure': val_Press + np.random.normal(0, 0.05),
                'Flow': max(0, current_flow + noise_Flow[i]), 
                'Episode_ID': episode,
                'Label': label
            }
            all_data.append(row)
            
    df = pd.DataFrame(all_data)
    df.to_csv('nuclear_fault_data_v2.csv', index=False)
    print(f"Success! Saved {len(df)} rows to 'nuclear_fault_data_v2.csv'.")
    print("Stats Check (Max Power should be ~1.1, not 10^19):")
    print(df[['Power', 'Fuel_Temp']].describe()) 

if __name__ == "__main__":
    generate_dataset_v2(100)