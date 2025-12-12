import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import random

# --- 1. PHYSICS CONSTANTS ---
BETA = 0.0065
LAMBDA = 0.08
GEN_TIME = 0.001
Cp_f = 300
Cp_c = 4200
UA = 2.5e6
W_nominal = 8000
Tin = 260
P_nominal = 10.0 # MPa (Pressure)

# --- 2. THE PHYSICS ENGINE (Optimized) ---
def reactor_dynamics(t, y, reactivity_func, flow_func):
    P, C, Tf, Tc = y
    rho = reactivity_func(t)
    W = flow_func(t)
    
    # Neutron Kinetics
    dP_dt = ((rho - BETA) / GEN_TIME) * P + LAMBDA * C
    dC_dt = (BETA / GEN_TIME) * P - LAMBDA * C
    
    # Thermal Hydraulics
    Power_Watts = P * 2000e6 
    Q_trans = UA * (Tf - Tc)
    dTf_dt = (Power_Watts - Q_trans) / (40000 * Cp_f)
    
    if W > 0:
        Heat_Removal = 2 * W * Cp_c * (Tc - Tin)
    else:
        Heat_Removal = 0 
    dTc_dt = (Q_trans - Heat_Removal) / (5000 * Cp_c)
    
    return [dP_dt, dC_dt, dTf_dt, dTc_dt]

# --- 3. DATA GENERATION LOOP ---
def generate_dataset(num_episodes=100):
    all_data = []
    
    print(f"Generating {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        # Randomly choose a scenario
        scenario_type = np.random.choice(['Normal', 'Scram', 'LOFA'], p=[0.4, 0.3, 0.3])
        
        # Duration of one episode (e.g., 60 seconds)
        t_span = [0, 60]
        t_eval = np.linspace(0, 60, 600) # 10Hz data (0.1s steps)
        
        # Define Fault Logic
        fault_time = np.random.uniform(10, 40) # Fault happens randomly between 10s and 40s
        
        def reactivity_func(t):
            if scenario_type == 'Scram' and t > fault_time:
                return -0.05
            # Add small random noise to normal operation
            return np.random.normal(0, 0.0001) 

        def flow_func(t):
            if scenario_type == 'LOFA' and t > fault_time:
                return W_nominal * np.exp(-(t - fault_time)/5.0)
            return W_nominal + np.random.normal(0, 10.0) # Small flow noise

        # Calculate Equilibrium (Stable Start)
        P0 = 1.0 + np.random.normal(0, 0.01) # Slight random start
        C0 = (BETA / (GEN_TIME * LAMBDA)) * P0
        Tc0 = Tin + (P0 * 2000e6) / (2 * W_nominal * Cp_c)
        Tf0 = Tc0 + (P0 * 2000e6) / UA
        y0 = [P0, C0, Tf0, Tc0]

        # Run Simulation
        sol = solve_ivp(
            fun=lambda t, y: reactor_dynamics(t, y, reactivity_func, flow_func),
            t_span=t_span,
            t_eval=t_eval,
            y0=y0,
            method='LSODA'
        )
        
        # Process Data for CSV
        for i in range(len(sol.t)):
            # Physics-Informed Pressure Calculation (P follows T_avg)
            T_avg = (sol.y[3][i] + Tin) / 2
            Pressure = P_nominal * (T_avg / 285.0) + np.random.normal(0, 0.05)
            
            # Determine Label at this timestamp
            current_label = 0 # Normal
            if sol.t[i] > fault_time:
                if scenario_type == 'Scram': current_label = 1
                if scenario_type == 'LOFA': current_label = 2
            
            row = {
                'Time': sol.t[i],
                'Power': sol.y[0][i],          # R
                'Fuel_Temp': sol.y[2][i],      # T (Fuel)
                'Coolant_Temp': sol.y[3][i],   # T (Coolant)
                'Pressure': Pressure,          # P
                'Flow': flow_func(sol.t[i]),   # F
                'Episode_ID': episode,
                'Label': current_label         # 0=Normal, 1=Scram, 2=LOFA
            }
            all_data.append(row)
            
    # Save
    df = pd.DataFrame(all_data)
    df.to_csv('nuclear_fault_data.csv', index=False)
    print(f"Dataset saved: 'nuclear_fault_data.csv' with {len(df)} rows.")
    return df

# Run it
if __name__ == "__main__":
    df = generate_dataset(num_episodes=50)
    print(df.head())