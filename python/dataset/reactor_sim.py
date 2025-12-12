import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- CONSTANTS (PHWR/CANDU approximations) ---
BETA = 0.0065        # Delayed neutron fraction
LAMBDA = 0.08        # Precursor decay constant (avg 1/s)
GEN_TIME = 0.001     # Neutron generation time (s)

# Thermal Props
Cp_f = 300           # Fuel specific heat (J/kg-K)
Cp_c = 4200          # Coolant specific heat (J/kg-K)
M_f = 40000          # Mass of fuel (kg)
M_c = 5000           # Mass of coolant in core (kg)
UA = 2.5e6           # Heat transfer coefficient (W/K)
W_nominal = 8000     # Nominal coolant flow rate (kg/s)
Tin = 260            # Inlet temp (C)

# --- THE PHYSICS KERNEL ---
def reactor_dynamics(t, y, reactivity_func, flow_func):
    """
    Solves the coupled Point Kinetics & Thermal-Hydraulics
    y[0]: Relative Neutron Power (n/n0) - Dimensionless
    y[1]: Precursor Concentration (Normalized)
    y[2]: Fuel Temperature (C)
    y[3]: Coolant Temperature (C)
    """
    P, C, Tf, Tc = y
    
    # Get current external controls
    rho = reactivity_func(t)
    W = flow_func(t)
    
    # 1. Neutron Kinetics (Point Kinetics)
    # dP/dt = ((rho - beta)/L)*P + lambda*C
    dP_dt = ((rho - BETA) / GEN_TIME) * P + LAMBDA * C
    
    # dC/dt = (beta/L)*P - lambda*C
    dC_dt = (BETA / GEN_TIME) * P - LAMBDA * C
    
    # 2. Thermal Hydraulics (Lumped Parameter)
    # Power generated in Watts (assuming P=1.0 is 2000 MWth)
    Power_Watts = P * 2000e6 
    
    # Heat transfer Fuel -> Coolant
    Q_trans = UA * (Tf - Tc)
    
    # Fuel Temp Eq: Heat In (Fission) - Heat Out (Transfer)
    dTf_dt = (Power_Watts - Q_trans) / (M_f * Cp_f)
    
    # Coolant Temp Eq: Heat In (Transfer) - Heat Out (Flow)
    # Note: If flow W is 0, the second term vanishes (LOFA)
    if W > 0:
        Heat_Removal = 2 * W * Cp_c * (Tc - Tin)
    else:
        Heat_Removal = 0 # Loss of Flow
        
    dTc_dt = (Q_trans - Heat_Removal) / (M_c * Cp_c)
    
    return [dP_dt, dC_dt, dTf_dt, dTc_dt]

# --- SCENARIO 1: SCRAM (Reactor Trip) ---
def run_scram_simulation():
    # Scenario: Reactor runs normal, then SCRAM at t=10s
    def reactivity_scram(t):
        if t < 10: return 0.0       # Critical
        return -0.05                # -5% delta k/k (Massive insertion)

    def flow_normal(t): 
        return W_nominal

    # Initial Conditions: Power=1.0, steady temps
    y0 = [1.0, 1.0, 600.0, 300.0] 
    
    # Solve
    sol = solve_ivp(
        fun=lambda t, y: reactor_dynamics(t, y, reactivity_scram, flow_normal),
        t_span=[0, 50],
        t_eval=np.linspace(0, 50, 500),
        y0=y0,
        method='LSODA' # Stiff solver required for kinetics
    )
    return sol

# --- SCENARIO 2: LOSS OF FLOW (Pump Trip) ---
def run_lofa_simulation():
    # Scenario: Pumps trip at t=10s, Reactivity stays 0 (Assume failure to Scram)
    def reactivity_normal(t): return 0.0

    def flow_coastdown(t):
        if t < 10: return W_nominal
        # Exponential pump coastdown
        return W_nominal * np.exp(-(t-10)/5.0) 

    y0 = [1.0, 1.0, 600.0, 300.0]
    
    sol = solve_ivp(
        fun=lambda t, y: reactor_dynamics(t, y, reactivity_normal, flow_coastdown),
        t_span=[0, 50],
        t_eval=np.linspace(0, 50, 500),
        y0=y0,
        method='LSODA'
    )
    return sol

# --- PLOTTING FOR VALIDATION ---
def validate():
    # Run Scram
    sol = run_scram_simulation()
    
    plt.figure(figsize=(10, 6))
    
    # Subplot 1: Power (The "Prompt Jump")
    plt.subplot(2, 1, 1)
    plt.plot(sol.t, sol.y[0], 'r-', label='Neutron Power')
    plt.title('Validation Check 1: Reactor SCRAM (Negative Reactivity Insertion)')
    plt.ylabel('Relative Power')
    plt.grid(True)
    plt.legend()
    
    # Subplot 2: Temps (Thermal Inertia)
    plt.subplot(2, 1, 2)
    plt.plot(sol.t, sol.y[2], 'orange', label='Fuel Temp')
    plt.plot(sol.t, sol.y[3], 'blue', label='Coolant Temp')
    plt.ylabel('Temperature (C)')
    plt.xlabel('Time (s)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('Validation_Graph.png')
    print("Validation graph generated: 'Validation_Graph.png'")

if __name__ == "__main__":
    validate()