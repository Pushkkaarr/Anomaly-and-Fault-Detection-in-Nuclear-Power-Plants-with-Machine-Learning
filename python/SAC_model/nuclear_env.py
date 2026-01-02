import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp

# --- 1. PHYSICS CONSTANTS ---
# Standard PHWR 2000MWt Parameters
CONSTANTS = {
    'BETA': 0.0065,
    'LAMBDA': 0.08,
    'GEN_TIME': 0.001,
    'Cp_f': 300.0,
    'Cp_c': 4200.0,
    'UA': 2.5e6,
    'W_nominal': 8000.0,
    'Tin': 260.0,
    'P_nominal': 10.0,
    'ALPHA_F': -1.5e-5,
    'Tf0': 1095.0,
    'M_fuel': 40000.0,
    'M_coolant': 5000.0
}

class NuclearReactorEnv(gym.Env):
    """
    The 'Video Game' Interface for the Nuclear Reactor.
    FINAL REWARD UPDATE: 'Wider Goal Posts' to fix Lazy AI.
    """
    
    def __init__(self):
        super(NuclearReactorEnv, self).__init__()
        
        # --- 2. ACTION SPACE (The Controller) ---
        # Action[0]: Control Rod Speed (-1.0 to +1.0)
        # Action[1]: Pump Flow Change (-1.0 to +1.0)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # --- 3. OBSERVATION SPACE (The Dashboard) ---
        # [Power, Precursors, Fuel_Temp, Coolant_Temp, Pressure]
        self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(5,), dtype=np.float32)
        
        self.state = None
        self.t = 0.0
        self.dt = 0.1 # 100ms time step

    def reset(self, seed=None, options=None):
        """Resets the reactor to Equilibrium before every episode."""
        super().reset(seed=seed)
        
        # Calculate Equilibrium Conditions
        P0 = 1.0
        C0 = (CONSTANTS['BETA'] / (CONSTANTS['GEN_TIME'] * CONSTANTS['LAMBDA'])) * P0
        Q_eq = P0 * 2000e6
        Tc0 = CONSTANTS['Tin'] + Q_eq / (2 * CONSTANTS['W_nominal'] * CONSTANTS['Cp_c'])
        Tf0 = Tc0 + Q_eq / CONSTANTS['UA']
        
        self.state = np.array([P0, C0, Tf0, Tc0], dtype=np.float32)
        self.t = 0.0
        
        return self._get_obs(), {}

    def _get_obs(self):
        """Helper to get current sensor readings."""
        P, C, Tf, Tc = self.state
        
        # Calculate Pressure (Approximate Physics)
        T_avg = (Tc + CONSTANTS['Tin']) / 2
        Pressure = CONSTANTS['P_nominal'] * (T_avg / 285.0)
        
        return np.array([P, C, Tf, Tc, Pressure], dtype=np.float32)

    def step(self, action):
        """The Main Physics Loop (Runs every 0.1 seconds)"""
        P, C, Tf, Tc = self.state
        
        # --- PHYSICS SOLVER ---
        def reactor_dynamics(t, y):
            P_loc, C_loc, Tf_loc, Tc_loc = y
            
            # 1. Neutronics
            rho_doppler = CONSTANTS['ALPHA_F'] * (Tf_loc - CONSTANTS['Tf0'])
            
            # Reduced Control Rod Sensitivity (0.001) for Stability
            rho_control = action[0] * 0.001 
            
            rho_total = rho_doppler + rho_control
            
            dP_dt = ((rho_total - CONSTANTS['BETA']) / CONSTANTS['GEN_TIME']) * P_loc + CONSTANTS['LAMBDA'] * C_loc
            dC_dt = (CONSTANTS['BETA'] / CONSTANTS['GEN_TIME']) * P_loc - CONSTANTS['LAMBDA'] * C_loc
            
            # 2. Thermodynamics
            Power_Watts = P_loc * 2000e6
            Q_trans = CONSTANTS['UA'] * (Tf_loc - Tc_loc)
            dTf_dt = (Power_Watts - Q_trans) / (CONSTANTS['M_fuel'] * CONSTANTS['Cp_f'])
            
            # Flow Physics
            current_flow = CONSTANTS['W_nominal'] + (action[1] * 2000.0)
            current_flow = np.clip(current_flow, 100.0, 12000.0) # Prevent 0 flow
            
            Heat_Removal = 2 * current_flow * CONSTANTS['Cp_c'] * (Tc_loc - CONSTANTS['Tin'])
            dTc_dt = (Q_trans - Heat_Removal) / (CONSTANTS['M_coolant'] * CONSTANTS['Cp_c'])
            
            return [dP_dt, dC_dt, dTf_dt, dTc_dt]

        # Integrate for 0.1 seconds
        try:
            sol = solve_ivp(reactor_dynamics, [0, self.dt], self.state, method='RK45')
            self.state = sol.y[:, -1]
            self.t += self.dt
        except:
            # Handle math errors (if AI breaks physics)
            return self._get_obs(), -1000.0, True, False, {}
        
        # Unpack new state
        obs = self._get_obs()
        P_new, _, Tf_new, _, _ = obs
        
        # --- REWARD FUNCTION (The "Staircase" Version) ---
        reward = 0.0
        error = abs(P_new - 1.0)
        
        # 1. Base Penalty (The Stick)
        reward -= 10.0 * error
        
        # 2. THE STAIRCASE (The Crumb Trail)
        # Step 1: "Okay, you're close" (10% Error / Power 1.10)
        # This catches your CURRENT AI immediately.
        if error < 0.10:
            reward += 2.0
            
        # Step 2: "Getting warmer" (5% Error / Power 1.05)
        if error < 0.05:
            reward += 5.0
            
        # Step 3: "Almost there" (2% Error / Power 1.02)
        if error < 0.02:
            reward += 8.0
            
        # Step 4: "Bullseye!" (1% Error / Power 1.01)
        if error < 0.01:
            reward += 15.0 # HUGE JACKPOT
            
        # 3. Safety Penalty
        if Tf_new > 1200.0:
            reward -= 50.0
            
        # 4. Survival Bonus
        reward += 1.0

        # --- TERMINATION (Game Over) ---
        terminated = False
        if Tf_new > 1600.0 or P_new > 2.5: # Relaxed limits slightly
            terminated = True
            reward -= 1000.0 # Crash Penalty
            
        truncated = False
        if self.t > 100.0: # Success! (100 seconds survived)
            truncated = True

        return obs, reward, terminated, truncated, {}

    def render(self):
        pass