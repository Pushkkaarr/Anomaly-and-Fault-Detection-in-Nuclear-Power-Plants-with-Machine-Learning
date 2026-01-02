import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp

# --- 1. PHYSICS CONSTANTS (The "Rules of Nature") ---
# These match your Phase 1 validation exactly.
CONSTANTS = {
    'BETA': 0.0065,
    'LAMBDA': 0.08,
    'GEN_TIME': 0.001,
    'Cp_f': 300.0,
    'Cp_c': 4200.0,
    'UA': 2.5e6,
    'W_nominal': 8000.0,  # 8000 kg/s
    'Tin': 260.0,         # Inlet Temp
    'P_nominal': 10.0,    # 10 MPa
    'ALPHA_F': -1.5e-5,   # Doppler Feedback
    'Tf0': 1095.0,        # Ref Fuel Temp
    'M_fuel': 40000.0,    # Effective Fuel Mass
    'M_coolant': 5000.0   # Active Coolant Mass
}

class NuclearReactorEnv(gym.Env):
    """
    The 'Video Game' Interface for the Nuclear Reactor.
    The AI 'Agent' interacts with this class.
    """
    
    def __init__(self):
        super(NuclearReactorEnv, self).__init__()
        
        # --- 2. DEFINE THE "CONTROLLER" (Action Space) ---
        # The AI has 2 continuous knobs:
        # Action[0]: Control Rod Velocity (-1.0 to +1.0) -> Represents Insertion/Withdrawal speed
        # Action[1]: Pump Flow Change (-1.0 to +1.0) -> Represents ramping flow Up/Down
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # --- 3. DEFINE THE "DASHBOARD" (Observation Space) ---
        # The AI sees 5 values: [Power, Precursors, Fuel_Temp, Coolant_Temp, Pressure]
        # We define min/max limits for normalization (0 to inf roughly)
        self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(5,), dtype=np.float32)
        
        # Initial State
        self.state = None
        self.t = 0.0
        self.dt = 0.1 # Time step (100ms per action)

    def reset(self, seed=None, options=None):
        """
        Resets the reactor to a clean start (Equilibrium).
        Called at the start of every new training episode.
        """
        super().reset(seed=seed)
        
        # Start at perfect equilibrium
        P0 = 1.0
        C0 = (CONSTANTS['BETA'] / (CONSTANTS['GEN_TIME'] * CONSTANTS['LAMBDA'])) * P0
        Q_eq = P0 * 2000e6
        Tc0 = CONSTANTS['Tin'] + Q_eq / (2 * CONSTANTS['W_nominal'] * CONSTANTS['Cp_c'])
        Tf0 = Tc0 + Q_eq / CONSTANTS['UA']
        
        self.state = np.array([P0, C0, Tf0, Tc0], dtype=np.float32)
        self.t = 0.0
        
        # Return the observation (Add calculated pressure)
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        """Helper to format the sensor readings."""
        P, C, Tf, Tc = self.state
        
        # Calculate Pressure (Approximate Gay-Lussac Law)
        # P / T = constant -> P = P_ref * (T_avg / T_ref)
        T_avg = (Tc + CONSTANTS['Tin']) / 2
        Pressure = CONSTANTS['P_nominal'] * (T_avg / 285.0) # 285C is approx avg ref
        
        return np.array([P, C, Tf, Tc, Pressure], dtype=np.float32)

    def step(self, action):
        """
        The Main Game Loop.
        1. AI provides 'action'.
        2. We calculate Physics for 0.1 seconds.
        3. We return the new State and Reward.
        """
        # Unpack Action (Scale them to real physical units)
        # Action[0] is Rod Speed: -1 (Insert Fast) to +1 (Withdraw Fast)
        # We integrate this to get Reactivity, but for simplicity in SAC training,
        # let's say the AI controls the *Reactivity Addition Rate* directly.
        rho_control_change = action[0] * 1e-4 # Small change per step
        
        # Action[1] is Flow Change: -1 (Slow down) to +1 (Speed up)
        flow_change = action[1] * 100.0 # Change flow by up to 100 kg/s per step
        
        # Current Physics State
        P, C, Tf, Tc = self.state
        
        # --- PHYSICS SOLVER (The 4 Equations) ---
        def reactor_dynamics(t, y):
            P_loc, C_loc, Tf_loc, Tc_loc = y
            
            # 1. Neutronics (Point Kinetics)
            # Doppler Feedback
            rho_doppler = CONSTANTS['ALPHA_F'] * (Tf_loc - CONSTANTS['Tf0'])
            # Simplified: We assume 'action[0]' modifies the rod position/reactivity
            # For this training env, we treat action[0] as a direct reactivity adjustment to stabilize
            rho_total = rho_doppler + (action[0] * 0.005) # +/- 5 mk control authority
            
            dP_dt = ((rho_total - CONSTANTS['BETA']) / CONSTANTS['GEN_TIME']) * P_loc + CONSTANTS['LAMBDA'] * C_loc
            dC_dt = (CONSTANTS['BETA'] / CONSTANTS['GEN_TIME']) * P_loc - CONSTANTS['LAMBDA'] * C_loc
            
            # 2. Thermodynamics
            Power_Watts = P_loc * 2000e6
            Q_trans = CONSTANTS['UA'] * (Tf_loc - Tc_loc)
            dTf_dt = (Power_Watts - Q_trans) / (CONSTANTS['M_fuel'] * CONSTANTS['Cp_f'])
            
            # Flow Physics
            current_flow = CONSTANTS['W_nominal'] + (action[1] * 2000.0) # AI can swing flow by +/- 2000
            current_flow = np.clip(current_flow, 0, 10000) # Safety limits
            
            if current_flow > 0:
                Heat_Removal = 2 * current_flow * CONSTANTS['Cp_c'] * (Tc_loc - CONSTANTS['Tin'])
            else:
                Heat_Removal = 0
                
            dTc_dt = (Q_trans - Heat_Removal) / (CONSTANTS['M_coolant'] * CONSTANTS['Cp_c'])
            
            return [dP_dt, dC_dt, dTf_dt, dTc_dt]

        # Integrate for 0.1 seconds
        sol = solve_ivp(reactor_dynamics, [0, self.dt], self.state, method='RK45')
        self.state = sol.y[:, -1]
        self.t += self.dt
        
        # Get new observation
        obs = self._get_obs()
        P_new, _, Tf_new, Tc_new, Press_new = obs
        
        # --- 4. REWARD FUNCTION (The "Brain" Training) ---
        # Goal: Keep Power=1.0 and Temp Safe.
        
        reward = 0.0
        
        # Penalty for deviation from Power=1.0
        reward -= 10.0 * abs(P_new - 1.0)
        
        # Penalty for High Temp (Safety)
        if Tf_new > 1200.0:
            reward -= 50.0 # Big penalty for overheating
            
        # Penalty for unstable actions (Don't jitter the rods!)
        reward -= 0.1 * np.sum(np.square(action))
        
        # Bonus for survival
        reward += 1.0

        # Check for "Game Over" (Meltdown)
        terminated = False
        if Tf_new > 1500.0 or P_new > 2.0:
            terminated = True
            reward -= 1000.0 # Game Over Penalty
            
        truncated = False
        if self.t > 100.0: # Episode ends after 100 seconds
            truncated = True

        return obs, reward, terminated, truncated, {}

    def render(self):
        pass # Optional visualization