import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp

# ============================================================================
# IMMUTABLE PHYSICS CONSTANTS - DO NOT MODIFY
# Standard PHWR 2000MWt Parameters
# ============================================================================
CONSTANTS = {
    'BETA': 0.0065,           # Delayed neutron fraction
    'LAMBDA': 0.08,           # Decay constant (1/s)
    'GEN_TIME': 0.001,        # Neutron generation time (s)
    'Cp_f': 300.0,            # Fuel specific heat (J/kg·K)
    'Cp_c': 4200.0,           # Coolant specific heat (J/kg·K)
    'UA': 2.5e6,              # Heat transfer coefficient (W/K)
    'W_nominal': 8000.0,      # Nominal coolant flow (kg/s)
    'Tin': 260.0,             # Inlet temperature (°C)
    'P_nominal': 10.0,        # Nominal pressure (MPa)
    'ALPHA_F': -1.5e-5,       # Fuel temperature reactivity coefficient
    'Tf0': 1095.0,            # Reference fuel temperature (°C)
    'M_fuel': 40000.0,        # Fuel mass (kg)
    'M_coolant': 5000.0       # Coolant mass (kg)
}

class NuclearReactorEnv(gym.Env):
    """
    Optimized Nuclear Reactor Environment for SAC Training
    
    THE 4 IMMUTABLE PHYSICS FORMULAS:
    
    1. NEUTRON KINETICS:
       dP/dt = [(ρ - β)/Λ] × P + λ × C
       dC/dt = (β/Λ) × P - λ × C
    
    2. FUEL TEMPERATURE:
       dTf/dt = (Q_fission - Q_transfer) / (M_fuel × Cp_f)
       where Q_transfer = UA × (Tf - Tc)
    
    3. COOLANT TEMPERATURE:
       dTc/dt = (Q_transfer - Q_removal) / (M_coolant × Cp_c)
       where Q_removal = 2W × Cp_c × (Tc - Tin)
    
    4. REACTIVITY FEEDBACK:
       ρ_total = α_f(Tf - Tf0) + ρ_control
    """
    
    metadata = {'render_modes': []}
    
    def __init__(self, reward_shaping='advanced'):
        super().__init__()
        
        # Action Space: Continuous control
        # [0]: Control rod reactivity insertion rate ($/s)
        # [1]: Coolant flow fractional change
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation Space: [Power, Precursors, Fuel_Temp, Coolant_Temp, Pressure, 
        #                      Power_Rate, Temp_Rate, Time_in_Episode]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 200.0, 200.0, 0.0, -10.0, -100.0, 0.0], dtype=np.float32),
            high=np.array([3.0, 1.0, 2000.0, 500.0, 20.0, 10.0, 100.0, 200.0], dtype=np.float32),
            dtype=np.float32
        )
        
        self.reward_shaping = reward_shaping
        self.dt = 0.1  # 100ms timestep
        self.max_episode_time = 100.0  # Add this line

        # State tracking
        self.state = None
        self.t = 0.0
        self.prev_power = 1.0
        self.prev_temp = 1095.0
        self.current_flow = CONSTANTS['W_nominal']
        self.cumulative_violation_time = 0.0
        self.episode_max_temp = 0.0
        
        # Performance metrics
        self.episode_rewards = []
        self.stability_history = []
        
    def reset(self, seed=None, options=None):
        """Initialize reactor at equilibrium with small perturbations"""
        super().reset(seed=seed)
        
        # Equilibrium calculations
        P0 = 1.0
        C0 = (CONSTANTS['BETA'] / (CONSTANTS['GEN_TIME'] * CONSTANTS['LAMBDA'])) * P0
        Q_eq = P0 * 2000e6
        Tc0 = CONSTANTS['Tin'] + Q_eq / (2 * CONSTANTS['W_nominal'] * CONSTANTS['Cp_c'])
        Tf0 = Tc0 + Q_eq / CONSTANTS['UA']
        
        # Add small random perturbations to make training more robust
        if self.np_random is not None:
            P0 += self.np_random.uniform(-0.02, 0.02)
            Tf0 += self.np_random.uniform(-5, 5)
            Tc0 += self.np_random.uniform(-2, 2)
        
        # Add this for variable episodes:
        self.max_episode_time = self.np_random.uniform(100, 150) if self.np_random is not None else 100.0
        self.state = np.array([P0, C0, Tf0, Tc0], dtype=np.float32)
        self.t = 0.0
        self.prev_power = P0
        self.prev_temp = Tf0
        self.current_flow = CONSTANTS['W_nominal']
        self.cumulative_violation_time = 0.0
        self.episode_max_temp = Tf0
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        """Enhanced observation with derivatives and time"""
        P, C, Tf, Tc = self.state
        
        # Pressure approximation
        T_avg = (Tc + CONSTANTS['Tin']) / 2
        Pressure = CONSTANTS['P_nominal'] * (T_avg / 285.0)
        
        # Rate of change estimates
        power_rate = (P - self.prev_power) / self.dt if self.dt > 0 else 0.0
        temp_rate = (Tf - self.prev_temp) / self.dt if self.dt > 0 else 0.0
        
        return np.array([
            P, C, Tf, Tc, Pressure, 
            power_rate, temp_rate, self.t
        ], dtype=np.float32)
    
    def step(self, action):
        """Execute one physics timestep with given control action"""
        P, C, Tf, Tc = self.state
        
        # Clip actions for safety
        action = np.clip(action, self.action_space.low, self.action_space.high)
        flow_multiplier = 1.0 + (action[1] * 0.25)
        self.current_flow = float(
            np.clip(
                CONSTANTS['W_nominal'] * flow_multiplier,
                0.2 * CONSTANTS['W_nominal'],
                1.5 * CONSTANTS['W_nominal']
            )
        )
        
        # ====================================================================
        # PHYSICS INTEGRATION - THE 4 FORMULAS IN ACTION
        # ====================================================================
        def reactor_dynamics(t, y):
            P_loc, C_loc, Tf_loc, Tc_loc = y
            
            # FORMULA 4: Reactivity calculation
            rho_doppler = CONSTANTS['ALPHA_F'] * (Tf_loc - CONSTANTS['Tf0'])
            # Scale control action with physical limits
            # Real PHWR: ~0.001$/s max rod speed
            rho_control = action[0] * 0.001  # Reduced for realism
            rho_total = rho_doppler + rho_control
            
            # FORMULA 1: Neutron kinetics (point kinetics with one delayed group)
            dP_dt = ((rho_total - CONSTANTS['BETA']) / CONSTANTS['GEN_TIME']) * P_loc + \
                    CONSTANTS['LAMBDA'] * C_loc
            dC_dt = (CONSTANTS['BETA'] / CONSTANTS['GEN_TIME']) * P_loc - \
                    CONSTANTS['LAMBDA'] * C_loc
            
            # FORMULA 2: Fuel temperature dynamics
            Power_Watts = P_loc * 2000e6
            Q_trans = CONSTANTS['UA'] * (Tf_loc - Tc_loc)
            dTf_dt = (Power_Watts - Q_trans) / (CONSTANTS['M_fuel'] * CONSTANTS['Cp_f'])
            
            # FORMULA 3: Coolant temperature dynamics
            # Scale flow action with realistic pump response
            # Real pumps can't change flow instantly - add inertia
            flow_multiplier = 1.0 + (action[1] * 0.25)  # ±25% max (more realistic)
            current_flow = CONSTANTS['W_nominal'] * flow_multiplier
            # Physical limits: minimum 20%, maximum 150% of nominal
            current_flow = np.clip(current_flow, 0.2 * CONSTANTS['W_nominal'], 1.5 * CONSTANTS['W_nominal'])
            
            Heat_Removal = 2 * current_flow * CONSTANTS['Cp_c'] * (Tc_loc - CONSTANTS['Tin'])
            dTc_dt = (Q_trans - Heat_Removal) / (CONSTANTS['M_coolant'] * CONSTANTS['Cp_c'])
            
            return [dP_dt, dC_dt, dTf_dt, dTc_dt]
        
        # Integrate physics
        try:
            sol = solve_ivp(
                reactor_dynamics, 
                [0, self.dt], 
                self.state, 
                method='RK45',
                max_step=0.01  # Finer integration for accuracy
            )
            new_state = sol.y[:, -1]
            
            # Sanity checks
            if np.any(np.isnan(new_state)) or np.any(np.isinf(new_state)):
                return self._get_obs(), -1000.0, True, False, {'reason': 'numerical_instability'}
            
            self.state = new_state
            self.t += self.dt
            
        except Exception as e:
            return self._get_obs(), -1000.0, True, False, {'reason': f'integration_error: {str(e)}'}
        
        # ====================================================================
        # ADVANCED REWARD SHAPING
        # ====================================================================
        obs = self._get_obs()
        P_new, _, Tf_new, Tc_new, Press, P_rate, T_rate, _ = obs
        
        reward = self._calculate_reward(P_new, Tf_new, Tc_new, P_rate, T_rate, action)
        
        # Update tracking
        self.prev_power = P_new
        self.prev_temp = Tf_new
        self.episode_max_temp = max(self.episode_max_temp, Tf_new)
        
        # ====================================================================
        # TERMINATION CONDITIONS
        # ====================================================================
        terminated = False
        info = {
            'coolant_flow_actual': float(self.current_flow)
        }
        
        # Critical safety violations
        if Tf_new > 1600.0:
            terminated = True
            reward -= 2000.0
            info['reason'] = 'fuel_meltdown'
        elif P_new > 2.5:
            terminated = True
            reward -= 1500.0
            info['reason'] = 'power_excursion'
        elif P_new < 0.1:
            terminated = True
            reward -= 1000.0
            info['reason'] = 'reactor_shutdown'
        elif Tc_new > 400.0:
            terminated = True
            reward -= 1200.0
            info['reason'] = 'coolant_boiling'
        
        # Episode success
        truncated = False
        if self.t >= self.max_episode_time:
            truncated = True
            # Bonus for completing episode
            reward += 500.0 if self.episode_max_temp < 1200.0 else 100.0
            info['reason'] = 'episode_complete'
        
        return obs, reward, terminated, truncated, info
    
    def _calculate_reward(self, P, Tf, Tc, P_rate, T_rate, action):
        """
        Nuclear Engineering-Grade Reward Function
        
        Principles:
        1. Power control via REACTIVITY (rods) - primary
        2. Heat removal via FLOW (coolant) - secondary
        3. Penalize using wrong tool for the job
        4. Reward coordinated control strategies
        """
        
        reward = 0.0
        power_error = abs(P - 1.0)
        rod_action = action[0]
        flow_action = action[1]
        
        # =================================================================
        # 1. PRIMARY OBJECTIVE: Power Setpoint Tracking
        # =================================================================
        # Exponential penalty - tighter control gets exponentially better rewards
        power_penalty = -50.0 * (power_error ** 2)
        reward += power_penalty
        
        # Precision tiers (non-stacking)
        if power_error < 0.005:  # ±0.5% - Exceptional
            reward += 40.0
        elif power_error < 0.01:  # ±1% - Excellent
            reward += 25.0
        elif power_error < 0.02:  # ±2% - Good
            reward += 10.0
        elif power_error < 0.05:  # ±5% - Acceptable
            reward += 3.0
        
        # =================================================================
        # 2. TEMPERATURE SAFETY - Critical Constraint
        # =================================================================
        temp_margin = 1200.0 - Tf
        
        if Tf > 1200.0:
            # Exponential danger penalty
            overheat = Tf - 1200.0
            reward -= 200.0 * (overheat / 100.0) ** 2
            self.cumulative_violation_time += self.dt
        elif temp_margin < 100.0:
            # Approaching danger - warning penalty
            reward -= 5.0 * (1.0 - temp_margin / 100.0)
        
        # =================================================================
        # 3. CONTROL STRATEGY ASSESSMENT - The Nuclear Engineering Part
        # =================================================================
        
        # A. REACTIVITY CONTROL (Rods) Assessment
        # Rods should be used for POWER changes, not temperature
        rod_magnitude = abs(rod_action)
        power_rate_magnitude = abs(P_rate)
        
        # Good: Using rods when power is changing
        if power_rate_magnitude > 0.02:  # Power is drifting
            if rod_magnitude > 0.05:  # Agent is using rods (correct!)
                reward += 5.0  # Reward proper reactivity control
            else:  # Power drifting but not using rods (wrong!)
                reward -= 3.0  # Penalty for not addressing reactivity
        
        # Penalize excessive rod movement when power is stable
        if power_error < 0.02 and rod_magnitude > 0.3:
            reward -= 8.0  # Don't thrash rods when power is fine
        
        # B. THERMAL MANAGEMENT (Flow) Assessment
        # Flow should be used for TEMPERATURE control
        flow_magnitude = abs(flow_action)
        temp_rate_magnitude = abs(T_rate)
        
        # Good: Using flow when temperature is changing
        if temp_rate_magnitude > 2.0:  # Temperature is drifting
            if flow_magnitude > 0.1:  # Agent is adjusting flow (correct!)
                reward += 5.0  # Reward proper thermal management
            else:  # Temperature drifting but not adjusting flow (wrong!)
                reward -= 3.0
        
        # Penalize excessive flow changes when temperature is stable
        if abs(Tf - 1095.0) < 10.0 and flow_magnitude > 0.5:
            reward -= 8.0  # Don't slam pumps when temp is fine
        
        # C. COORDINATION BONUS - Simplified
        # Reward using both controls together (less specific conditions)
        rod_magnitude = abs(rod_action)
        flow_magnitude = abs(flow_action)
        if rod_magnitude > 0.05 and flow_magnitude > 0.1:
            # Both controls active = coordinated response
            reward += 3.0  # Reduced from 10.0
        
        # =================================================================
        # 4. CONTROL QUALITY - Smooth Professional Operation
        # =================================================================
        
        # Penalize EXTREME actions (slamming controls)
        if abs(rod_action) > 0.8:
            reward -= 10.0 * (abs(rod_action) - 0.8)  # Emergency use only
        
        if abs(flow_action) > 0.9:
            reward -= 15.0 * (abs(flow_action) - 0.9)  # Very dangerous
        
        # Reward BALANCED control (both controls used appropriately)
        if 0.05 < rod_magnitude < 0.3 and 0.1 < flow_magnitude < 0.5:
            reward += 3.0  # Using both controls in reasonable ranges
        
        # =================================================================
        # 5. STABILITY METRICS - Professional Operation
        # =================================================================
        
        # Penalize oscillations
        if power_rate_magnitude > 0.05:
            reward -= 3.0 * power_rate_magnitude
        
        if temp_rate_magnitude > 10.0:
            reward -= 2.0 * (temp_rate_magnitude / 10.0)
        
        # Bonus for achieving steady-state
        if power_rate_magnitude < 0.01 and temp_rate_magnitude < 1.0:
            reward += 5.0  # Steady state bonus
        
        # =================================================================
        # 6. BASELINE SURVIVAL
        # =================================================================
        reward += 1.0  # Small bonus for staying alive

        # =================================================================
        # 7. REWARD NORMALIZATION - Balanced for Learning Speed & Stability
        # =================================================================
        # Sweet spot: 0.15 scaling (between 0.1 too slow and 0.2 unstable)
        reward = reward * 0.15  # CHANGED: from 0.1 to 0.15

        # Clip to reasonable bounds
        reward = np.clip(reward, -12.0, 12.0)  # CHANGED: from ±15 to ±12

        return reward
    
    def render(self):
        """Optional: visualization hook"""
        pass
    
    def get_episode_statistics(self):
        """Return performance metrics for monitoring"""
        return {
            'max_temp': self.episode_max_temp,
            'violation_time': self.cumulative_violation_time,
            'episode_length': self.t
        }
