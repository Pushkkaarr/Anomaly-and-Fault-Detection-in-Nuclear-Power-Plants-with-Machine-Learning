"""
Nuclear Reactor Physics Environment

Wrapper around the gymnasium-based nuclear reactor environment
for API integration
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import numpy as np

from backend.utils.config import NUCLEAR_ENV_PATH, SIMULATION_CONFIG
from backend.utils.logger import LoggerMixin, setup_logger
from backend.utils.errors import EnvironmentError, ScenarioError


class ReactorEnvironmentWrapper(LoggerMixin):
    """
    Wrapper around NuclearReactorEnv for API usage
    
    Handles environment initialization, stepping, and state management
    """
    
    # State dimension and component names
    STATE_DIM = 8
    STATE_COMPONENTS = [
        "power",
        "precursors",
        "fuel_temp",
        "coolant_temp",
        "pressure",
        "power_rate",
        "temp_rate",
        "time"
    ]
    
    ACTION_DIM = 2
    ACTION_COMPONENTS = [
        "control_rod",
        "coolant_flow"
    ]
    
    def __init__(self, reward_shaping: str = "advanced"):
        """
        Initialize reactor environment
        
        Args:
            reward_shaping: Type of reward shaping ('basic' or 'advanced')
            
        Raises:
            EnvironmentError: If environment fails to initialize
        """
        self.reward_shaping = reward_shaping
        self.env = None
        self._episode_data = {
            "states": [],
            "actions": [],
            "rewards": [],
            "times": []
        }
        
        self._initialize_environment()
    
    def _initialize_environment(self) -> None:
        """
        Initialize gymnasium environment
        
        Raises:
            EnvironmentError: If initialization fails
        """
        try:
            # Add nuclear env path to sys.path
            sys.path.insert(0, str(NUCLEAR_ENV_PATH))
            
            from nuclear_env_optimized import NuclearReactorEnv  # type: ignore
            
            self.logger.info("Initializing Nuclear Reactor Environment")
            self.env = NuclearReactorEnv(reward_shaping=self.reward_shaping)
            self.logger.info("✓ Environment initialized")
            
        except ImportError as e:
            raise EnvironmentError(f"Failed to import environment: {e}")
        except Exception as e:
            raise EnvironmentError(f"Failed to initialize environment: {e}")
    
    def reset(self) -> Dict[str, Any]:
        """
        Reset environment to initial state
        
        Returns:
            Dictionary containing initial state components
            
        Raises:
            EnvironmentError: If reset fails
        """
        try:
            obs, info = self.env.reset()
            self._episode_data = {
                "states": [obs.tolist()],
                "actions": [],
                "rewards": [],
                "times": [0.0]
            }
            
            return self._format_state(obs)
        
        except Exception as e:
            raise EnvironmentError(f"Environment reset failed: {e}")
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment
        
        Args:
            action: Control action [rod_position, flow_rate]
            
        Returns:
            Tuple of (state_dict, reward, done, info)
            
        Raises:
            EnvironmentError: If step fails
        """
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Record episode data
            self._episode_data["states"].append(obs.tolist())
            self._episode_data["actions"].append(action.tolist() if hasattr(action, 'tolist') else list(action))
            self._episode_data["rewards"].append(float(reward))
            self._episode_data["times"].append(float(self.env.t))
            
            state_dict = self._format_state(obs)
            
            return state_dict, float(reward), done, info
        
        except Exception as e:
            raise EnvironmentError(f"Environment step failed: {e}")
    
    def apply_scenario(self, scenario_type: str, trigger_time: float = 5.0) -> None:
        """
        Apply scenario disturbance to environment
        
        Args:
            scenario_type: Type of scenario ('normal', 'lofa', 'rod_stuck', etc.)
            trigger_time: Time at which to trigger the disturbance
            
        Raises:
            ScenarioError: If scenario application fails
        """
        try:
            if scenario_type == "normal":
                # No disturbance
                pass
            
            elif scenario_type == "lofa":
                # Loss of Flow Accident - reduce coolant flow
                self.logger.info(f"Applying LOFA scenario at t={trigger_time}s")
                self.env.scenario_trigger_time = trigger_time
                self.env.scenario_type = "lofa"
            
            elif scenario_type == "rod_stuck":
                # Control rod stuck scenario
                self.logger.info(f"Applying rod stuck scenario at t={trigger_time}s")
                self.env.scenario_trigger_time = trigger_time
                self.env.scenario_type = "rod_stuck"
            
            elif scenario_type == "power_ramp":
                # Power demand ramp
                self.logger.info(f"Applying power ramp scenario at t={trigger_time}s")
                self.env.scenario_trigger_time = trigger_time
                self.env.scenario_type = "power_ramp"
            
            else:
                raise ScenarioError(f"Unknown scenario type: {scenario_type}")
        
        except ScenarioError:
            raise
        except Exception as e:
            raise ScenarioError(f"Failed to apply scenario: {e}")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current environment state
        
        Returns:
            Dictionary with current state components
            
        Raises:
            EnvironmentError: If state retrieval fails
        """
        try:
            if self.env is None:
                raise EnvironmentError("Environment not initialized")
            
            # Get raw state from environment
            state = self.env.state if hasattr(self.env, 'state') else self.env._get_state()
            return self._format_state(state)
        
        except Exception as e:
            raise EnvironmentError(f"Failed to get state: {e}")
    
    def _format_state(self, obs: np.ndarray) -> Dict[str, float]:
        """
        Format observation array as dictionary
        
        Args:
            obs: Observation array from environment
            
        Returns:
            Dictionary mapping component names to values
        """
        if isinstance(obs, np.ndarray):
            obs = obs.astype(float)
        
        return {
            name: float(obs[i]) if i < len(obs) else 0.0
            for i, name in enumerate(self.STATE_COMPONENTS)
        }
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """
        Get summary of current episode
        
        Returns:
            Dictionary with episode statistics
        """
        if not self._episode_data["rewards"]:
            return {
                "total_steps": 0,
                "total_reward": 0.0,
                "avg_reward_per_step": 0.0,
                "duration": 0.0,
                "max_fuel_temp": 0.0,
                "max_coolant_temp": 0.0
            }
        
        # Extract temperature data
        fuel_temps = [state[2] for state in self._episode_data["states"]]
        coolant_temps = [state[3] for state in self._episode_data["states"]]
        
        return {
            "total_steps": len(self._episode_data["rewards"]),
            "total_reward": float(np.sum(self._episode_data["rewards"])),
            "avg_reward_per_step": float(np.mean(self._episode_data["rewards"])) if self._episode_data["rewards"] else 0.0,
            "duration": float(self._episode_data["times"][-1]) if self._episode_data["times"] else 0.0,
            "max_fuel_temp": float(np.max(fuel_temps)) if fuel_temps else 0.0,
            "max_coolant_temp": float(np.max(coolant_temps)) if coolant_temps else 0.0,
            "min_fuel_temp": float(np.min(fuel_temps)) if fuel_temps else 0.0,
            "min_coolant_temp": float(np.min(coolant_temps)) if coolant_temps else 0.0
        }
    
    def get_observation_space_info(self) -> Dict[str, Any]:
        """Get information about observation space"""
        if self.env is None:
            return {}
        
        obs_space = self.env.observation_space
        return {
            "shape": list(obs_space.shape) if hasattr(obs_space, 'shape') else [],
            "low": obs_space.low.tolist() if hasattr(obs_space, 'low') else [],
            "high": obs_space.high.tolist() if hasattr(obs_space, 'high') else [],
            "dtype": str(obs_space.dtype) if hasattr(obs_space, 'dtype') else ""
        }
    
    def get_action_space_info(self) -> Dict[str, Any]:
        """Get information about action space"""
        if self.env is None:
            return {}
        
        action_space = self.env.action_space
        return {
            "shape": list(action_space.shape) if hasattr(action_space, 'shape') else [],
            "low": action_space.low.tolist() if hasattr(action_space, 'low') else [],
            "high": action_space.high.tolist() if hasattr(action_space, 'high') else [],
            "dtype": str(action_space.dtype) if hasattr(action_space, 'dtype') else ""
        }
