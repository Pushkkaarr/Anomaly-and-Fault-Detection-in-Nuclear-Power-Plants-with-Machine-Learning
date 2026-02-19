"""
Application Configuration

Centralized configuration management for Flask backend
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Model paths
MODELS_BASE_PATH = PROJECT_ROOT / "python" / "SAC_enhanced_model" / "nuclear_reactor_sac" / "models"

MODELS_CONFIG = {
    "enhanced": {
        "id": "enhanced",
        "name": "Enhanced SAC Controller",
        "description": "Enhanced SAC model with advanced reward shaping",
        "path": MODELS_BASE_PATH / "SAC_Models_With_good Results" / "best_model.pth",
        "reward_per_step": 48.6,
        "training_steps": 250000,
        "status": "active"
    },
    "optimized": {
        "id": "optimized",
        "name": "Optimized SAC Controller",
        "description": "Optimized SAC model with improved performance",
        "path": MODELS_BASE_PATH / "SAC_optimized" / "best_model.pth",
        "reward_per_step": 7.3,
        "training_steps": 150000,
        "status": "active"
    }
}

# Nuclear environment paths
NUCLEAR_ENV_PATH = PROJECT_ROOT / "python" / "SAC_enhanced_model" / "nuclear_reactor_sac"

# Flask configuration
class Config:
    """Base Flask configuration"""
    DEBUG = False
    TESTING = False
    JSON_SORT_KEYS = False
    PROPAGATE_EXCEPTIONS = True


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True


# Application settings
SIMULATION_CONFIG = {
    "default_duration": 20.0,  # seconds - reduced from 60 for faster scenarios
    "timestep": 0.1,  # seconds
    "max_steps": 200,  # reduced from 1000 - 200 steps * 0.1s = 20s simulation time
    "device": "cpu",  # or 'cuda' if available
}

SCENARIOS = {
    "normal": {
        "id": "normal",
        "name": "Normal Operation",
        "description": "Steady state baseline - reactor at 100% power",
        "difficulty": "easy",
        "disturbance_time": None,
        "disturbance_type": None
    },
    "lofa": {
        "id": "lofa",
        "name": "Loss of Flow Accident",
        "description": "Coolant pump failure at t=5s, flow drops 40%",
        "difficulty": "hard",
        "disturbance_time": 5.0,
        "disturbance_type": "lofa"
    },
    "rod_malfunction": {
        "id": "rod_malfunction",
        "name": "Control Rod Stuck",
        "description": "One control rod unable to move at t=3s",
        "difficulty": "hard",
        "disturbance_time": 3.0,
        "disturbance_type": "rod_stuck"
    },
    "power_ramp": {
        "id": "power_ramp",
        "name": "Power Demand Ramp",
        "description": "Target power changes over time",
        "difficulty": "medium",
        "disturbance_time": 2.0,
        "disturbance_type": "power_ramp"
    }
}
