"""
Model Manager

Handles loading, caching, and inference with pre-trained SAC models
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np

import torch

from backend.utils.config import MODELS_CONFIG, NUCLEAR_ENV_PATH, SIMULATION_CONFIG
from backend.utils.logger import LoggerMixin, setup_logger
from backend.utils.errors import ModelLoadError, ModelInferenceError


class SACAgentWrapper(LoggerMixin):
    """
    Wrapper for SAC agent inference
    
    Handles model loading and provides action selection interface
    """
    
    def __init__(self, model_path: Path, state_dim: int = 8, action_dim: int = 2):
        """
        Initialize SAC agent wrapper
        
        Args:
            model_path: Path to saved model file
            state_dim: State dimension (default: 8 for nuclear env)
            action_dim: Action dimension (default: 2 for nuclear env)
            
        Raises:
            ModelLoadError: If model fails to load
        """
        self.model_path = Path(model_path)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(SIMULATION_CONFIG["device"])
        
        # Import SAC components - add path dynamically
        sys.path.insert(0, str(NUCLEAR_ENV_PATH))
        try:
            from sac_networks import SACAgent  # type: ignore
            self.SACAgent = SACAgent
        except ImportError as e:
            raise ModelLoadError(f"Failed to import SAC agent: {e}")
        
        self.agent = None
        self._load_model()
    
    def _load_model(self) -> None:
        """
        Load pre-trained model from checkpoint
        
        Raises:
            ModelLoadError: If checkpoint doesn't exist or fails to load
        """
        if not self.model_path.exists():
            raise ModelLoadError(
                f"Model checkpoint not found: {self.model_path}"
            )
        
        try:
            self.logger.info(f"Loading model from: {self.model_path}")
            
            # Initialize agent
            self.agent = self.SACAgent(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                device=self.device
            )
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle both state_dict and full checkpoint formats
            if isinstance(checkpoint, dict) and 'actor_state_dict' in checkpoint:
                self.agent.actor.load_state_dict(checkpoint['actor_state_dict'])
                self.agent.critic1.load_state_dict(checkpoint['critic1_state_dict'])
                self.agent.critic2.load_state_dict(checkpoint['critic2_state_dict'])
            else:
                # Assume it's just actor state dict
                if hasattr(checkpoint, 'keys') and 'actor' in checkpoint:
                    self.agent.actor.load_state_dict(checkpoint['actor'])
                else:
                    self.agent.actor.load_state_dict(checkpoint)
            
            # Set to evaluation mode
            self.agent.actor.eval()
            self.agent.critic1.eval()
            self.agent.critic2.eval()
            
            self.logger.info("✓ Model loaded successfully")
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {str(e)}")
    
    def select_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Select action for given state
        
        Args:
            state: Current environment state (array of shape [state_dim])
            deterministic: If True, use mean action; if False, sample
            
        Returns:
            Action array of shape [action_dim]
            
        Raises:
            ModelInferenceError: If inference fails
        """
        if self.agent is None:
            raise ModelInferenceError("Agent not loaded")
        
        try:
            with torch.no_grad():
                action = self.agent.select_action(
                    state.astype(np.float32),
                    deterministic=deterministic
                )
            return action
        except Exception as e:
            raise ModelInferenceError(f"Action selection failed: {str(e)}")
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata"""
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "device": str(self.device),
            "model_path": str(self.model_path)
        }


class ModelManager(LoggerMixin):
    """
    Manages loading, caching, and switching between models
    
    Implements singleton pattern for efficient resource usage
    """
    
    _instance = None
    _models: Dict[str, SACAgentWrapper] = {}
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize model manager"""
        if self._initialized:
            return
        
        self._initialized = True
        self._current_model_id: Optional[str] = None
        self.logger.info("Model Manager initialized")
    
    def load_model(self, model_id: str) -> SACAgentWrapper:
        """
        Load model by ID (cached)
        
        Args:
            model_id: Model identifier from MODELS_CONFIG
            
        Returns:
            Loaded SACAgentWrapper instance
            
        Raises:
            ModelLoadError: If model_id invalid or load fails
        """
        if model_id not in MODELS_CONFIG:
            raise ModelLoadError(f"Unknown model ID: {model_id}")
        
        # Return cached model if available
        if model_id in self._models:
            self.logger.info(f"Using cached model: {model_id}")
            return self._models[model_id]
        
        # Load new model
        config = MODELS_CONFIG[model_id]
        self.logger.info(f"Loading model: {config['name']}")
        
        try:
            model = SACAgentWrapper(
                model_path=config["path"],
                state_dim=8,
                action_dim=2
            )
            self._models[model_id] = model
            self._current_model_id = model_id
            self.logger.info(f"✓ Model loaded: {model_id}")
            return model
        
        except ModelLoadError as e:
            self.logger.error(f"Failed to load model {model_id}: {e}")
            raise
    
    def get_model(self, model_id: str) -> SACAgentWrapper:
        """
        Get model instance (loading if necessary)
        
        Args:
            model_id: Model identifier
            
        Returns:
            SACAgentWrapper instance
            
        Raises:
            ModelLoadError: If model cannot be loaded
        """
        if model_id in self._models:
            return self._models[model_id]
        return self.load_model(model_id)
    
    def list_available_models(self) -> list:
        """
        List all available models
        
        Returns:
            List of model configurations
        """
        models = []
        for model_id, config in MODELS_CONFIG.items():
            model_info = {
                "id": config["id"],
                "name": config["name"],
                "description": config["description"],
                "reward_per_step": config["reward_per_step"],
                "training_steps": config["training_steps"],
                "status": config["status"],
                "loaded": model_id in self._models
            }
            models.append(model_info)
        return models
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get detailed model information
        
        Args:
            model_id: Model identifier
            
        Returns:
            Dictionary with model details
            
        Raises:
            ModelLoadError: If model_id invalid
        """
        if model_id not in MODELS_CONFIG:
            raise ModelLoadError(f"Unknown model ID: {model_id}")
        
        config = MODELS_CONFIG[model_id]
        return {
            "id": config["id"],
            "name": config["name"],
            "description": config["description"],
            "reward_per_step": config["reward_per_step"],
            "training_steps": config["training_steps"],
            "status": config["status"],
            "loaded": model_id in self._models,
            "model_path": str(config["path"])
        }
    
    def clear_cache(self) -> None:
        """Clear all cached models to free memory"""
        self._models.clear()
        self._current_model_id = None
        self.logger.info("Model cache cleared")
    
    @property
    def current_model_id(self) -> Optional[str]:
        """Get currently active model ID"""
        return self._current_model_id
