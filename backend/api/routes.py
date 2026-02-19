"""
API Routes and Endpoints

Main API endpoints for frontend communication
"""

from flask import Blueprint, request, jsonify
import numpy as np

from backend.core.model_manager import ModelManager
from backend.core.reactor_physics import ReactorEnvironmentWrapper
from backend.api.responses import ResponseFormatter
from backend.utils.config import SCENARIOS
from backend.utils.logger import setup_logger
from backend.utils.errors import (
    NuclearBackendError, ModelLoadError, EnvironmentError, 
    InvalidInputError, ScenarioError
)


api_bp = Blueprint('api', __name__, url_prefix='/api')
logger = setup_logger(__name__)

# Global state management
_simulation_state = {
    "current_model": None,
    "environment": None,
    "current_state": None,
    "episode_step": 0,
    "is_running": False
}


# ============================================================================
# HEALTH & STATUS ENDPOINTS
# ============================================================================

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify(ResponseFormatter.success(
        data={"status": "healthy"},
        message="Backend is operational"
    )), 200


@api_bp.route('/status', methods=['GET'])
def get_status():
    """Get backend status and configuration"""
    data = {
        "is_running": _simulation_state["is_running"],
        "current_model": _simulation_state["current_model"],
        "episode_step": _simulation_state["episode_step"],
        "available_models": ModelManager().list_available_models(),
        "available_scenarios": list(SCENARIOS.keys()),
        "simulation_config": {
            "timestep": 0.1,
            "max_steps": 1000,
            "default_duration": 60.0
        }
    }
    
    return jsonify(ResponseFormatter.success(
        data=data,
        message="Status retrieved successfully"
    )), 200


# ============================================================================
# MODEL ENDPOINTS
# ============================================================================

@api_bp.route('/models', methods=['GET'])
def list_models():
    """List all available models"""
    try:
        manager = ModelManager()
        models = manager.list_available_models()
        
        return jsonify(ResponseFormatter.success(
            data={"models": models},
            message=f"Found {len(models)} available models"
        )), 200
    
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return jsonify(ResponseFormatter.error(
            message="Failed to list models",
            errors=[str(e)]
        )), 500


@api_bp.route('/models/<model_id>', methods=['GET'])
def get_model_info(model_id: str):
    """Get detailed information about a specific model"""
    try:
        manager = ModelManager()
        model_info = manager.get_model_info(model_id)
        
        return jsonify(ResponseFormatter.success(
            data=ResponseFormatter.model_info(model_info),
            message=f"Model information retrieved"
        )), 200
    
    except ModelLoadError as e:
        return jsonify(ResponseFormatter.error(
            message=f"Model not found: {model_id}",
            status_code=404
        )), 404
    
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify(ResponseFormatter.error(
            message="Failed to retrieve model information",
            errors=[str(e)]
        )), 500


@api_bp.route('/models/<model_id>/load', methods=['POST'])
def load_model(model_id: str):
    """Load a specific model"""
    try:
        manager = ModelManager()
        model = manager.load_model(model_id)
        
        _simulation_state["current_model"] = model_id
        logger.info(f"Model loaded: {model_id}")
        
        model_info = manager.get_model_info(model_id)
        
        return jsonify(ResponseFormatter.success(
            data=ResponseFormatter.model_info(model_info),
            message=f"Model '{model_id}' loaded successfully",
            status_code=200
        )), 200
    
    except ModelLoadError as e:
        logger.error(f"Failed to load model: {e}")
        return jsonify(ResponseFormatter.error(
            message=f"Failed to load model: {str(e)}",
            errors=[str(e)],
            status_code=400
        )), 400
    
    except Exception as e:
        logger.error(f"Unexpected error loading model: {e}")
        return jsonify(ResponseFormatter.error(
            message="Unexpected error loading model",
            errors=[str(e)]
        )), 500


# ============================================================================
# SIMULATION ENDPOINTS
# ============================================================================

@api_bp.route('/scenarios', methods=['GET'])
def list_scenarios():
    """List all available scenarios"""
    scenarios_list = [
        {
            "id": scenario_id,
            "name": config["name"],
            "description": config["description"],
            "difficulty": config["difficulty"]
        }
        for scenario_id, config in SCENARIOS.items()
    ]
    
    return jsonify(ResponseFormatter.success(
        data={"scenarios": scenarios_list},
        message=f"Found {len(scenarios_list)} scenarios"
    )), 200


@api_bp.route('/simulation/reset', methods=['POST'])
def reset_simulation():
    """Reset simulation environment"""
    try:
        # Initialize environment if needed
        if _simulation_state["environment"] is None:
            _simulation_state["environment"] = ReactorEnvironmentWrapper(
                reward_shaping="advanced"
            )
        
        env = _simulation_state["environment"]
        initial_state = env.reset()
        
        _simulation_state["current_state"] = initial_state
        _simulation_state["episode_step"] = 0
        _simulation_state["is_running"] = False
        
        logger.info("Simulation reset")
        
        return jsonify(ResponseFormatter.success(
            data={
                "state": initial_state,
                "step": 0,
                "message": "Simulation reset successfully"
            },
            message="Environment reset successfully"
        )), 200
    
    except EnvironmentError as e:
        logger.error(f"Environment error: {e}")
        return jsonify(ResponseFormatter.error(
            message=f"Environment error: {str(e)}",
            errors=[str(e)]
        )), 500
    
    except Exception as e:
        logger.error(f"Error resetting simulation: {e}")
        return jsonify(ResponseFormatter.error(
            message="Failed to reset simulation",
            errors=[str(e)]
        )), 500


@api_bp.route('/simulation/start', methods=['POST'])
def start_simulation():
    """Start a new simulation run"""
    try:
        data = request.get_json() or {}
        
        model_id = data.get("model_id")
        scenario_id = data.get("scenario_id", "normal")
        
        # Validate inputs
        if not model_id:
            return jsonify(ResponseFormatter.error(
                message="model_id is required",
                status_code=400
            )), 400
        
        if scenario_id not in SCENARIOS:
            return jsonify(ResponseFormatter.error(
                message=f"Unknown scenario: {scenario_id}",
                status_code=400
            )), 400
        
        # Load model
        manager = ModelManager()
        model = manager.load_model(model_id)
        _simulation_state["current_model"] = model_id
        
        # Initialize environment if needed
        if _simulation_state["environment"] is None:
            _simulation_state["environment"] = ReactorEnvironmentWrapper()
        
        env = _simulation_state["environment"]
        
        # Apply scenario
        scenario = SCENARIOS[scenario_id]
        if scenario["disturbance_type"]:
            env.apply_scenario(
                scenario["disturbance_type"],
                trigger_time=scenario["disturbance_time"] or 5.0
            )
        
        # Reset environment
        initial_state = env.reset()
        _simulation_state["current_state"] = initial_state
        _simulation_state["episode_step"] = 0
        _simulation_state["is_running"] = True
        
        logger.info(f"Simulation started - Model: {model_id}, Scenario: {scenario_id}")
        
        return jsonify(ResponseFormatter.success(
            data={
                "state": initial_state,
                "model_id": model_id,
                "scenario_id": scenario_id,
                "step": 0
            },
            message="Simulation started successfully"
        )), 200
    
    except ModelLoadError as e:
        return jsonify(ResponseFormatter.error(
            message=f"Model error: {str(e)}",
            status_code=400
        )), 400
    
    except ScenarioError as e:
        return jsonify(ResponseFormatter.error(
            message=f"Scenario error: {str(e)}",
            status_code=400
        )), 400
    
    except Exception as e:
        logger.error(f"Error starting simulation: {e}")
        return jsonify(ResponseFormatter.error(
            message="Failed to start simulation",
            errors=[str(e)]
        )), 500


@api_bp.route('/simulation/step', methods=['POST'])
def simulation_step():
    """Execute one step of simulation with AI action"""
    try:
        if not _simulation_state["is_running"]:
            return jsonify(ResponseFormatter.error(
                message="Simulation not running",
                status_code=400
            )), 400
        
        if _simulation_state["current_model"] is None:
            return jsonify(ResponseFormatter.error(
                message="No model loaded",
                status_code=400
            )), 400
        
        # Get model and environment
        manager = ModelManager()
        model = manager.get_model(_simulation_state["current_model"])
        env = _simulation_state["environment"]
        
        if env is None:
            return jsonify(ResponseFormatter.error(
                message="Environment not initialized",
                status_code=400
            )), 400
        
        # Get current state and convert to numpy array
        current_state = _simulation_state["current_state"]
        state_array = np.array([
            current_state.get("power", 1.0),
            current_state.get("precursors", 0.0),
            current_state.get("fuel_temp", 1095.0),
            current_state.get("coolant_temp", 285.0),
            current_state.get("pressure", 10.0),
            current_state.get("power_rate", 0.0),
            current_state.get("temp_rate", 0.0),
            current_state.get("time", 0.0)
        ], dtype=np.float32)
        
        # Get action from model
        action = model.select_action(state_array, deterministic=True)
        
        # Execute step
        next_state, reward, done, info = env.step(action)
        
        _simulation_state["current_state"] = next_state
        _simulation_state["episode_step"] += 1
        
        if done:
            _simulation_state["is_running"] = False
        
        return jsonify(ResponseFormatter.success(
            data={
                "state": next_state,
                "action": {
                    "control_rod": float(action[0]),
                    "coolant_flow": float(action[1])
                },
                "reward": float(reward),
                "done": bool(done),
                "step": _simulation_state["episode_step"]
            },
            message="Simulation step executed"
        )), 200
    
    except Exception as e:
        logger.error(f"Error during simulation step: {e}")
        _simulation_state["is_running"] = False
        return jsonify(ResponseFormatter.error(
            message="Simulation step failed",
            errors=[str(e)]
        )), 500


@api_bp.route('/simulation/action', methods=['POST'])
def manual_action():
    """Execute simulation step with manual control action"""
    try:
        if not _simulation_state["is_running"]:
            return jsonify(ResponseFormatter.error(
                message="Simulation not running",
                status_code=400
            )), 400
        
        data = request.get_json() or {}
        
        # Parse action
        try:
            rod_action = float(data.get("rod_action", 0.0))
            flow_action = float(data.get("flow_action", 0.0))
            
            # Clip to valid ranges
            rod_action = np.clip(rod_action, -1.0, 1.0)
            flow_action = np.clip(flow_action, -1.0, 1.0)
            
            action = np.array([rod_action, flow_action], dtype=np.float32)
        
        except (ValueError, TypeError):
            return jsonify(ResponseFormatter.error(
                message="Invalid action values",
                status_code=400
            )), 400
        
        # Execute step
        env = _simulation_state["environment"]
        next_state, reward, done, info = env.step(action)
        
        _simulation_state["current_state"] = next_state
        _simulation_state["episode_step"] += 1
        
        if done:
            _simulation_state["is_running"] = False
        
        return jsonify(ResponseFormatter.success(
            data={
                "state": next_state,
                "action": {
                    "control_rod": float(action[0]),
                    "coolant_flow": float(action[1])
                },
                "reward": float(reward),
                "done": bool(done),
                "step": _simulation_state["episode_step"]
            },
            message="Manual action executed"
        )), 200
    
    except Exception as e:
        logger.error(f"Error executing manual action: {e}")
        return jsonify(ResponseFormatter.error(
            message="Failed to execute action",
            errors=[str(e)]
        )), 500


@api_bp.route('/simulation/state', methods=['GET'])
def get_state():
    """Get current simulation state"""
    if _simulation_state["current_state"] is None:
        return jsonify(ResponseFormatter.error(
            message="Simulation not initialized",
            status_code=400
        )), 400
    
    return jsonify(ResponseFormatter.success(
        data={
            "state": _simulation_state["current_state"],
            "step": _simulation_state["episode_step"],
            "is_running": _simulation_state["is_running"],
            "model": _simulation_state["current_model"]
        },
        message="Current state retrieved"
    )), 200


@api_bp.route('/simulation/stop', methods=['POST'])
def stop_simulation():
    """Stop the current simulation"""
    try:
        _simulation_state["is_running"] = False
        
        # Get episode summary
        env = _simulation_state["environment"]
        summary = {}
        if env is not None:
            summary = env.get_episode_summary()
        
        logger.info("Simulation stopped")
        
        return jsonify(ResponseFormatter.success(
            data={
                "summary": summary,
                "step": _simulation_state["episode_step"]
            },
            message="Simulation stopped"
        )), 200
    
    except Exception as e:
        logger.error(f"Error stopping simulation: {e}")
        return jsonify(ResponseFormatter.error(
            message="Failed to stop simulation",
            errors=[str(e)]
        )), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@api_bp.errorhandler(400)
def bad_request(error):
    """Handle bad request errors"""
    return jsonify(ResponseFormatter.error(
        message="Bad request",
        status_code=400,
        errors=[str(error)]
    )), 400


@api_bp.errorhandler(404)
def not_found(error):
    """Handle not found errors"""
    return jsonify(ResponseFormatter.error(
        message="Resource not found",
        status_code=404
    )), 404


@api_bp.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify(ResponseFormatter.error(
        message="Internal server error",
        status_code=500,
        errors=[str(error)]
    )), 500
