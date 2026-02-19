#!/usr/bin/env python
"""
Quick API Endpoint Test

Tests the /api/models and /api/scenarios endpoints
"""

import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.utils.config import MODELS_CONFIG, SCENARIOS


def test_models_config():
    """Test MODELS_CONFIG is loaded correctly"""
    print("\n=== MODELS_CONFIG ===")
    print(f"Found {len(MODELS_CONFIG)} model(s)")
    for model_id, config in MODELS_CONFIG.items():
        print(f"  - {model_id}: {config['name']}")
        print(f"    Path: {config.get('path', 'N/A')}")
        print(f"    Status: {config.get('status', 'N/A')}")


def test_scenarios_config():
    """Test SCENARIOS is loaded correctly"""
    print("\n=== SCENARIOS ===")
    print(f"Found {len(SCENARIOS)} scenario(s)")
    for scenario_id, config in SCENARIOS.items():
        print(f"  - {scenario_id}: {config['name']}")
        print(f"    Description: {config.get('description', 'N/A')}")


def test_model_manager():
    """Test ModelManager.list_available_models()"""
    print("\n=== ModelManager.list_available_models() ===")
    try:
        from backend.core.model_manager import ModelManager
        manager = ModelManager()
        models = manager.list_available_models()
        print(f"Found {len(models)} model(s)")
        for model in models:
            print(f"  - {model['id']}: {model['name']}")
            print(f"    Reward: {model.get('reward_per_step')}")
            print(f"    Training Steps: {model.get('training_steps')}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


def test_response_formatter():
    """Test ResponseFormatter"""
    print("\n=== ResponseFormatter ===")
    try:
        from backend.api.responses import ResponseFormatter
        response = ResponseFormatter.success(
            data={"models": [{"id": "test", "name": "Test Model"}]},
            message="Test message"
        )
        print(f"Response structure: {response}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 70)
    print("Backend API Configuration Test")
    print("=" * 70)
    
    test_models_config()
    test_scenarios_config()
    test_model_manager()
    test_response_formatter()
    
    print("\n" + "=" * 70)
    print("Test Complete")
    print("=" * 70)
