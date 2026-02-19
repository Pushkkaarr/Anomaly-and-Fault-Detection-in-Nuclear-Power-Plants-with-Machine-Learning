"""
API Response Formatting

Standardized response structures for consistent API communication
"""

from typing import Dict, Any, List, Optional
from dataclasses import asdict, dataclass
import json


@dataclass
class APIResponse:
    """Standard API response structure"""
    success: bool
    status_code: int
    data: Any = None
    message: str = ""
    errors: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "success": self.success,
            "status_code": self.status_code,
            "message": self.message,
        }
        if self.data is not None:
            result["data"] = self.data
        if self.errors:
            result["errors"] = self.errors
        return result


class ResponseFormatter:
    """Helper class for creating consistent API responses"""
    
    @staticmethod
    def success(
        data: Any = None,
        message: str = "Success",
        status_code: int = 200
    ) -> Dict[str, Any]:
        """Create success response"""
        response = APIResponse(
            success=True,
            status_code=status_code,
            data=data,
            message=message
        )
        return response.to_dict()
    
    @staticmethod
    def error(
        message: str,
        status_code: int = 400,
        errors: Optional[List[str]] = None,
        data: Any = None
    ) -> Dict[str, Any]:
        """Create error response"""
        response = APIResponse(
            success=False,
            status_code=status_code,
            message=message,
            errors=errors,
            data=data
        )
        return response.to_dict()
    
    @staticmethod
    def model_info(model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format model information response"""
        return {
            "id": model_data.get("id"),
            "name": model_data.get("name"),
            "description": model_data.get("description"),
            "reward_per_step": float(model_data.get("reward_per_step", 0)),
            "training_steps": int(model_data.get("training_steps", 0)),
            "status": model_data.get("status", "unknown"),
            "loaded": model_data.get("loaded", False)
        }
    
    @staticmethod
    def state_update(
        state: Dict[str, float],
        action: Dict[str, float],
        reward: float,
        done: bool,
        info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format state update response"""
        return {
            "state": state,
            "action": action,
            "reward": float(reward),
            "done": bool(done),
            "info": info
        }
