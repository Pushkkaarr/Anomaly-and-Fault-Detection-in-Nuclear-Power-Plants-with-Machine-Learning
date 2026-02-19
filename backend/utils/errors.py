"""
Custom Exception Classes

Application-specific exceptions for better error handling
"""


class NuclearBackendError(Exception):
    """Base exception for all backend errors"""
    pass


class ModelLoadError(NuclearBackendError):
    """Raised when model fails to load"""
    pass


class ModelInferenceError(NuclearBackendError):
    """Raised when model inference fails"""
    pass


class EnvironmentError(NuclearBackendError):
    """Raised when environment initialization or operation fails"""
    pass


class SimulationError(NuclearBackendError):
    """Raised when simulation encounters an error"""
    pass


class InvalidInputError(NuclearBackendError):
    """Raised when input validation fails"""
    pass


class ScenarioError(NuclearBackendError):
    """Raised when scenario application fails"""
    pass
