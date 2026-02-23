"""
Flask Application Factory and Entry Point

Main application initialization and configuration
"""

import sys
from pathlib import Path

from flask import Flask
from flask_cors import CORS

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.utils.config import DevelopmentConfig, ProductionConfig
from backend.utils.logger import setup_logger
from backend.api.routes import api_bp
from backend.api.websocket_handler import init_socketio


logger = setup_logger(__name__)


def create_app(config_name: str = "development") -> Flask:
    """
    Application factory function
    
    Args:
        config_name: Configuration name ('development' or 'production')
        
    Returns:
        Configured Flask application instance
    """
    # Create Flask app
    app = Flask(__name__)
    
    # Load configuration
    if config_name == "production":
        app.config.from_object(ProductionConfig)
    else:
        app.config.from_object(DevelopmentConfig)
    
    # Enable CORS for frontend communication
    CORS(app, resources={
        r"/api/*": {
            "origins": [
                "http://localhost:3000",  # Next.js dev server
                "http://localhost:5000",
                "http://localhost:5001",
                "http://localhost:8001",
                "http://localhost:8000",
            ],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "supports_credentials": True
        }
    })
    
    # Initialize SocketIO for real-time WebSocket + long-polling support
    socketio = init_socketio(app)
    logger.info("✓ SocketIO support initialized (ws://localhost:8000/api/ws with http long-polling fallback)")
    
    # Register blueprints
    app.register_blueprint(api_bp)
    
    # Set up logging
    logger.info(f"Flask app initialized with {config_name} configuration")
    
    # Root health check
    @app.route('/', methods=['GET'])
    def index():
        """Root endpoint"""
        return {
            "name": "Nuclear Reactor Control Backend",
            "version": "1.0.0",
            "status": "operational",
            "endpoints": {
                "health": "/api/health",
                "status": "/api/status",
                "models": "/api/models",
                "scenarios": "/api/scenarios",
                "simulation": "/api/simulation/*"
            }
        }, 200
    
    # Global error handler
    @app.errorhandler(Exception)
    def handle_exception(error):
        """Handle uncaught exceptions"""
        logger.error(f"Unhandled exception: {error}", exc_info=True)
        return {
            "success": False,
            "status_code": 500,
            "message": "Internal server error",
            "error": str(error)
        }, 500
    
    return app


if __name__ == '__main__':
    import os
    import socket

    app = create_app(config_name="development")

    # Import socketio from websocket_handler to run with it
    from backend.api.websocket_handler import socketio

    # Find an available port (prefer 8000, fall back to 8001)
    def is_port_free(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('0.0.0.0', port))
                return True
            except OSError:
                return False

    # Port priority: 5000 → 5001 → 8001 (8000 is often reserved by Windows Hyper-V)
    port = 5000
    for candidate in [5000, 5001, 8001, 8080]:
        if is_port_free(candidate):
            port = candidate
            break

    logger.info("=" * 70)
    logger.info("Nuclear Reactor Control Backend - Starting")
    logger.info("=" * 70)
    logger.info(f"Server running on: http://localhost:{port}")
    logger.info(f"API Documentation: http://localhost:{port}/api/status")
    logger.info("=" * 70)

    # NOTE: use_reloader=False prevents Windows socket permission errors.
    # The Flask debug reloader on Windows spawns a child process that races
    # for the same port, causing 'An attempt was made to access a socket
    # in a way forbidden by its access permissions'.
    socketio.run(
        app,
        host='0.0.0.0',
        port=port,
        debug=True,
        use_reloader=False,          # IMPORTANT: must be False on Windows
        allow_unsafe_werkzeug=True
    )
