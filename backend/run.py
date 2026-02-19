#!/usr/bin/env python
"""
Simple script to run the Flask app from the backend directory
Use this instead of python main.py
"""

import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import and run
from backend.main import create_app

if __name__ == '__main__':
    app = create_app(config_name="development")
    
    print("=" * 70)
    print("Nuclear Reactor Control Backend - Starting")
    print("=" * 70)
    print("Server running on: http://localhost:8000")
    print("API Documentation: http://localhost:8000/api/status")
    print("=" * 70)
    
    app.run(
        host='0.0.0.0',
        port=8000,
        debug=True,
        use_reloader=True
    )
