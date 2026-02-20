"""
WebSocket Handler using Flask-SocketIO

This module provides WebSocket support for streaming simulation states
in real-time with automatic fallback to long-polling.

Flask-SocketIO handles:
- WebSocket connections
- Automatic fallback to HTTP long-polling if WebSocket unavailable
- Message broadcasting
- Client management
- Reconnection logic
"""

from flask import request
from flask_socketio import SocketIO, emit, join_room, leave_room
import json
import logging
from typing import Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types"""
    def default(self, obj):
        try:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            else:
                return super().default(obj)
        except TypeError:
            logger.debug(f"Could not serialize type {type(obj).__name__}, returning None")
            return None


class WebSocketManager:
    """Manages real-time communication with clients via SocketIO"""
    
    def __init__(self, socketio: SocketIO):
        self.socketio = socketio
        self.clients: Dict[str, dict] = {}
        logger.info("[WebSocketManager] Initialized")
    
    def register_client(self, client_id: str) -> None:
        """Register a new client"""
        self.clients[client_id] = {
            "subscribed_to": []
        }
        logger.info(f"[WebSocketManager] Client registered: {client_id} (total: {len(self.clients)})")
    
    def unregister_client(self, client_id: str) -> None:
        """Unregister a client"""
        if client_id in self.clients:
            del self.clients[client_id]
            logger.info(f"[WebSocketManager] Client unregistered: {client_id} (remaining: {len(self.clients)})")
    
    def subscribe_client(self, client_id: str, channel: str) -> None:
        """Subscribe client to a channel"""
        if client_id in self.clients:
            if channel not in self.clients[client_id]["subscribed_to"]:
                self.clients[client_id]["subscribed_to"].append(channel)
                logger.info(f"[WebSocketManager] Client {client_id} subscribed to {channel}")
    
    def broadcast(self, message: dict, channel: str = "simulation") -> None:
        """Broadcast message to all subscribed clients on a channel"""
        if not message:
            logger.warning("[WebSocketManager] Attempted to broadcast empty message")
            return
        
        try:
            # Validate message is serializable
            if not isinstance(message, dict):
                logger.error(f"[WebSocketManager] Cannot broadcast non-dict message: {type(message)}")
                return
            
            # Prepare payload with custom JSON encoding for numpy types
            payload = {
                "type": "state_update",
                "data": message,
                "timestamp": message.get("timestamp")
            }
            
            # Test serialization
            json.dumps(payload, cls=NumpyEncoder)
            
            # Count subscribed clients
            subscribed_count = sum(1 for c in self.clients.values() if channel in c['subscribed_to'])
            
            # Emit to the channel - SocketIO handles delivery to all subscribed clients
            logger.debug(f"[WebSocketManager] Broadcasting to '{channel}' channel, subscribed clients: {subscribed_count}")
            self.socketio.emit(
                "state_update",
                payload,
                room=channel,
                namespace="/api/ws"
            )
            
        except Exception as e:
            logger.error(f"[WebSocketManager] Broadcast error: {e}", exc_info=True)


# Global instances
socketio = None
ws_manager = None


def init_socketio(app):
    """
    Initialize SocketIO for the Flask app
    
    Args:
        app: Flask application instance
        
    Returns:
        SocketIO instance
    """
    global socketio, ws_manager
    
    logger.info("[SocketIO] Initializing SocketIO...")
    
    # Initialize SocketIO with the app
    # async_mode='threading' works with Flask's development server
    socketio = SocketIO(
        app,
        cors_allowed_origins=["http://localhost:3000", "http://localhost:8000"],
        async_mode="threading",
        ping_timeout=60,
        ping_interval=30,
        logger=True,
        engineio_logger=True,
        json=NumpyEncoder()
    )
    
    # Initialize manager
    ws_manager = WebSocketManager(socketio)
    
    # Register SocketIO event handlers on /api/ws namespace
    @socketio.on("connect", namespace="/api/ws")
    def handle_connect():
        """Handle client connection"""
        client_id = request.sid
        logger.info(f"[SocketIO] Client connected: {client_id}")
        logger.info(f"[SocketIO] Transport: {request.environ.get('HTTP_UPGRADE', 'http-polling')}")
        ws_manager.register_client(client_id)
        
        # Emit connection confirmation
        emit("connection_response", {
            "status": "connected",
            "message": "Connected to simulation server",
            "client_id": client_id
        })
    
    @socketio.on("disconnect", namespace="/api/ws")
    def handle_disconnect():
        """Handle client disconnection"""
        client_id = request.sid
        logger.info(f"[SocketIO] Client disconnected: {client_id}")
        ws_manager.unregister_client(client_id)
    
    @socketio.on("subscribe", namespace="/api/ws")
    def handle_subscribe(data):
        """Handle subscription to a channel"""
        client_id = request.sid
        channel = data.get("channel", "simulation")
        
        logger.info(f"[SocketIO] Client {client_id} subscribing to {channel}")
        
        # Subscribe client to room (SocketIO's way of grouping subscriptions)
        join_room(channel, namespace="/api/ws")
        ws_manager.subscribe_client(client_id, channel)
        
        # Emit confirmation
        emit("subscription_response", {
            "status": "subscribed",
            "channel": channel,
            "message": f"Successfully subscribed to {channel}"
        })
    
    @socketio.on("unsubscribe", namespace="/api/ws")
    def handle_unsubscribe(data):
        """Handle unsubscription from a channel"""
        client_id = request.sid
        channel = data.get("channel", "simulation")
        
        logger.info(f"[SocketIO] Client {client_id} unsubscribing from {channel}")
        
        # Unsubscribe client from room
        leave_room(channel, namespace="/api/ws")
        if client_id in ws_manager.clients:
            if channel in ws_manager.clients[client_id]["subscribed_to"]:
                ws_manager.clients[client_id]["subscribed_to"].remove(channel)
        
        # Emit confirmation
        emit("unsubscription_response", {
            "status": "unsubscribed",
            "channel": channel
        })
    
    @socketio.on_error_default
    def default_error_handler(e):
        """Handle errors"""
        logger.error(f"[SocketIO] Error: {e}", exc_info=True)
    
    logger.info("[SocketIO] SocketIO initialized successfully on /api/ws")
    
    return socketio


def broadcast_simulation_state(state: dict) -> None:
    """
    Broadcast current simulation state to all subscribed clients
    Called from simulation loop
    
    Usage:
    from backend.api.websocket_handler import broadcast_simulation_state
    
    # In your simulation loop:
    broadcast_simulation_state({
        "reactor_state": {...},
        "episode_step": 42,
        "action": {...},
        "done": False
    })
    """
    if ws_manager:
        ws_manager.broadcast(state, channel="simulation")
    else:
        logger.warning("[broadcast_simulation_state] WebSocketManager not initialized")


# Export for external use
__all__ = [
    "init_socketio",
    "broadcast_simulation_state",
    "socketio",
    "ws_manager",
    "NumpyEncoder",
]
