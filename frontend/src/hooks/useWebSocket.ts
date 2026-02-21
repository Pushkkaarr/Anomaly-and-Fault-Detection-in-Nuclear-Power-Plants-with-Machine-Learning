"use client";

import { useEffect, useRef, useCallback, useState } from "react";
import {
  WebSocketService,
  getWebSocketService,
  WebSocketMessage,
  WebSocketConfig,
} from "@/lib/websocket";

/**
 * Hook for WebSocket communication
 * Handles connection lifecycle and message handling
 */
export function useWebSocket(
  onMessage?: (message: WebSocketMessage) => void,
  config?: WebSocketConfig
) {
  const wsRef = useRef<WebSocketService | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<"disconnected" | "connecting" | "connected">("disconnected");

  // Initialize WebSocket on mount
  useEffect(() => {
    wsRef.current = getWebSocketService();

    const connect = async () => {
      try {
        setStatus("connecting");
        await wsRef.current!.connect();
        setIsConnected(true);
        setStatus("connected");
        setError(null);
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : "Connection failed";
        setError(errorMsg);
        setStatus("disconnected");
        console.error("WebSocket connection error:", errorMsg);
      }
    };

    connect();

    // Register default message handler if provided
    let unsubscribe: (() => void) | null = null;
    if (onMessage) {
      unsubscribe = wsRef.current.on("*", onMessage);
    }

    return () => {
      if (unsubscribe) {
        unsubscribe();
      }
      // Don't disconnect on unmount to keep connection persistent
    };
  }, [onMessage]);

  // Send message helper
  const send = useCallback(
    (type: string, data: any) => {
      if (!wsRef.current || !wsRef.current.isConnected()) {
        console.warn("WebSocket not connected, cannot send message:", { type, data });
        return false;
      }

      wsRef.current.send(type, data);
      return true;
    },
    []
  );

  // Subscribe to specific message type
  const on = useCallback(
    (type: string, handler: (data: any) => void) => {
      if (!wsRef.current) {
        console.warn("WebSocket not initialized");
        return () => {};
      }

      return wsRef.current.on(type, handler);
    },
    []
  );

  // Get current connection status
  const getStatus = useCallback(
    () => wsRef.current?.getStatus() ?? { connected: false, url: "", reconnectAttempts: 0 },
    []
  );

  return {
    isConnected,
    status,
    error,
    send,
    on,
    getStatus,
    ws: wsRef.current,
  };
}

/**
 * Hook specifically for simulation state updates via WebSocket
 * Handles state_update messages and updates Zustand store
 */
export function useSimulationWebSocket() {
  const { send, on, isConnected, error } = useWebSocket();

  // Subscribe to state updates
  const subscribeToStateUpdates = useCallback(
    (callback: (data: any) => void) => {
      return on("state_update", callback);
    },
    [on]
  );

  // Subscribe to action events
  const subscribeToActions = useCallback(
    (callback: (data: any) => void) => {
      return on("action", callback);
    },
    [on]
  );

  // Request simulation step via WebSocket
  const requestStep = useCallback(() => {
    return send("step", {});
  }, [send]);

  // Send manual action
  const sendManualAction = useCallback(
    (action: { control_rod: number; coolant_flow: number }) => {
      return send("manual_action", action);
    },
    [send]
  );

  return {
    isConnected,
    error,
    subscribeToStateUpdates,
    subscribeToActions,
    requestStep,
    sendManualAction,
  };
}
