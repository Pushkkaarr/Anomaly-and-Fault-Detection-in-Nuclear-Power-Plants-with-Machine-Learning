"use client";

import { useState, useCallback, useEffect } from "react";
import { apiClient, getErrorMessage, isApiError } from "@/lib/api";
import { useSimulation } from "@/store/simulation";
import {
  Model,
  Scenario,
  ReactorState,
  SimulationEvent,
  Action,
} from "@/types/reactor";

interface UseApiState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

/**
 * Generic API hook for fetching data
 */
export function useApi<T>(
  fetcher: () => Promise<T>,
  autoFetch: boolean = true
) {
  const [state, setState] = useState<UseApiState<T>>({
    data: null,
    loading: autoFetch,
    error: null,
  });

  const fetch = useCallback(async () => {
    setState((prev) => ({ ...prev, loading: true, error: null }));
    try {
      const data = await fetcher();
      setState({ data, loading: false, error: null });
    } catch (error) {
      const errorMsg = getErrorMessage(error);
      setState({ data: null, loading: false, error: errorMsg });
    }
  }, []);

  useEffect(() => {
    if (autoFetch) {
      fetch();
    }
  }, [autoFetch, fetch]);

  return { ...state, refetch: fetch };
}

/**
 * Hook for managing models
 */
export function useModels() {
  const { data: models, loading, error, refetch } = useApi(() => apiClient.getModels());

  const loadModel = useCallback(async (modelId: string) => {
    try {
      const result = await apiClient.loadModel(modelId);
      return result;
    } catch (error) {
      throw new Error(getErrorMessage(error));
    }
  }, []);

  return { models: (models || []) as Model[], loading, error, loadModel, refetch };
}

/**
 * Hook for managing scenarios
 */
export function useScenarios() {
  const { data: scenarios, loading, error, refetch } = useApi(() =>
    apiClient.getScenarios()
  );

  return { scenarios: (scenarios || []) as Scenario[], loading, error, refetch };
}

/**
 * Hook for managing backend health
 */
export function useBackendHealth() {
  const [isHealthy, setIsHealthy] = useState(false);
  const [isChecking, setIsChecking] = useState(true);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        await apiClient.checkHealth();
        setIsHealthy(true);
      } catch {
        setIsHealthy(false);
      } finally {
        setIsChecking(false);
      }
    };

    // Wait a moment for port discovery to complete before first health check
    const initialDelay = setTimeout(checkHealth, 2000);
    const interval = setInterval(checkHealth, 8000);

    return () => {
      clearTimeout(initialDelay);
      clearInterval(interval);
    };
  }, []);

  return { isHealthy, isChecking };
}

/**
 * Hook for simulation control
 */
export function useSimulationControl() {
  const store = useSimulation();
  const [loading, setLoading] = useState(false);
  const [lastCriticalTempStatus, setLastCriticalTempStatus] = useState<"safe" | "critical" | null>(null);

  const startSimulation = useCallback(
    async (modelId: string, scenarioId: string) => {
      setLoading(true);
      setLastCriticalTempStatus(null); // Reset alert tracking on new simulation
      try {
        // Load model first
        await apiClient.loadModel(modelId);

        // Start simulation
        const response = await apiClient.startSimulation(modelId, scenarioId);

        store.setCurrentModel(modelId);
        store.setCurrentScenario(scenarioId);
        store.setReactorState(response.reactor_state);
        store.clearHistory();
        store.setIsRunning(true);
        store.clearEvents();

        // Add start event
        store.addEvent({
          timestamp: 0,
          type: "success",
          message: `Simulation started with ${modelId} (${scenarioId})`,
          icon: "play",
        });

        return response;
      } catch (error) {
        const errorMsg = getErrorMessage(error);
        store.setErrorMessage(errorMsg);
        store.addEvent({
          timestamp: 0,
          type: "critical",
          message: `Failed to start simulation: ${errorMsg}`,
          icon: "alert-circle",
        });
        throw error;
      } finally {
        setLoading(false);
      }
    },
    [store]
  );

  const stepSimulation = useCallback(async () => {
    try {
      const response = await apiClient.stepSimulation();

      // Validation: Check response has required data
      if (!response || !response.reactor_state) {
        throw new Error("Invalid response from stepSimulation: missing reactor_state");
      }

      store.setReactorState(response.reactor_state);
      store.addStateToHistory(response.reactor_state);
      store.setEpisodeStep(response.episode_step || store.episode_step + 1);

      // Save the AI's control action - ONLY log on significant changes (every 5 steps)
      if (response.action && response.episode_step % 5 === 0) {
        store.setLastAction(response.action);
        // Log once every 5 steps to reduce spam
        store.addEvent({
          timestamp: response.reactor_state.time,
          type: "info",
          message: `AI: Rod=${response.action.control_rod.toFixed(3)}, Flow=${response.action.coolant_flow.toFixed(3)}`,
          icon: "zap",
        });
      } else if (response.action) {
        store.setLastAction(response.action);
        // Update the action silently without logging every step
      }

      // Check for critical conditions - ONLY log on status CHANGE, not every step
      const isCritical = response.reactor_state.fuel_temp > 900;
      const currentStatus = isCritical ? "critical" : "safe";

      if (lastCriticalTempStatus !== currentStatus) {
        setLastCriticalTempStatus(currentStatus);
        if (isCritical) {
          store.addEvent({
            timestamp: response.reactor_state.time,
            type: "critical",
            message: `⚠️ Temperature critical: ${response.reactor_state.fuel_temp.toFixed(1)}K (threshold 900K)`,
            icon: "thermometer",
          });
        } else {
          store.addEvent({
            timestamp: response.reactor_state.time,
            type: "success",
            message: `✓ Temperature returned to safe zone: ${response.reactor_state.fuel_temp.toFixed(1)}K`,
            icon: "thermometer",
          });
        }
      }

      if (response.done) {
        store.setIsRunning(false);
        store.addEvent({
          timestamp: response.reactor_state.time,
          type: "info",
          message: "Simulation episode completed - retrieving metrics...",
          icon: "check",
        });

        // Get final metrics
        try {
          const summary = await apiClient.stopSimulation();
          store.setMetrics({
            total_reward: summary?.total_reward ?? 0,
            episode_steps: summary?.episode_steps ?? 0,
            episode_duration: summary?.episode_duration ?? 0,
            max_fuel_temp: summary?.max_fuel_temp ?? 0,
            max_coolant_temp: summary?.max_coolant_temp ?? 0,
            avg_pressure: summary?.avg_pressure ?? 0,
            power_change_rate: 0, // Not returned from backend
            safety_events: 0, // Not returned from backend
          });
        } catch (error) {
          console.error("Failed to get final metrics:", error);
        }
      }

      return response;
    } catch (error) {
      const errorMsg = getErrorMessage(error);
      console.error("[stepSimulation] Error:", {
        message: errorMsg,
        error: error instanceof Error ? error.stack : String(error),
      });

      // If polling is active, we still want to continue trying
      // Only stop on critical errors
      if (error instanceof Error && error.message.includes("Invalid response from stepSimulation")) {
        store.setIsRunning(false);
        store.addEvent({
          timestamp: Date.now() / 1000,
          type: "critical",
          message: `Simulation error: ${errorMsg}`,
          icon: "alert-circle",
        });
      }

      // Log the error but don't crash polling
      console.error("[stepSimulation] Error details:", { errorMsg, error });
      throw error;
    }
  }, [store, lastCriticalTempStatus]);

  const manualControl = useCallback(
    async (action: Action) => {
      try {
        const response = await apiClient.manualAction(action);

        store.setReactorState(response.reactor_state);
        store.addStateToHistory(response.reactor_state);
        store.setEpisodeStep(response.episode_step || store.episode_step + 1);

        store.addEvent({
          timestamp: response.reactor_state.time,
          type: "info",
          message: `Manual action: Rod=${action.control_rod.toFixed(2)}, Flow=${action.coolant_flow.toFixed(2)}`,
          icon: "command",
        });

        if (response.done) {
          store.setIsRunning(false);
        }

        return response;
      } catch (error) {
        const errorMsg = getErrorMessage(error);
        store.setErrorMessage(errorMsg);
        throw error;
      }
    },
    [store]
  );

  const stopSimulation = useCallback(async () => {
    try {
      const summary = await apiClient.stopSimulation();

      store.setIsRunning(false);

      // Save metrics for scenario summary
      store.setMetrics({
        total_reward: summary?.total_reward || 0,
        episode_steps: summary?.episode_steps || 0,
        episode_duration: summary?.episode_duration || 0,
        max_fuel_temp: summary?.max_fuel_temp || 0,
        max_coolant_temp: summary?.max_coolant_temp || 0,
        avg_pressure: summary?.avg_pressure || 0,
        power_change_rate: 0, // Not returned from backend
        safety_events: 0, // Not returned from backend
      });

      store.addEvent({
        timestamp: summary?.episode_duration || 0,
        type: "success",
        message: `Simulation stopped. Reward: ${(summary?.total_reward || 0).toFixed(2)}`,
        icon: "square",
      });

      return summary;
    } catch (error) {
      const errorMsg = getErrorMessage(error);
      console.error("Stop simulation error:", errorMsg);
      store.setErrorMessage(errorMsg);
      throw error;
    }
  }, [store]);

  const resetSimulation = useCallback(async () => {
    try {
      await apiClient.resetSimulation();
      store.reset();

      store.addEvent({
        timestamp: 0,
        type: "info",
        message: "Simulation reset to initial state",
        icon: "rotate-ccw",
      });
    } catch (error) {
      const errorMsg = getErrorMessage(error);
      store.setErrorMessage(errorMsg);
      throw error;
    }
  }, [store]);

  return {
    loading,
    startSimulation,
    stepSimulation,
    manualControl,
    stopSimulation,
    resetSimulation,
  };
}

/**
 * Hook for WebSocket-based real-time simulation updates
 * Much faster than HTTP polling - event-driven instead of request-based
 */
/**
 * Hook for Socket.IO-based real-time simulation updates
 * Automatically falls back to HTTP long-polling if WebSocket unavailable
 * Much faster than pure HTTP polling - event-driven instead of request-based
 */
export function useWebSocketSimulation(isEnabled: boolean) {
  const store = useSimulation();
  const [wsConnected, setWsConnected] = useState(false);
  const [wsError, setWsError] = useState<string | null>(null);
  const [lastCriticalTempStatus, setLastCriticalTempStatus] = useState<"safe" | "critical" | null>(null);

  useEffect(() => {
    if (!isEnabled || !store.is_running) {
      setWsConnected(false);
      return;
    }

    // Lazy import to avoid SSR issues
    let socket: any = null;

    const connectSocketIO = async () => {
      try {
        // Dynamically import socket.io-client
        const { io } = await import("socket.io-client");

        // Use the same port that apiClient discovered — avoids hardcoding 8000
        const apiBase = apiClient.getBaseURL(); // e.g. "http://localhost:5000/api"
        const socketURL = apiBase.replace('/api', '');

        console.log("[Socket.IO] Attempting to connect to:", socketURL);

        // Create Socket.IO connection
        socket = io(socketURL, {
          path: "/api/ws",
          transports: ["websocket", "polling"], // Try WebSocket first, fall back to long-polling
          reconnection: true,
          reconnectionDelay: 1000,
          reconnectionDelayMax: 5000,
          reconnectionAttempts: 5,
          autoConnect: true,
          forceNew: false,
        });

        // Connection established
        socket.on("connect", () => {
          console.log("[Socket.IO] ✓ Connected (transport:", socket.io.engine.transport.name + ")");
          setWsConnected(true);
          setWsError(null);
          setLastCriticalTempStatus(null);

          // Subscribe to simulation channel
          socket.emit("subscribe", {
            channel: "simulation",
          });
        });

        // Receive subscription confirmation
        socket.on("subscription_response", (data: any) => {
          console.log("[Socket.IO] Subscription confirmed:", data);
        });

        // Receive simulation state updates
        socket.on("state_update", (message: any) => {
          try {
            if (message.type === "state_update" && message.data) {
              const data = message.data;

              // Update reactor state
              if (data.reactor_state) {
                store.setReactorState(data.reactor_state);
                store.addStateToHistory(data.reactor_state);
              }

              // Update step count
              if (data.episode_step !== undefined) {
                store.setEpisodeStep(data.episode_step);
              }

              // Update AI action
              if (data.action) {
                store.setLastAction(data.action);
                // Log action every 5 steps to reduce spam
                if (data.episode_step % 5 === 0) {
                  store.addEvent({
                    timestamp: data.reactor_state?.time || Date.now() / 1000,
                    type: "info",
                    message: `AI: Rod=${data.action.control_rod.toFixed(3)}, Flow=${data.action.coolant_flow.toFixed(3)}`,
                    icon: "zap",
                  });
                }
              }

              // Check for critical temperature status changes
              if (data.reactor_state) {
                const isCritical = data.reactor_state.fuel_temp > 900;
                const currentStatus = isCritical ? "critical" : "safe";

                if (lastCriticalTempStatus !== currentStatus) {
                  setLastCriticalTempStatus(currentStatus);
                  if (isCritical) {
                    store.addEvent({
                      timestamp: data.reactor_state.time,
                      type: "critical",
                      message: `⚠️ Temperature critical: ${data.reactor_state.fuel_temp.toFixed(1)}K (threshold 900K)`,
                      icon: "thermometer",
                    });
                  } else {
                    store.addEvent({
                      timestamp: data.reactor_state.time,
                      type: "success",
                      message: `✓ Temperature safe: ${data.reactor_state.fuel_temp.toFixed(1)}K`,
                      icon: "thermometer",
                    });
                  }
                }
              }

              // Check if simulation ended
              if (data.done) {
                store.setIsRunning(false);
                store.addEvent({
                  timestamp: data.reactor_state?.time || Date.now() / 1000,
                  type: "info",
                  message: "Simulation episode completed",
                  icon: "check",
                });

                // Try to get metrics
                if (data.metrics) {
                  store.setMetrics(data.metrics);
                }
              }
            }
          } catch (error) {
            console.error("[Socket.IO] Failed to process state update:", {
              error,
              message: message,
            });
          }
        });

        // Handle connection errors
        socket.on("error", (error: any) => {
          console.error("[Socket.IO] Connection error:", error);
          setWsError("Socket.IO connection failed");
          setWsConnected(false);
        });

        // Handle disconnection
        socket.on("disconnect", (reason: string) => {
          console.log("[Socket.IO] Disconnected: " + reason);
          setWsConnected(false);
          if (reason === "io server disconnect") {
            // Server deliberately disconnected - reconnect
            console.log("[Socket.IO] Attempting to reconnect...");
            socket.connect();
          }
        });

        // Transport change events
        socket.io.engine.on("upgrade", (transport: any) => {
          console.log("[Socket.IO] Transport upgraded to:", transport.name);
        });
      } catch (error) {
        console.error("[Socket.IO] Failed to initialize:", error);
        setWsError("Failed to initialize Socket.IO");
        setWsConnected(false);
      }
    };

    connectSocketIO();

    return () => {
      if (socket) {
        console.log("[Socket.IO] Cleaning up connection");
        socket.disconnect();
        socket = null;
      }
    };
  }, [isEnabled, store.is_running]);

  return { wsConnected, wsError };
}

/**
 * Hook for auto-stepping simulation (DEPRECATED - use WebSocket instead)
 * Falls back to HTTP polling if WebSocket fails
 */
export function useAutoStep(isAutoStepping: boolean, interval: number = 100) {
  const { stepSimulation } = useSimulationControl();
  const store = useSimulation();
  const { wsConnected, wsError } = useWebSocketSimulation(isAutoStepping && store.is_running);

  // Only use HTTP polling if WebSocket is not available
  useEffect(() => {
    if (!isAutoStepping || !store.is_running || wsConnected) {
      console.log(`[AutoStep] Polling disabled - isAutoStepping: ${isAutoStepping}, is_running: ${store.is_running}, wsConnected: ${wsConnected}`);
      return;
    }

    console.log("[AutoStep] WebSocket unavailable, falling back to HTTP polling...");

    let stepCount = 0;
    const timer = setInterval(async () => {
      try {
        stepCount++;
        console.log(`[AutoStep] Poll attempt #${stepCount}, store running: ${store.is_running}`);

        if (!store.is_running) {
          console.log("[AutoStep] Simulation stopped, clearing polling interval");
          clearInterval(timer);
          return;
        }

        await stepSimulation();
      } catch (error) {
        console.error("[AutoStep] Failed:", error instanceof Error ? error.message : String(error));
        console.error("[AutoStep] Full error object:", error);
      }
    }, interval);

    return () => {
      console.log(`[AutoStep] Cleanup - cleared ${stepCount} attempts`);
      clearInterval(timer);
    };
  }, [isAutoStepping, interval, stepSimulation, store.is_running, wsConnected]);

  return { wsConnected, wsError };
}
