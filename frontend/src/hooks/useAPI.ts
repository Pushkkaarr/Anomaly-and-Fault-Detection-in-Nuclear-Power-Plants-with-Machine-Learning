"use client";

import { useState, useCallback, useEffect } from "react";
import { apiClient, getErrorMessage } from "@/lib/api";
import { useSimulation, useSimulationStore } from "@/store/simulation";
import {
  Model,
  Scenario,
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

    checkHealth();
    const interval = setInterval(checkHealth, 5000);
    return () => clearInterval(interval);
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

      // Accumulate reward for final metrics
      const newReward = (store.current_reward || 0) + (response.reward || 0);
      store.setCurrentReward(newReward);

      // Save the AI's control action — log every 5 steps to avoid spam
      if (response.action) {
        store.setLastAction(response.action);
        if (response.episode_step % 5 === 0) {
          store.addEvent({
            timestamp: response.reactor_state.time,
            type: "info",
            message: `AI: Rod=${response.action.control_rod.toFixed(3)}, Flow=${response.action.coolant_flow.toFixed(3)}`,
            icon: "zap",
          });
        }
      }

      // Alert only on status CHANGE (not every step)
      const isCritical = response.reactor_state.fuel_temp > 900;
      const currentStatus = isCritical ? "critical" : "safe";
      if (lastCriticalTempStatus !== currentStatus) {
        setLastCriticalTempStatus(currentStatus);
        store.addEvent({
          timestamp: response.reactor_state.time,
          type: isCritical ? "critical" : "success",
          message: isCritical
            ? `⚠️ Temp critical: ${response.reactor_state.fuel_temp.toFixed(1)}K (>900K threshold)`
            : `✓ Temp safe: ${response.reactor_state.fuel_temp.toFixed(1)}K`,
          icon: "thermometer",
        });
      }

      // Episode ended naturally (backend sets is_running=False before responding)
      if (response.done) {
        store.setIsRunning(false);
        store.addEvent({
          timestamp: response.reactor_state.time,
          type: "success",
          message: `Episode complete — ${response.episode_step} steps | reward ${newReward.toFixed(1)}`,
          icon: "check",
        });

        // Try stopSimulation to get full backend summary, but don't crash if backend
        // already marked the run finished (returns 400 when is_running=False)
        try {
          const summary = await apiClient.stopSimulation();
          store.setMetrics({
            total_reward: summary?.total_reward ?? newReward,
            episode_steps: summary?.episode_steps ?? response.episode_step,
            episode_duration: summary?.episode_duration ?? response.reactor_state.time,
            max_fuel_temp: summary?.max_fuel_temp ?? response.reactor_state.fuel_temp,
            max_coolant_temp: summary?.max_coolant_temp ?? response.reactor_state.coolant_temp,
            avg_pressure: summary?.avg_pressure ?? response.reactor_state.pressure,
            power_change_rate: 0,
            safety_events: 0,
          });
        } catch {
          // Backend already stopped — compute metrics from what we accumulated
          const history = store.history || [];
          const maxFuelTemp = history.reduce((m, s) => Math.max(m, s.fuel_temp), 0);
          const maxCoolant = history.reduce((m, s) => Math.max(m, s.coolant_temp), 0);
          const avgPressure = history.length
            ? history.reduce((s, st) => s + st.pressure, 0) / history.length
            : response.reactor_state.pressure;
          store.setMetrics({
            total_reward: newReward,
            episode_steps: response.episode_step,
            episode_duration: response.reactor_state.time,
            max_fuel_temp: maxFuelTemp || response.reactor_state.fuel_temp,
            max_coolant_temp: maxCoolant || response.reactor_state.coolant_temp,
            avg_pressure: avgPressure,
            power_change_rate: 0,
            safety_events: 0,
          });
        }
      }

      return response;
    } catch (error) {
      const errorMsg = getErrorMessage(error);

      // Stop on structural errors only, not on transient 400s
      if (error instanceof Error && error.message.includes("Invalid response from stepSimulation")) {
        store.setIsRunning(false);
        store.addEvent({
          timestamp: Date.now() / 1000,
          type: "critical",
          message: `Simulation error: ${errorMsg}`,
          icon: "alert-circle",
        });
      }

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

        // Backend always runs on port 8000
        const socketURL = "http://localhost:8000";

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
 * Hook for auto-stepping simulation — HTTP polling fallback when WebSocket unavailable.
 * Uses Zustand's getState() for stale-closure-safe is_running checks inside setInterval.
 */
export function useAutoStep(isAutoStepping: boolean, interval: number = 100) {
  const { stepSimulation } = useSimulationControl();
  const store = useSimulation();
  const { wsConnected, wsError } = useWebSocketSimulation(isAutoStepping && store.is_running);

  useEffect(() => {
    // Don't poll if: not enabled, not running, or WS is covering real-time updates
    if (!isAutoStepping || !store.is_running || wsConnected) return;

    let consecutiveErrors = 0;
    const MAX_ERRORS = 3; // Stop polling after 3 back-to-back 400s

    const timer = setInterval(async () => {
      // ✅ Read CURRENT store state — avoids stale React closure
      const isCurrentlyRunning = useSimulationStore.getState().is_running;

      if (!isCurrentlyRunning) {
        clearInterval(timer);
        return;
      }

      try {
        await stepSimulation();
        consecutiveErrors = 0; // Reset on success
      } catch (error) {
        consecutiveErrors++;
        if (consecutiveErrors >= MAX_ERRORS) {
          // Backend says simulation not running — stop polling
          useSimulationStore.getState().setIsRunning(false);
          clearInterval(timer);
        }
      }
    }, interval);

    return () => clearInterval(timer);
  }, [isAutoStepping, interval, stepSimulation, store.is_running, wsConnected]);

  return { wsConnected, wsError };
}
