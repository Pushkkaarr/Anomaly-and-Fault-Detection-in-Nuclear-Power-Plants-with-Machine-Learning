"use client";

import React, { useState, useEffect } from "react";
import { Zap } from "lucide-react";
import { useSimulation } from "@/store/simulation";
import {
  useModels,
  useScenarios,
  useSimulationControl,
  useAutoStep,
  useBackendHealth,
} from "@/hooks/useAPI";
import { Action } from "@/types/reactor";

// Components
import { Alert, Spinner } from "@/components/ui";
import {
  ModelSelector,
  ScenarioSelector,
  ControlButtons,
  ManualControl,
  SimulationStatus,
} from "@/components/Controls";
import AIActionDisplay from "@/components/AIActionDisplay";
import ScenarioSummary from "@/components/ScenarioSummary";
import SafetyParameters from "@/components/SafetyParameters";
import { ReactorVisualization } from "@/components/ReactorVisualization";
import { EventLog } from "@/components/Metrics";

export const Dashboard: React.FC = () => {
  const store = useSimulation();
  const { isHealthy, isChecking } = useBackendHealth();

  // API Hooks
  const { models, loading: modelsLoading, error: modelsError } = useModels();
  const { scenarios, loading: scenariosLoading, error: scenariosError } =
    useScenarios();
  const {
    loading: controlLoading,
    startSimulation,
    stepSimulation,
    manualControl,
    stopSimulation,
    resetSimulation,
  } = useSimulationControl();

  // Local state
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [selectedScenario, setSelectedScenario] = useState<string | null>(null);
  const [isAutoStepping, setIsAutoStepping] = useState(false);
  const [lastUpdateTime, setLastUpdateTime] = useState(Date.now());

  // WebSocket real-time updates: automatic when simulation running
  const { wsConnected, wsError } = useAutoStep(isAutoStepping && store.is_running, 30);

  // Track update frequency
  useEffect(() => {
    if (store.is_running) {
      setLastUpdateTime(Date.now());
    }
  }, [store.reactor_state, store.episode_step]);

  const handleStart = async () => {
    if (!selectedModel || !selectedScenario) return;
    try {
      await startSimulation(selectedModel, selectedScenario);
      setIsAutoStepping(true);
    } catch (error) {
      console.error("Failed to start simulation:", error);
    }
  };

  const handleStop = async () => {
    setIsAutoStepping(false);
    try {
      await stopSimulation();
    } catch (error) {
      console.error("Failed to stop simulation:", error);
    }
  };

  const handleReset = async () => {
    setIsAutoStepping(false);
    try {
      await resetSimulation();
    } catch (error) {
      console.error("Failed to reset simulation:", error);
    }
  };

  const handleManualControl = async (action: Action) => {
    try {
      await manualControl(action);
    } catch (error) {
      console.error("Failed to apply manual control:", error);
    }
  };

  const canStart = !!selectedModel && !!selectedScenario && !store.is_running;

  // Backend status
  if (isChecking) {
    return (
      <div className="flex h-screen items-center justify-center bg-gray-50">
        <div className="flex flex-col items-center gap-4">
          <Spinner size="lg" />
          <p className="text-gray-600">Connecting to backend...</p>
        </div>
      </div>
    );
  }

  if (!isHealthy) {
    return (
      <div className="flex h-screen items-center justify-center bg-gray-50 p-4">
        <Alert
          type="error"
          title="Backend Unavailable"
          className="max-w-md"
        >
          <p>
            Could not connect to the backend server at http://localhost:8000.
            Please ensure the Flask backend is running.
          </p>
        </Alert>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-linear-to-br from-gray-950 via-gray-900 to-gray-950 text-white">
      {/* ═══════════════════════════════════════════════════════════════
          SECTION 1: CRITICAL STATUS BANNER - What's happening RIGHT NOW
          ═══════════════════════════════════════════════════════════════ */}
      <div className="border-b border-gray-700 bg-gray-900/80 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-6 py-4">
          <div className="flex items-center justify-between">
            {/* Status Light & Title */}
            <div className="flex items-center gap-4">
              <div className={`h-4 w-4 rounded-full ${
                store.is_running ? "bg-yellow-400 animate-pulse" : "bg-gray-500"
              }`} />
              <h1 className="text-2xl font-bold">
                Nuclear Reactor Control
              </h1>
              {store.is_running && (
                <span className="ml-4 inline-block rounded-full bg-yellow-900/30 px-3 py-1 text-sm font-semibold text-yellow-300">
                  SIMULATION RUNNING
                </span>
              )}
            </div>

            {/* Connection Status */}
            <div className="flex items-center gap-3">
              <div className={`flex items-center gap-2 px-3 py-1 rounded-lg ${
                wsConnected
                  ? "bg-blue-900/30 border border-blue-500"
                  : "bg-orange-900/30 border border-orange-500"
              }`}>
                <div className={`h-2 w-2 rounded-full ${
                  wsConnected ? "bg-blue-400" : "bg-orange-400"
                } animate-pulse`} />
                <span className={`text-xs font-semibold ${
                  wsConnected ? "text-blue-300" : "text-orange-300"
                }`}>
                  {wsConnected ? "🟦 REALTIME" : "🟧 POLLING"}
                </span>
              </div>
              <div className={`flex items-center gap-2 px-3 py-1 rounded-lg ${
                isHealthy
                  ? "bg-green-900/30 border border-green-500"
                  : "bg-red-900/30 border border-red-500"
              }`}>
                <div className={`h-2 w-2 rounded-full ${
                  isHealthy ? "bg-green-400" : "bg-red-400"
                } animate-pulse`} />
                <span className={`text-xs font-semibold ${
                  isHealthy ? "text-green-300" : "text-red-300"
                }`}>
                  {isHealthy ? "✓ BACKEND" : "✗ OFFLINE"}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ═══════════════════════════════════════════════════════════════
          SECTION 2: CRITICAL ALERTS - Only show if something's wrong
          ═══════════════════════════════════════════════════════════════ */}
      {store.error_message && (
        <Alert
          type="error"
          title="❌ ERROR"
          className="mx-6 mt-4 mb-0"
          onClose={() => store.setErrorMessage(null)}
        >
          <p className="text-sm font-semibold">{store.error_message}</p>
        </Alert>
      )}

      {store.reactor_state && store.reactor_state.fuel_temp > 1100 && (
        <div className="mx-6 mt-4 rounded-lg border-2 border-red-500 bg-red-950/30 p-4">
          <p className="text-lg font-bold text-red-300">
            🔥 CRITICAL: Fuel temperature {store.reactor_state.fuel_temp.toFixed(0)}K (MAX 1100K)
          </p>
          <p className="text-sm text-red-200 mt-2">
            AI is not cooling the reactor properly. Check coolant flow and control rod position.
          </p>
        </div>
      )}

      {/* ═══════════════════════════════════════════════════════════════
          SECTION 3: MAIN CONTENT - 3 EQUAL COLUMNS of focused info
          ═══════════════════════════════════════════════════════════════ */}
      <main className="mx-auto max-w-7xl px-6 py-6">
        <div className="grid grid-cols-3 gap-6">
          
          {/* ═══ COLUMN 1: CONTROLS & CONFIGURATION ═══ */}
          <div className="space-y-4 flex flex-col">
            {/* Model Selection */}
            <ModelSelector
              models={models}
              selectedModel={selectedModel}
              onSelect={setSelectedModel}
              isLoading={modelsLoading}
            />

            {/* Scenario Selection */}
            <ScenarioSelector
              scenarios={scenarios}
              selectedScenario={selectedScenario}
              onSelect={setSelectedScenario}
              isLoading={scenariosLoading}
            />

            {/* Control buttons */}
            <div className="rounded-lg border border-gray-700 bg-gray-800/50 p-4 backdrop-blur-sm">
              <h2 className="text-sm font-bold text-gray-300 uppercase tracking-wider mb-3">
                ▶️ Controls
              </h2>
              <ControlButtons
                isRunning={store.is_running}
                isPaused={store.is_paused}
                isLoading={controlLoading}
                canStart={canStart}
                onStart={handleStart}
                onStop={handleStop}
                onReset={handleReset}
              />
            </div>

            {/* Simple status readout */}
            {store.is_running && (
              <div className="rounded-lg border border-gray-700 bg-linear-to-br from-gray-800/50 to-gray-900/50 p-4 backdrop-blur-sm flex-1">
                <h2 className="text-sm font-bold text-gray-300 uppercase tracking-wider mb-3">
                  📊 Progress
                </h2>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Time:</span>
                    <span className="font-mono font-bold text-green-400">
                      {store.reactor_state?.time.toFixed(1) || "0"}s
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Steps:</span>
                    <span className="font-mono font-bold text-green-400">
                      {store.episode_step} / 200
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Score:</span>
                    <span className={`font-mono font-bold ${
                      (store.metrics?.total_reward ?? 0) > 0 ? "text-green-400" : "text-gray-400"
                    }`}>
                      {(store.metrics?.total_reward ?? 0).toFixed(1)}
                    </span>
                  </div>
                  <div className="mt-3 h-2 bg-gray-700 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-linear-to-r from-blue-500 to-cyan-400"
                      style={{ width: `${(store.episode_step / 200) * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* ═══ COLUMN 2: REACTOR STATE (MAIN VISUALIZATION) ═══ */}
          <div className="space-y-4 flex flex-col">
            {/* THE BIG REACTOR VISUALIZATION */}
            {store.reactor_state && (
              <div className="rounded-lg border border-gray-700 bg-gray-800/50 p-4 backdrop-blur-sm flex-1 flex flex-col">
                <h2 className="text-sm font-bold text-gray-300 uppercase tracking-wider mb-3">
                  ⚛️ Reactor State
                </h2>
                <ReactorVisualization
                  state={store.reactor_state}
                  rodPosition={store.last_action?.control_rod ?? 0}
                  coolantFlow={store.last_action?.coolant_flow ?? 0}
                />
              </div>
            )}

            {/* AI Manual Override - only when running */}
            {store.is_running && (
              <div className="rounded-lg border border-gray-700 bg-gray-800/50 p-4 backdrop-blur-sm">
                <h2 className="text-sm font-bold text-gray-300 uppercase tracking-wider mb-3">
                  🎮 Manual Override
                </h2>
                <ManualControl
                  onControlChange={handleManualControl}
                  isEnabled={store.is_running}
                />
              </div>
            )}
          </div>

          {/* ═══ COLUMN 3: CURRENT VALUES & LATEST EVENT ═══ */}
          <div className="space-y-4 flex flex-col">
            {/* Big safety numbers */}
            {store.reactor_state && (
              <div className="rounded-lg border border-gray-700 bg-gray-800/50 p-4 backdrop-blur-sm">
                <h2 className="text-sm font-bold text-gray-300 uppercase tracking-wider mb-4">
                  🌡️ Current Readings
                </h2>
                <div className="space-y-3">
                  {/* Fuel Temperature - MOST IMPORTANT */}
                  <div className="rounded-lg bg-gray-900/50 p-3 border-l-4" style={{
                    borderColor: store.reactor_state.fuel_temp > 1100 ? "#ef4444" : 
                               store.reactor_state.fuel_temp > 950 ? "#eab308" : "#22c55e"
                  }}>
                    <p className="text-xs text-gray-400">Fuel Temperature</p>
                    <p className={`text-2xl font-bold font-mono ${
                      store.reactor_state.fuel_temp > 1100 ? "text-red-400" :
                      store.reactor_state.fuel_temp > 950 ? "text-yellow-400" : "text-green-400"
                    }`}>
                      {store.reactor_state.fuel_temp.toFixed(0)}K
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      Target: 950K | Max: 1100K
                    </p>
                  </div>

                  {/* Pressure */}
                  <div className="rounded-lg bg-gray-900/50 p-3 border-l-4" style={{
                    borderColor: (store.reactor_state.pressure < 8 || store.reactor_state.pressure > 12) ? "#eab308" : "#22c55e"
                  }}>
                    <p className="text-xs text-gray-400">System Pressure</p>
                    <p className={`text-2xl font-bold font-mono ${
                      (store.reactor_state.pressure < 8 || store.reactor_state.pressure > 12) ? "text-yellow-400" : "text-green-400"
                    }`}>
                      {store.reactor_state.pressure.toFixed(1)} bar
                    </p>
                    <p className="text-xs text-gray-500 mt-1">Safe: 8-12 bar</p>
                  </div>

                  {/* Coolant Temperature */}
                  <div className="rounded-lg bg-gray-900/50 p-3 border-l-4" style={{
                    borderColor: (store.reactor_state.coolant_temp < 280 || store.reactor_state.coolant_temp > 310) ? "#eab308" : "#22c55e"
                  }}>
                    <p className="text-xs text-gray-400">Coolant Temperature</p>
                    <p className={`text-2xl font-bold font-mono ${
                      (store.reactor_state.coolant_temp < 280 || store.reactor_state.coolant_temp > 310) ? "text-yellow-400" : "text-green-400"
                    }`}>
                      {store.reactor_state.coolant_temp.toFixed(0)}K
                    </p>
                    <p className="text-xs text-gray-500 mt-1">Safe: 280-310K</p>
                  </div>

                  {/* Power */}
                  <div className="rounded-lg bg-gray-900/50 p-3 border-l-4" style={{
                    borderColor: (store.reactor_state.power < 0.8 || store.reactor_state.power > 1.2) ? "#eab308" : "#22c55e"
                  }}>
                    <p className="text-xs text-gray-400">Reactor Power</p>
                    <p className={`text-2xl font-bold font-mono ${
                      (store.reactor_state.power < 0.8 || store.reactor_state.power > 1.2) ? "text-yellow-400" : "text-green-400"
                    }`}>
                      {store.reactor_state.power.toFixed(2)} MW
                    </p>
                    <p className="text-xs text-gray-500 mt-1">Target: 0.8-1.2 MW</p>
                  </div>
                </div>
              </div>
            )}

            {/* AI's Latest Action */}
            {store.is_running && (
              <div className="rounded-lg border border-gray-700 bg-gray-800/50 p-4 backdrop-blur-sm">
                <h2 className="text-sm font-bold text-gray-300 uppercase tracking-wider mb-3">
                  🤖 AI's Last Action
                </h2>
                {store.last_action ? (
                  <div className="space-y-2 text-sm font-mono">
                    <div>
                      <p className="text-gray-400">Control Rod:</p>
                      <p className="text-lg font-bold text-blue-300">
                        {store.last_action.control_rod.toFixed(4)}
                      </p>
                      <p className="text-xs text-gray-500">(-1 to +1)</p>
                    </div>
                    <div>
                      <p className="text-gray-400">Coolant Flow:</p>
                      <p className="text-lg font-bold text-blue-300">
                        {store.last_action.coolant_flow.toFixed(4)}
                      </p>
                      <p className="text-xs text-gray-500">(-1 to +1)</p>
                    </div>
                  </div>
                ) : (
                  <p className="text-gray-500 text-sm">No actions yet</p>
                )}
              </div>
            )}

            {/* Latest Event */}
            <div className="rounded-lg border border-gray-700 bg-gray-800/50 p-4 backdrop-blur-sm flex-1">
              <h2 className="text-sm font-bold text-gray-300 uppercase tracking-wider mb-3">
                📝 Latest Event
              </h2>
              {store.events.length > 0 ? (
                <div className="text-sm space-y-2">
                  <p className={`font-semibold leading-snug ${
                    store.events[0].type === "critical" ? "text-red-300" :
                    store.events[0].type === "warning" ? "text-yellow-300" :
                    store.events[0].type === "success" ? "text-green-300" : "text-blue-300"
                  }`}>
                    {store.events[0].message}
                  </p>
                  <p className="text-xs text-gray-500">
                    t = {store.events[0].timestamp.toFixed(1)}s
                  </p>
                </div>
              ) : (
                <p className="text-gray-500 text-sm">Waiting...</p>
              )}
            </div>
          </div>
        </div>
      </main>

      {/* FOOTER */}
      <footer className="border-t border-gray-700 bg-gray-900/50 px-6 py-3 mt-8">
        <p className="text-xs text-gray-500">
          SAC Model | 30ms | Step {store.episode_step}/200 | {store.reactor_state?.time.toFixed(1)}s
        </p>
      </footer>
    </div>
  );
};

export default Dashboard;
