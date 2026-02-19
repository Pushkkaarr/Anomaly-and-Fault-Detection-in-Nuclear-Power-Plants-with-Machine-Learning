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

  // Fast auto-stepping: 30ms for real-time response
  useAutoStep(isAutoStepping && store.is_running, 30);

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
    <div className="min-h-screen bg-linear-to-b from-gray-900 to-gray-800">
      {/* Professional Header */}
      <header className="border-b border-gray-700 bg-gray-900 shadow-lg">
        <div className="mx-auto max-w-full px-6 py-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-linear-to-br from-blue-500 to-cyan-500">
                <Zap className="h-7 w-7 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">
                  Nuclear Reactor Control System
                </h1>
                <p className="text-sm text-gray-400">
                  AI-Powered Safety Management | Real-Time Monitoring
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className={`h-3 w-3 rounded-full ${isHealthy ? "bg-green-500" : "bg-red-500"} animate-pulse`} />
              <span className="text-sm font-semibold text-gray-300">
                {isHealthy ? "System Online" : "Offline"}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="p-6">
        {store.error_message && (
          <Alert
            type="error"
            title="Error"
            className="mb-6"
            onClose={() => store.setErrorMessage(null)}
          >
            <p className="text-sm">{store.error_message}</p>
          </Alert>
        )}

        {/* TOP SECTION: Safety Parameters (Real-time) */}
        <section className="mb-8">
          <div className="mb-4 flex items-center gap-3">
            <div className="h-1 w-6 bg-linear-to-r from-blue-500 to-cyan-500 rounded" />
            <h2 className="text-xl font-bold text-white">
              🔴 Safety Parameters & Real-Time Monitoring
            </h2>
          </div>
          <SafetyParameters
            state={store.reactor_state}
            lastActionTime={lastUpdateTime}
          />
        </section>

        {/* MIDDLE SECTION: Control & Visualization Grid */}
        <div className="grid grid-cols-3 gap-6 mb-8">
          {/* Left Panel: Setup & Configuration */}
          <div className="space-y-4">
            <div>
              <div className="mb-3 flex items-center gap-2">
                <div className="h-1 w-4 bg-purple-500 rounded" />
                <h3 className="text-lg font-bold text-white">Configuration</h3>
              </div>
              {modelsError && (
                <Alert type="error" title="Models Error">
                  <p className="text-xs">{modelsError}</p>
                </Alert>
              )}

              {scenariosError && (
                <Alert type="error" title="Scenarios Error">
                  <p className="text-xs">{scenariosError}</p>
                </Alert>
              )}

              {modelsLoading || scenariosLoading ? (
                <div className="flex items-center justify-center rounded-lg border border-gray-700 bg-gray-800 p-8">
                  <Spinner />
                </div>
              ) : (
                <>
                  <div className="space-y-4">
                    <ModelSelector
                      models={models}
                      selectedModel={selectedModel}
                      onSelect={setSelectedModel}
                      isLoading={modelsLoading}
                    />

                    <ScenarioSelector
                      scenarios={scenarios}
                      selectedScenario={selectedScenario}
                      onSelect={setSelectedScenario}
                      isLoading={scenariosLoading}
                    />

                    <div className="rounded-lg border border-gray-700 bg-gray-800 p-4">
                      <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
                        💡 Model Information
                      </p>
                      {selectedModel && models.length > 0 ? (
                        <div className="space-y-2 text-sm text-gray-300">
                          <p>
                            <span className="font-semibold">Training Steps:</span>{" "}
                            {models.find((m) => m.id === selectedModel)?.training_steps.toLocaleString()}
                          </p>
                          <p>
                            <span className="font-semibold">Avg Reward:</span>{" "}
                            {models.find((m) => m.id === selectedModel)?.reward_per_step.toFixed(1)}/step
                          </p>
                          <p className="text-xs text-gray-400 mt-3 italic">
                            {models.find((m) => m.id === selectedModel)?.description}
                          </p>
                        </div>
                      ) : (
                        <p className="text-xs text-gray-500">Select a model to view details</p>
                      )}
                    </div>
                  </div>
                </>
              )}
            </div>

            {/* Control Buttons */}
            <div>
              <div className="mb-3 flex items-center gap-2">
                <div className="h-1 w-4 bg-green-500 rounded" />
                <h3 className="text-lg font-bold text-white">Simulation Control</h3>
              </div>
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

            {/* Status Box */}
            <div className="rounded-lg border border-gray-700 bg-gray-800 p-4">
              <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
                📊 Status
              </p>
              <SimulationStatus
                isRunning={store.is_running}
                currentModel={store.current_model}
                currentScenario={store.current_scenario}
                episodeStep={store.episode_step}
                totalReward={store.metrics?.total_reward}
              />
            </div>
          </div>

          {/* Center Panel: AI Control Actions */}
          <div className="space-y-4">
            <div>
              <div className="mb-3 flex items-center gap-2">
                <div className="h-1 w-4 bg-yellow-500 rounded" />
                <h3 className="text-lg font-bold text-white">AI Decision Making</h3>
              </div>
              {store.is_running && (
                <AIActionDisplay
                  action={store.last_action}
                  isAutomatic={true}
                  modelName={
                    models.find((m) => m.id === store.current_model)?.name ||
                    "AI Model"
                  }
                  currentStep={store.episode_step}
                  maxSteps={200}
                />
              )}
              {!store.is_running && store.metrics && store.current_scenario && (
                <ScenarioSummary
                  scenario={store.current_scenario}
                  metrics={store.metrics}
                  fuelTempMax={store.metrics.max_fuel_temp}
                  fuelTempCurrent={store.reactor_state?.fuel_temp || 0}
                  episodeSteps={store.episode_step}
                />
              )}
            </div>

            {/* Manual Control */}
            {store.is_running && (
              <div>
                <div className="mb-3 flex items-center gap-2">
                  <div className="h-1 w-4 bg-orange-500 rounded" />
                  <h3 className="text-lg font-bold text-white">Manual Override</h3>
                </div>
                <ManualControl
                  onControlChange={handleManualControl}
                  isEnabled={store.is_running}
                />
              </div>
            )}
          </div>

          {/* Right Panel: Event Log */}
          <div>
            <div className="mb-3 flex items-center gap-2">
              <div className="h-1 w-4 bg-red-500 rounded" />
              <h3 className="text-lg font-bold text-white">Event Log & Alerts</h3>
            </div>
            <div className="rounded-lg border border-gray-700 bg-gray-800 overflow-hidden">
              <EventLog
                events={store.events}
                onClear={() => store.clearEvents()}
                maxHeight="max-h-96"
              />
            </div>
          </div>
        </div>

        {/* BOTTOM SECTION: Analysis & Performance */}
        {store.is_running && (
          <section className="mt-8">
            <div className="mb-4 flex items-center gap-3">
              <div className="h-1 w-6 bg-linear-to-r from-green-500 to-emerald-500 rounded" />
              <h2 className="text-xl font-bold text-white">
                📈 Performance Analytics
              </h2>
            </div>
            <div className="grid grid-cols-4 gap-4">
              <div className="rounded-lg border border-gray-700 bg-gray-800 p-4">
                <p className="text-xs font-semibold text-gray-400 uppercase">
                  Update Rate
                </p>
                <p className="mt-2 text-2xl font-bold text-green-400">
                  ~30ms
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  ✓ Real-time updates
                </p>
              </div>
              <div className="rounded-lg border border-gray-700 bg-gray-800 p-4">
                <p className="text-xs font-semibold text-gray-400 uppercase">
                  Total Reward
                </p>
                <p className="mt-2 text-2xl font-bold text-blue-400">
                  {store.metrics?.total_reward.toFixed(1) || "0.0"}
                </p>
              </div>
              <div className="rounded-lg border border-gray-700 bg-gray-800 p-4">
                <p className="text-xs font-semibold text-gray-400 uppercase">
                  Simulation Time
                </p>
                <p className="mt-2 text-2xl font-bold text-cyan-400">
                  {store.reactor_state?.time.toFixed(1) || "0"}s
                </p>
              </div>
              <div className="rounded-lg border border-gray-700 bg-gray-800 p-4">
                <p className="text-xs font-semibold text-gray-400 uppercase">
                  AI Steps Taken
                </p>
                <p className="mt-2 text-2xl font-bold text-yellow-400">
                  {store.episode_step}/200
                </p>
              </div>
            </div>
          </section>
        )}
      </main>

      {/* Professional Footer */}
      <footer className="border-t border-gray-700 bg-gray-900 px-6 py-4 mt-8">
        <div className="flex items-center justify-between text-xs text-gray-400">
          <p>
            Nuclear Reactor Control System v1.0 | AI Model: SAC (Soft Actor-Critic)
          </p>
          <p>
            {store.is_running
              ? "🔴 Simulation Running - Real-time Monitoring Active"
              : "⚪ Ready for simulation"}
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Dashboard;
