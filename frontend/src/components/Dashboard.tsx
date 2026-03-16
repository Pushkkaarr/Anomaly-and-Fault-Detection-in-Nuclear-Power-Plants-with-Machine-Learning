"use client";

import React, { useState, useEffect } from "react";
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
import {
  ModelSelector,
  ScenarioSelector,
  ControlButtons,
  ManualControl,
  SimulationStatus,
} from "@/components/Controls";
import { ReactorVisualization } from "@/components/ReactorVisualization";
import { EventLog, MetricsSummary } from "@/components/Metrics";
import { AnalogGauge } from "@/components/AnalogGauge";
import { LiveGraphs } from "@/components/LiveGraphs";

// ─────────────────────────────────────────────────────────────────────────────
// LOADING SCREEN
// ─────────────────────────────────────────────────────────────────────────────
const LoadingScreen = () => (
  <div
    className="flex h-screen items-center justify-center"
    style={{ background: "#020812" }}
  >
    <div className="text-center">
      <div
        className="text-6xl mb-6"
        style={{
          animationName: "reactor-spin",
          animationDuration: "3s",
          animationTimingFunction: "linear",
          animationIterationCount: "infinite",
          display: "inline-block"
        }}
      >
        ⚛️
      </div>
      <p
        className="text-sm font-semibold tracking-widest uppercase"
        style={{ color: "rgba(0,212,255,0.6)" }}
      >
        Connecting to Reactor Control System...
      </p>
    </div>
  </div>
);

// ─────────────────────────────────────────────────────────────────────────────
// OFFLINE SCREEN
// ─────────────────────────────────────────────────────────────────────────────
const OfflineScreen = () => (
  <div
    className="flex h-screen items-center justify-center p-4"
    style={{ background: "#020812" }}
  >
    <div
      className="max-w-sm w-full text-center rounded-xl p-8"
      style={{
        background: "rgba(255,59,59,0.06)",
        border: "1px solid rgba(255,59,59,0.3)",
        boxShadow: "0 0 30px rgba(255,59,59,0.1)",
      }}
    >
      <div className="text-5xl mb-4">⚠️</div>
      <h2 className="text-lg font-bold mb-2" style={{ color: "#ff6b6b" }}>
        Backend Offline
      </h2>
      <p className="text-sm mb-4" style={{ color: "rgba(107,143,168,0.8)" }}>
        Cannot connect to the reactor control backend at{" "}
        <code className="font-mono text-xs" style={{ color: "#00d4ff" }}>
          http://localhost:8000
        </code>
      </p>
      <div
        className="text-xs rounded-lg p-3 text-left font-mono"
        style={{ background: "rgba(0,0,0,0.4)", color: "rgba(0,212,255,0.7)" }}
      >
        <p style={{ color: "rgba(107,143,168,0.6)" }}># Start the backend:</p>
        <p>cd backend</p>
        <p>python -m backend.main</p>
      </div>
    </div>
  </div>
);

// ─────────────────────────────────────────────────────────────────────────────
// MAIN DASHBOARD
// ─────────────────────────────────────────────────────────────────────────────
export const Dashboard: React.FC = () => {
  const store = useSimulation();
  const { isHealthy, isChecking } = useBackendHealth();

  const { models, loading: modelsLoading } = useModels();
  const { scenarios, loading: scenariosLoading } = useScenarios();
  const {
    loading: controlLoading,
    startSimulation,
    stepSimulation,
    manualControl,
    stopSimulation,
    resetSimulation,
  } = useSimulationControl();

  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [selectedScenario, setSelectedScenario] = useState<string | null>(null);
  const [isAutoStepping, setIsAutoStepping] = useState(false);

  const { wsConnected } = useAutoStep(isAutoStepping && store.is_running, 30);

  const handleStart = async () => {
    if (!selectedModel || !selectedScenario) return;
    try {
      await startSimulation(selectedModel, selectedScenario);
      setIsAutoStepping(true);
    } catch (e) {
      console.error("Start failed:", e);
    }
  };

  const handleStop = async () => {
    setIsAutoStepping(false);
    try {
      await stopSimulation();
    } catch (e) {
      console.error("Stop failed:", e);
    }
  };

  const handleReset = async () => {
    setIsAutoStepping(false);
    try {
      await resetSimulation();
    } catch (e) {
      console.error("Reset failed:", e);
    }
  };

  const handleManual = async (action: Action) => {
    try {
      await manualControl(action);
    } catch (e) {
      console.error("Manual action failed:", e);
    }
  };

  const canStart = !!selectedModel && !!selectedScenario && !store.is_running;

  const state = store.reactor_state;
  const isCritical = state ? state.fuel_temp > 1100 : false;
  const isWarning = state ? state.fuel_temp > 950 && !isCritical : false;

  if (isChecking) return <LoadingScreen />;
  if (!isHealthy) return <OfflineScreen />;

  return (
    <div
      className="min-h-screen"
      style={{ background: "linear-gradient(135deg, #020812 0%, #050f1f 50%, #020812 100%)" }}
    >
      {/* ══════════════════════════════════════════════════════════
          HEADER — Nuclear Control Room Identity Bar
          ══════════════════════════════════════════════════════════ */}
      <header
        style={{
          background: "rgba(5,15,31,0.95)",
          borderBottom: "1px solid rgba(0,212,255,0.12)",
          backdropFilter: "blur(12px)",
          position: "sticky",
          top: 0,
          zIndex: 50,
        }}
      >
        <div className="mx-auto max-w-screen-2xl px-4 py-3 flex items-center justify-between">
          {/* Left: Identity */}
          <div className="flex items-center gap-3">
            <div
              className="text-2xl"
              style={{
                filter: "drop-shadow(0 0 8px rgba(0,212,255,0.6))",
                animationName: store.is_running ? "reactor-spin" : "none",
                animationDuration: "8s",
                animationTimingFunction: "linear",
                animationIterationCount: "infinite",
              }}
            >
              ☢
            </div>
            <div>
              <h1
                className="text-sm font-bold tracking-widest uppercase"
                style={{ color: "#00d4ff", textShadow: "0 0 12px rgba(0,212,255,0.4)" }}
              >
                Nuclear Reactor Control System
              </h1>
              <p className="text-xs" style={{ color: "rgba(107,143,168,0.6)" }}>
                SAC Agent v2 · Anomaly & Fault Detection
              </p>
            </div>
          </div>

          {/* Center: Status */}
          <div className="hidden md:flex items-center gap-4">
            <SimulationStatus
              isRunning={store.is_running}
              currentModel={store.current_model}
              episodeStep={store.episode_step}
            />
            {state && (
              <div
                className="flex items-center gap-2 text-xs font-mono px-3 py-1.5 rounded-lg"
                style={{ background: "rgba(0,0,0,0.3)", border: "1px solid rgba(0,212,255,0.1)" }}
              >
                <span style={{ color: "rgba(107,143,168,0.6)" }}>t =</span>
                <span style={{ color: "#00d4ff" }}>{state.time.toFixed(1)}s</span>
                <span style={{ color: "rgba(107,143,168,0.3)" }}>·</span>
                <span style={{ color: "rgba(107,143,168,0.6)" }}>step</span>
                <span style={{ color: "#00d4ff" }}>{store.episode_step}/200</span>
              </div>
            )}
          </div>

          {/* Right: Connection indicators */}
          <div className="flex items-center gap-2">
            <div
              className="flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-lg"
              style={{
                background: wsConnected ? "rgba(0,230,118,0.08)" : "rgba(255,214,0,0.08)",
                border: `1px solid ${wsConnected ? "rgba(0,230,118,0.3)" : "rgba(255,214,0,0.3)"}`,
                color: wsConnected ? "#00e676" : "#ffd600",
              }}
            >
              <div className={wsConnected ? "led-green" : "led-yellow"} style={{ width: 6, height: 6 }} />
              {wsConnected ? "WS Live" : "Polling"}
            </div>
            <div
              className="flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-lg"
              style={{
                background: "rgba(0,230,118,0.08)",
                border: "1px solid rgba(0,230,118,0.3)",
                color: "#00e676",
              }}
            >
              <div className="led-green" style={{ width: 6, height: 6 }} />
              Backend Online
            </div>
          </div>
        </div>
      </header>

      {/* ══════════════════════════════════════════════════════════
          CRITICAL ALERT BANNER
          ══════════════════════════════════════════════════════════ */}
      {isCritical && (
        <div
          className="px-4 py-2 text-center text-sm font-bold"
          style={{
            background: "rgba(255,59,59,0.15)",
            borderBottom: "2px solid rgba(255,59,59,0.8)",
            color: "#ff6b6b",
            animationName: "critical-pulse",
            animationDuration: "1s",
            animationTimingFunction: "ease-in-out",
            animationIterationCount: "infinite",
          }}
        >
          🔥 CRITICAL: Fuel Temperature {state?.fuel_temp.toFixed(0)}K — EXCEEDS 1100K SAFETY LIMIT
        </div>
      )}

      {/* ══════════════════════════════════════════════════════════
          ERROR BAR
          ══════════════════════════════════════════════════════════ */}
      {store.error_message && (
        <div
          className="mx-4 mt-3 rounded-lg px-4 py-2 flex items-center justify-between text-sm"
          style={{ background: "rgba(255,59,59,0.1)", border: "1px solid rgba(255,59,59,0.3)", color: "#ff8080" }}
        >
          <span>⚠ {store.error_message}</span>
          <button
            onClick={() => store.setErrorMessage(null)}
            style={{ color: "rgba(255,128,128,0.6)" }}
            className="text-lg leading-none ml-4"
          >
            ×
          </button>
        </div>
      )}

      {/* ══════════════════════════════════════════════════════════
          MAIN 3-PANEL LAYOUT
          ══════════════════════════════════════════════════════════ */}
      <main className="mx-auto max-w-screen-2xl px-4 py-4">
        <div
          className="grid gap-4"
          style={{ gridTemplateColumns: "300px 1fr 340px", minHeight: "calc(100vh - 120px)" }}
        >

          {/* ══════ LEFT PANEL: CONTROLS ══════ */}
          <div className="space-y-3">
            {/* Model Selection */}
            <div className="nuclear-panel p-4">
              <ModelSelector
                models={models}
                selectedModel={selectedModel}
                onSelect={setSelectedModel}
                isLoading={modelsLoading}
                disabled={store.is_running}
              />
            </div>

            {/* Scenario Selection */}
            <div className="nuclear-panel p-4">
              <ScenarioSelector
                scenarios={scenarios}
                selectedScenario={selectedScenario}
                onSelect={setSelectedScenario}
                isLoading={scenariosLoading}
                disabled={store.is_running}
              />
            </div>

            {/* Control Buttons */}
            <div className="nuclear-panel p-4">
              <p className="section-label mb-3">Simulation Control</p>
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

            {/* Progress */}
            {store.is_running && state && (
              <div className="nuclear-panel p-4 space-y-3">
                <p className="section-label">Episode Progress</p>
                {/* Progress bar */}
                <div>
                  <div className="flex justify-between text-xs mb-1" style={{ color: "rgba(107,143,168,0.7)" }}>
                    <span>Step {store.episode_step}</span>
                    <span>{((store.episode_step / 200) * 100).toFixed(0)}%</span>
                  </div>
                  <div className="h-1.5 rounded-full overflow-hidden" style={{ background: "rgba(0,212,255,0.1)" }}>
                    <div
                      className="h-full rounded-full transition-all duration-300"
                      style={{
                        width: `${(store.episode_step / 200) * 100}%`,
                        background: "linear-gradient(90deg, #00d4ff, #00e676)",
                        boxShadow: "0 0 6px rgba(0,212,255,0.5)",
                      }}
                    />
                  </div>
                </div>
                {/* Quick stats */}
                <div className="grid grid-cols-2 gap-2">
                  <div style={{ background: "rgba(0,0,0,0.3)", border: "1px solid rgba(0,212,255,0.1)" }} className="rounded-lg p-2 text-center">
                    <p className="section-label" style={{ fontSize: "0.55rem" }}>Sim Time</p>
                    <p className="font-mono text-sm font-bold" style={{ color: "#00d4ff" }}>
                      {state.time.toFixed(1)}s
                    </p>
                  </div>
                  <div style={{ background: "rgba(0,0,0,0.3)", border: "1px solid rgba(0,230,118,0.1)" }} className="rounded-lg p-2 text-center">
                    <p className="section-label" style={{ fontSize: "0.55rem" }}>Total Score</p>
                    <p className="font-mono text-sm font-bold" style={{ color: "#00e676" }}>
                      {(store.metrics?.total_reward ?? 0).toFixed(1)}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Post-simulation metrics - "Final Report" style */}
            {!store.is_running && store.metrics && store.metrics.episode_steps > 0 && (
              <div 
                className="nuclear-panel p-4 border-2 border-[#00d4ff]/30"
                style={{
                  animationName: "enter",
                  animationDuration: "500ms",
                  animationTimingFunction: "ease-out",
                  animationFillMode: "both"
                }}
              >
                <div className="flex items-center gap-2 mb-4">
                  <div className="h-5 w-1 bg-[#00d4ff]" />
                  <p className="text-xs font-bold tracking-widest uppercase text-[#00d4ff]">
                    Final Reactor Deployment Report
                  </p>
                </div>
                <div className="space-y-4">
                  <div className="bg-black/40 rounded-lg p-3 border border-white/5">
                    <p className="text-[0.6rem] uppercase text-white/30 mb-2">Primary Objective Status</p>
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-bold text-white/90">
                        {store.metrics.total_reward > 0 ? "✓ SUCCESSFUL STABILIZATION" : "❌ SYSTEM INSTABILITY"}
                      </span>
                      <span className="text-xs font-mono text-[#00e676]">
                        SCORE: {store.metrics.total_reward.toFixed(1)}
                      </span>
                    </div>
                  </div>
                  <MetricsSummary metrics={store.metrics} isRunning={false} />
                  <button
                    onClick={() => store.reset()}
                    className="w-full py-2 text-[0.6rem] uppercase tracking-widest font-bold border border-white/10 rounded hover:bg-white/5 transition-colors"
                  >
                    Acknowledge & Dismiss
                  </button>
                </div>
              </div>
            )}

            {/* Manual Override */}
            <div className="nuclear-panel p-4">
              <p className="section-label mb-3">Manual Override</p>
              <ManualControl
                onControlChange={handleManual}
                isEnabled={store.is_running}
              />
            </div>
          </div>

          {/* ══════ CENTER PANEL: REACTOR VISUALIZATION ══════ */}
          <div className="space-y-4">
            {/* Main reactor SVG */}
            <ReactorVisualization
              state={state}
              rodPosition={store.last_action?.control_rod ?? 0}
              coolantFlow={store.last_action?.coolant_flow ?? 0}
              isRunning={store.is_running}
            />

            {/* 3 Analog Gauges */}
            {state && (
              <div
                className="nuclear-panel p-4"
              >
                <p className="section-label mb-4">Core Instrument Panel</p>
                <div className="grid grid-cols-3 gap-4">
                  <AnalogGauge
                    value={state.power * 100}
                    min={0}
                    max={150}
                    label="Power"
                    unit="%"
                    safeMin={80}
                    safeMax={120}
                    warnMax={135}
                    size={130}
                  />
                  <AnalogGauge
                    value={state.fuel_temp}
                    min={700}
                    max={1200}
                    label="Fuel Temp"
                    unit="K"
                    safeMin={800}
                    safeMax={1000}
                    warnMax={1100}
                    size={130}
                  />
                  <AnalogGauge
                    value={state.pressure}
                    min={5}
                    max={16}
                    label="Pressure"
                    unit="bar"
                    safeMin={8}
                    safeMax={12}
                    warnMax={13.5}
                    size={130}
                  />
                </div>
              </div>
            )}

            {/* AI Action Display */}
            {store.is_running && store.last_action && (
              <div className="nuclear-panel p-4 border border-[#00d4ff]/20 bg-[#00d4ff]/05">
                <div className="flex items-center justify-between mb-4">
                  <p className="section-label">🤖 AI Decision Engine (SAC)</p>
                  <span className="text-[0.6rem] font-mono text-[#00d4ff]/60 animate-pulse">
                    ANALYZING...
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-4 mb-4">
                  {/* Rod position bar */}
                  <div>
                    <div className="flex justify-between text-[0.65rem] mb-1.5 uppercase font-medium">
                      <span style={{ color: "rgba(107,143,168,0.7)" }}>Rod Delta</span>
                      <span
                        className="font-mono font-bold"
                        style={{ color: Math.abs(store.last_action.control_rod) > 0.3 ? "#ff6b6b" : "#00d4ff" }}
                      >
                        {store.last_action.control_rod > 0.005 ? "↓ INSERT" : store.last_action.control_rod < -0.005 ? "↑ WITHDRAW" : "HOLD"}
                      </span>
                    </div>
                    <div className="h-1.5 rounded-full overflow-hidden bg-white/5">
                      <div
                        className="h-full rounded-full transition-all duration-300"
                        style={{
                          width: `${Math.min(100, Math.abs(store.last_action.control_rod) * 200)}%`,
                          marginLeft: store.last_action.control_rod < 0 ? "auto" : "0",
                          background: store.last_action.control_rod > 0.005 ? "#ff5252" : store.last_action.control_rod < -0.005 ? "#40c4ff" : "#ffd600",
                          boxShadow: `0 0 8px ${store.last_action.control_rod > 0.005 ? "#ff525280" : "#40c4ff80"}`,
                        }}
                      />
                    </div>
                  </div>
                  {/* Coolant flow bar */}
                  <div>
                    <div className="flex justify-between text-[0.65rem] mb-1.5 uppercase font-medium">
                      <span style={{ color: "rgba(107,143,168,0.7)" }}>Flow Adjust</span>
                      <span
                        className="font-mono font-bold"
                        style={{ color: "#00e676" }}
                      >
                        {store.last_action.coolant_flow > 0.005 ? "+ BOOST" : store.last_action.coolant_flow < -0.005 ? "- REDUCE" : "STEADY"}
                      </span>
                    </div>
                    <div className="h-1.5 rounded-full overflow-hidden bg-white/5">
                      <div
                        className="h-full rounded-full transition-all duration-300"
                        style={{
                          width: `${Math.min(100, Math.abs(store.last_action.coolant_flow) * 200)}%`,
                          marginLeft: store.last_action.coolant_flow < 0 ? "auto" : "0",
                          background: "#00e676",
                          boxShadow: "0 0 8px rgba(0,230,118,0.4)",
                        }}
                      />
                    </div>
                  </div>
                </div>

                <div className="pt-3 border-t border-white/5">
                  <p className="text-[0.6rem] uppercase tracking-wider text-white/30 mb-1">Controller Intent</p>
                  <p className="text-xs font-medium text-[#a0d8e8] leading-snug">
                    {store.last_action.control_rod > 0.01 ? "Inserting rods to dampen fission reactivity and control rising core heat." :
                      store.last_action.control_rod < -0.01 ? "Withdrawing rods to increase thermal neutrons and boost power output." :
                        store.last_action.coolant_flow > 0.01 ? "Increasing secondary flow to maximize heat transfer and optimize cooling." :
                          store.last_action.coolant_flow < -0.01 ? "Reducing coolant throughput to maintain hydraulic pressure stability." :
                            "Maintaining steady-state reactor equilibrium and monitoring telemetry."}
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* ══════ RIGHT PANEL: DATA & EVENTS ══════ */}
          <div className="space-y-3">

            {/* Live readings */}
            {state && (
              <div className="nuclear-panel p-4">
                <p className="section-label mb-3">Live Telemetry</p>
                <div className="space-y-2">
                  {[
                    {
                      label: "Fuel Temperature",
                      value: `${state.fuel_temp.toFixed(1)} K`,
                      target: "Target: ~950K",
                      pct: Math.min(state.fuel_temp / 1150, 1),
                      color: isCritical ? "#ff3b3b" : isWarning ? "#ffd600" : "#00e676",
                    },
                    {
                      label: "Reactor Power",
                      value: `${(state.power * 100).toFixed(1)} %`,
                      target: "Target: 80–120%",
                      pct: Math.min(state.power / 1.5, 1),
                      color: (state.power < 0.8 || state.power > 1.2) ? "#ffd600" : "#00e676",
                    },
                    {
                      label: "Coolant Temperature",
                      value: `${state.coolant_temp.toFixed(1)} K`,
                      target: "Safe: 280–310K",
                      pct: Math.min(state.coolant_temp / 340, 1),
                      color: (state.coolant_temp < 280 || state.coolant_temp > 310) ? "#ffd600" : "#40c4ff",
                    },
                    {
                      label: "System Pressure",
                      value: `${state.pressure.toFixed(2)} bar`,
                      target: "Safe: 8–12 bar",
                      pct: Math.min((state.pressure - 5) / 11, 1),
                      color: (state.pressure < 8 || state.pressure > 12) ? "#ffd600" : "#00e676",
                    },
                  ].map((item, i) => (
                    <div
                      key={i}
                      className="rounded-lg px-3 py-2.5"
                      style={{
                        background: "rgba(0,0,0,0.3)",
                        borderLeft: `3px solid ${item.color}`,
                        border: `1px solid rgba(255,255,255,0.04)`,
                        borderLeftWidth: 3,
                        borderLeftColor: item.color,
                      }}
                    >
                      <div className="flex justify-between items-baseline">
                        <span className="text-xs" style={{ color: "rgba(107,143,168,0.7)" }}>{item.label}</span>
                        <span
                          className="font-mono font-bold text-sm"
                          style={{
                            color: item.color,
                            textShadow: `0 0 6px ${item.color}60`,
                            transition: "all 0.3s ease",
                          }}
                        >
                          {item.value}
                        </span>
                      </div>
                      <div className="mt-1.5 h-1 rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.05)" }}>
                        <div
                          className="h-full rounded-full transition-all duration-300"
                          style={{
                            width: `${item.pct * 100}%`,
                            background: `linear-gradient(90deg, ${item.color}aa, ${item.color})`,
                            boxShadow: `0 0 4px ${item.color}60`,
                          }}
                        />
                      </div>
                      <p className="text-xs mt-0.5" style={{ color: "rgba(107,143,168,0.4)", fontSize: "0.6rem" }}>
                        {item.target}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Live Graph */}
            <div className="nuclear-panel p-4">
              <p className="section-label mb-3">Live Data Graph</p>
              <LiveGraphs history={store.history || []} isRunning={store.is_running} />
            </div>

            {/* Neutron rates */}
            {state && (
              <div className="nuclear-panel p-4">
                <p className="section-label mb-2">Rate Indicators</p>
                <div className="grid grid-cols-2 gap-2">
                  {[
                    {
                      label: "Power Rate",
                      value: state.power_rate,
                      color: Math.abs(state.power_rate) > 0.05 ? "#ffd600" : "#00e676",
                    },
                    {
                      label: "Temp Rate",
                      value: state.temp_rate,
                      color: Math.abs(state.temp_rate) > 5 ? "#ffd600" : "#40c4ff",
                    },
                  ].map((item, i) => (
                    <div
                      key={i}
                      className="rounded-lg p-2 text-center"
                      style={{ background: "rgba(0,0,0,0.3)", border: "1px solid rgba(0,212,255,0.06)" }}
                    >
                      <p className="section-label" style={{ fontSize: "0.55rem" }}>{item.label}</p>
                      <p
                        className="font-mono text-sm font-bold mt-0.5"
                        style={{ color: item.color }}
                      >
                        {item.value > 0 ? "+" : ""}{item.value.toFixed(4)}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Event Log */}
            <div className="nuclear-panel p-4" style={{ flex: 1 }}>
              <EventLog
                events={store.events}
                onClear={() => store.clearEvents()}
              />
            </div>

          </div>
        </div>
      </main>

      {/* FOOTER */}
      <footer
        className="mt-4 py-2 px-4 text-center text-xs"
        style={{
          background: "rgba(5,15,31,0.8)",
          borderTop: "1px solid rgba(0,212,255,0.08)",
          color: "rgba(107,143,168,0.4)",
        }}
      >
        Nuclear Reactor SAC Control System · PyTorch + Flask + Next.js ·
        {state ? ` Step ${store.episode_step}/200 · t=${state.time.toFixed(1)}s` : " Standby"}
      </footer>
    </div>
  );
};

export default Dashboard;
