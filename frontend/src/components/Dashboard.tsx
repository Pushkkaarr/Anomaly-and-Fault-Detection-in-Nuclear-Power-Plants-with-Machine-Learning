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
import { ScenarioIntelPanel } from "@/components/ScenarioIntelPanel";
import { MissionReport } from "@/components/MissionReport";
import { LiveNarrator } from "@/components/LiveNarrator";

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
        style={{ color: "rgba(0,255,136,0.6)" }}
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
      <h2 className="text-lg font-bold mb-2" style={{ color: "#ff6568" }}>
        Backend Offline
      </h2>
      <p className="text-sm mb-4" style={{ color: "rgba(107,143,168,0.8)" }}>
        Cannot connect to the reactor control backend at{" "}
        <code className="font-mono text-xs" style={{ color: "var(--brand-accent)" }}>
          http://localhost:8000
        </code>
      </p>
      <div
        className="text-xs rounded-lg p-3 text-left font-mono"
        style={{ background: "rgba(0,0,0,0.4)", color: "rgba(0,255,136,0.7)" }}
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
  const fault = store.fault_prediction;
  const isCritical = state ? state.fuel_temp > 1100 : false;
  const isWarning  = state ? state.fuel_temp > 950 && !isCritical : false;
  const faultAccent =
    fault?.risk_level === "critical"
      ? "#ff6568"
      : fault?.risk_level === "high"
        ? "#ff9f43"
        : fault?.risk_level === "medium"
          ? "#fbbf24"
          : "var(--brand-accent)";

  if (isChecking) return <LoadingScreen />;
  if (!isHealthy)  return <OfflineScreen />;

  return (
    <div
      className="min-h-screen overflow-x-hidden"
      style={{ background: "linear-gradient(135deg, #020812 0%, #050f1f 50%, #020812 100%)" }}
    >
      {/* ═══ HEADER ═══════════════════════════════════════════════════════════ */}
      <header
        style={{
          background: "rgba(5,15,31,0.95)",
          borderBottom: "1px solid rgba(0,255,136,0.12)",
          backdropFilter: "blur(12px)",
          position: "sticky",
          top: 0,
          zIndex: 50,
        }}
      >
        <div className="mx-auto flex max-w-screen-2xl flex-wrap items-center justify-between gap-3 px-4 py-3">
          {/* Left: Identity */}
          <div className="flex items-center gap-3">
            <div
              className="text-2xl"
              style={{
                filter: "drop-shadow(0 0 8px rgba(0,255,136,0.6))",
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
                style={{ color: "var(--brand-accent)", textShadow: "0 0 12px rgba(0,255,136,0.4)" }}
              >
                Nuclear Reactor Control System
              </h1>
              <p className="text-xs" style={{ color: "rgba(107,143,168,0.6)" }}>
                SAC Agent v2 · Anomaly & Fault Detection · Live Simulation
              </p>
            </div>
          </div>

          {/* Center: Status */}
          <div className="hidden lg:flex items-center gap-4">
            <SimulationStatus
              isRunning={store.is_running}
              currentModel={store.current_model}
              episodeStep={store.episode_step}
            />
            {state && (
              <div
                className="flex items-center gap-2 text-xs font-mono px-3 py-1.5 rounded-lg"
                style={{ background: "rgba(0,0,0,0.3)", border: "1px solid rgba(0,255,136,0.1)" }}
              >
                <span style={{ color: "rgba(107,143,168,0.6)" }}>t =</span>
                <span style={{ color: "var(--brand-accent)" }}>{state.time.toFixed(1)}s</span>
                <span style={{ color: "rgba(107,143,168,0.3)" }}>·</span>
                <span style={{ color: "rgba(107,143,168,0.6)" }}>step</span>
                <span style={{ color: "var(--brand-accent)" }}>{store.episode_step}/200</span>
              </div>
            )}
            {selectedScenario && (
              <div
                className="text-xs px-3 py-1.5 rounded-lg font-semibold"
                style={{
                  background: "rgba(0,196,255,0.08)",
                  border: "1px solid rgba(0,196,255,0.2)",
                  color: "#a0d8e8",
                }}
              >
                {selectedScenario === "lofa" ? "🌊 LOFA"
                  : selectedScenario === "rod_stuck" ? "🔒 Rod Stuck"
                  : selectedScenario === "power_ramp" ? "📈 Power Ramp"
                  : selectedScenario === "sensor_noise" ? "📡 Sensor Noise"
                  : selectedScenario === "normal" ? "🔋 Normal"
                  : selectedScenario}
              </div>
            )}
          </div>

          {/* Right: Connection indicators */}
          <div className="flex items-center gap-2">
            <div
              className="flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-lg"
              style={{
                background: wsConnected ? "rgba(0,255,136,0.08)" : "rgba(255,214,0,0.08)",
                border: `1px solid ${wsConnected ? "rgba(0,255,136,0.3)" : "rgba(255,214,0,0.3)"}`,
                color: wsConnected ? "var(--brand-accent)" : "#fbbf24",
              }}
            >
              <div className={wsConnected ? "led-green" : "led-yellow"} style={{ width: 6, height: 6 }} />
              {wsConnected ? "WS Live" : "Polling"}
            </div>
            <div
              className="flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-lg"
              style={{
                background: "rgba(0,255,136,0.08)",
                border: "1px solid rgba(0,255,136,0.3)",
                color: "var(--brand-accent)",
              }}
            >
              <div className="led-green" style={{ width: 6, height: 6 }} />
              Backend Online
            </div>
          </div>
        </div>
      </header>

      {/* ═══ CRITICAL ALERT BANNER ════════════════════════════════════════════ */}
      {isCritical && (
        <div
          className="px-4 py-2 text-center text-sm font-bold"
          style={{
            background: "rgba(255,59,59,0.15)",
            borderBottom: "2px solid rgba(255,59,59,0.8)",
            color: "#ff6568",
            animationName: "critical-pulse",
            animationDuration: "1s",
            animationTimingFunction: "ease-in-out",
            animationIterationCount: "infinite",
          }}
        >
          🔥 CRITICAL: Fuel Temperature {state?.fuel_temp.toFixed(0)}K — EXCEEDS 1100K SAFETY LIMIT
        </div>
      )}

      {/* ═══ ERROR BAR ════════════════════════════════════════════════════════ */}
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

      {/* ═══ NEW INTRO BANNER — explains what to do ═══════════════════════════ */}
      {!store.is_running && !state && (
        <div className="mx-auto max-w-screen-2xl px-4 pt-4">
          <div
            className="rounded-xl p-5"
            style={{
              background: "rgba(0,255,136,0.04)",
              border: "1px solid rgba(0,255,136,0.12)",
            }}
          >
            <div className="flex flex-wrap items-start gap-6">
              <div className="flex-1 min-w-0">
                <h2 className="text-sm font-bold mb-1" style={{ color: "var(--brand-accent)" }}>
                  How to Use This Dashboard
                </h2>
                <p className="text-xs leading-relaxed" style={{ color: "rgba(160,216,232,0.7)" }}>
                  This is a live AI simulation of a nuclear reactor. Choose a <strong style={{ color: "#e2f0ff" }}>scenario</strong> from the left panel
                  (like LOFA or Power Ramp), then click <strong style={{ color: "#e2f0ff" }}>Launch Simulation</strong>.
                  The AI controller will take over and try to stabilize the reactor while you watch in real time.
                </p>
              </div>
              <div className="flex gap-4 flex-shrink-0 text-xs">
                {[
                  { icon: "🌊", label: "LOFA", desc: "Coolant pump fails" },
                  { icon: "📈", label: "Power Ramp", desc: "Grid demand spikes" },
                  { icon: "🔒", label: "Rod Stuck", desc: "Control rod jams" },
                  { icon: "📡", label: "Sensor Noise", desc: "Bad readings" },
                ].map((s) => (
                  <div
                    key={s.label}
                    className="text-center px-3 py-2 rounded-lg"
                    style={{ background: "rgba(0,0,0,0.3)", border: "1px solid rgba(255,255,255,0.05)" }}
                  >
                    <div className="text-xl mb-1">{s.icon}</div>
                    <p className="font-bold" style={{ color: "#e2f0ff" }}>{s.label}</p>
                    <p style={{ color: "rgba(107,143,168,0.6)" }}>{s.desc}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ═══ MAIN 3-PANEL LAYOUT ══════════════════════════════════════════════ */}
      <main className="mx-auto max-w-screen-2xl px-4 py-4">
        <div
          className="grid gap-4 lg:grid-cols-[320px_minmax(0,1fr)_360px]"
          style={{ minHeight: "calc(100vh - 140px)" }}
        >

          {/* ══════ LEFT PANEL: CONTROLS + SCENARIO INTEL ══════ */}
          <aside className="order-2 space-y-3 lg:order-1">

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

            {/* Progress Bar */}
            {store.is_running && state && (
              <div className="nuclear-panel p-4 space-y-3">
                <p className="section-label">Episode Progress</p>
                <div>
                  <div className="flex justify-between text-xs mb-1" style={{ color: "rgba(107,143,168,0.7)" }}>
                    <span>Step {store.episode_step}</span>
                    <span>{((store.episode_step / 200) * 100).toFixed(0)}%</span>
                  </div>
                  <div className="h-1.5 rounded-full overflow-hidden" style={{ background: "rgba(0,255,136,0.1)" }}>
                    <div
                      className="h-full rounded-full transition-all duration-300"
                      style={{
                        width: `${(store.episode_step / 200) * 100}%`,
                        background: "var(--brand-accent)",
                        boxShadow: "0 0 6px rgba(0,255,136,0.5)",
                      }}
                    />
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-2">
                  {[
                    { label: "Sim Time", value: `${state.time.toFixed(1)}s`, color: "var(--brand-accent)" },
                    { label: "Score", value: (store.metrics?.total_reward ?? 0).toFixed(1), color: (store.metrics?.total_reward ?? 0) >= 0 ? "var(--brand-accent)" : "#ff6568" },
                    { label: "Events", value: store.events.length.toString(), color: "#fbbf24" },
                  ].map((item, i) => (
                    <div
                      key={i}
                      style={{ background: "rgba(0,0,0,0.3)", border: "1px solid rgba(0,255,136,0.1)" }}
                      className="rounded-lg p-2 text-center"
                    >
                      <p className="section-label" style={{ fontSize: "0.5rem" }}>{item.label}</p>
                      <p className="font-mono text-sm font-bold" style={{ color: item.color }}>
                        {item.value}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* ── Scenario Intel Panel ── */}
            <ScenarioIntelPanel
              selectedScenario={selectedScenario}
              isRunning={store.is_running}
              currentState={state}
            />

            {/* Manual Override */}
            <div className="nuclear-panel p-4">
              <p className="section-label mb-3">Manual Override</p>
              <ManualControl
                onControlChange={handleManual}
                isEnabled={store.is_running}
              />
            </div>
          </aside>

          {/* ══════ CENTER PANEL: REACTOR VISUALIZATION ══════ */}
          <section className="order-1 space-y-4 lg:order-2 lg:min-w-0">

            {/* Main reactor SVG — bigger */}
            <ReactorVisualization
              state={state}
              rodPosition={store.last_action?.control_rod ?? 0}
              coolantFlow={store.last_action?.coolant_flow ?? 0}
              isRunning={store.is_running}
            />

            {/* 3 Analog Gauges */}
            {state && (
              <div className="nuclear-panel p-4">
                <p className="section-label mb-4">Core Instrument Panel</p>
                <div className="grid grid-cols-3 gap-4">
                  <AnalogGauge
                    value={state.power * 100}
                    min={0} max={150}
                    label="Power" unit="%"
                    safeMin={80} safeMax={120} warnMax={135}
                    size={130}
                  />
                  <AnalogGauge
                    value={state.fuel_temp}
                    min={700} max={1200}
                    label="Fuel Temp" unit="K"
                    safeMin={800} safeMax={1000} warnMax={1100}
                    size={130}
                  />
                  <AnalogGauge
                    value={state.pressure}
                    min={5} max={16}
                    label="Pressure" unit="bar"
                    safeMin={8} safeMax={12} warnMax={13.5}
                    size={130}
                  />
                </div>
              </div>
            )}

            {/* AI Decision Engine panel */}
            {store.is_running && store.last_action && (
              <div
                className="nuclear-panel p-4"
                style={{ border: "1px solid rgba(0,255,136,0.15)", background: "rgba(0,255,136,0.02)" }}
              >
                <div className="flex items-center justify-between mb-4">
                  <p className="section-label">🤖 AI Decision Engine (SAC)</p>
                  <span
                    className="text-xs font-mono"
                    style={{
                      color: "rgba(0,255,136,0.5)",
                      animationName: "led-blink",
                      animationDuration: "1.2s",
                      animationIterationCount: "infinite",
                    }}
                  >
                    ANALYZING...
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-4 mb-4">
                  {/* Rod action */}
                  <div>
                    <div className="flex justify-between text-xs mb-2 uppercase font-medium">
                      <span style={{ color: "rgba(107,143,168,0.7)" }}>Control Rod Command</span>
                      <span
                        className="font-mono font-bold"
                        style={{ color: store.last_action.control_rod > 0.005 ? "#fb2c36" : store.last_action.control_rod < -0.005 ? "#9ca3af" : "#fbbf24" }}
                      >
                        {store.last_action.control_rod > 0.005
                          ? "↓ INSERT"
                          : store.last_action.control_rod < -0.005
                          ? "↑ WITHDRAW"
                          : "HOLD"}
                      </span>
                    </div>
                    <div className="h-2 rounded-full overflow-hidden bg-white/5">
                      <div
                        className="h-full rounded-full transition-all duration-300"
                        style={{
                          width: `${Math.min(100, Math.abs(store.last_action.control_rod) * 200)}%`,
                          background: store.last_action.control_rod > 0.005 ? "#fb2c36" : store.last_action.control_rod < -0.005 ? "#9ca3af" : "#fbbf24",
                          boxShadow: `0 0 8px ${store.last_action.control_rod > 0.005 ? "#fb2c3660" : "#9ca3af60"}`,
                        }}
                      />
                    </div>
                    <p className="text-xs mt-1 font-mono" style={{ color: "rgba(107,143,168,0.5)" }}>
                      Δ = {store.last_action.control_rod.toFixed(5)}
                    </p>
                  </div>

                  {/* Coolant action */}
                  <div>
                    <div className="flex justify-between text-xs mb-2 uppercase font-medium">
                      <span style={{ color: "rgba(107,143,168,0.7)" }}>Coolant Flow Adjust</span>
                      <span
                        className="font-mono font-bold"
                        style={{ color: "var(--brand-accent)" }}
                      >
                        {store.last_action.coolant_flow > 0.005
                          ? "+ BOOST"
                          : store.last_action.coolant_flow < -0.005
                          ? "– REDUCE"
                          : "STEADY"}
                      </span>
                    </div>
                    <div className="h-2 rounded-full overflow-hidden bg-white/5">
                      <div
                        className="h-full rounded-full transition-all duration-300"
                        style={{
                          width: `${Math.min(100, Math.abs(store.last_action.coolant_flow) * 200)}%`,
                          background: "var(--brand-accent)",
                          boxShadow: "0 0 8px rgba(0,255,136,0.4)",
                        }}
                      />
                    </div>
                    <p className="text-xs mt-1 font-mono" style={{ color: "rgba(107,143,168,0.5)" }}>
                      Δ = {store.last_action.coolant_flow > 0 ? "+" : ""}{store.last_action.coolant_flow.toFixed(5)}
                    </p>
                  </div>
                </div>

                {/* Intent explanation */}
                <div
                  className="rounded-lg px-3 py-2.5"
                  style={{ background: "rgba(0,0,0,0.4)", border: "1px solid rgba(255,255,255,0.04)" }}
                >
                  <p className="text-xs uppercase tracking-wider mb-1" style={{ color: "rgba(107,143,168,0.4)" }}>
                    Why the AI is taking this action:
                  </p>
                  <p className="text-xs font-medium leading-relaxed" style={{ color: "#a0d8e8" }}>
                    {store.last_action.control_rod > 0.01
                      ? "🔴 Inserting control rods deeper into the core. This absorbs neutrons, slowing the fission chain reaction and reducing heat generation."
                      : store.last_action.control_rod < -0.01
                      ? "⬆️ Withdrawing control rods from the core. This allows more neutrons to cause fission, increasing reactor power output."
                      : store.last_action.coolant_flow > 0.01
                      ? "💧 Boosting coolant flow rate. More coolant removes heat from the core faster, preventing temperature rise."
                      : store.last_action.coolant_flow < -0.01
                      ? "💦 Reducing coolant flow. Slowing the coolant allows it to absorb more heat before exiting, raising system pressure slightly."
                      : "✅ Holding steady — current reactor state is within acceptable limits. No control action required this step."}
                  </p>
                </div>
              </div>
            )}

            {/* Live Narrator — real-time plain English */}
            <LiveNarrator
              state={state}
              action={store.last_action}
              scenario={selectedScenario}
              isRunning={store.is_running}
              step={store.episode_step}
            />

            {/* Live Graph */}
            <div className="nuclear-panel p-4">
              <p className="section-label mb-3">Live Parameter Trends</p>
              <LiveGraphs history={store.history || []} isRunning={store.is_running} />
            </div>
          </section>

          {/* ══════ RIGHT PANEL: TELEMETRY, FAULT DETECTOR, REPORT ══════ */}
          <aside className="order-3 space-y-3 lg:min-w-0">

            {/* Live readings — always shows when state exists */}
            {state && (
              <div className="nuclear-panel p-4">
                <p className="section-label mb-3">Live Telemetry</p>
                <div className="space-y-2">
                  {[
                    {
                      label: "Fuel Temperature",
                      value: `${state.fuel_temp.toFixed(1)} K`,
                      target: "Danger: > 1100K",
                      detail: isCritical ? "CRITICAL — SHUTDOWN RISK" : isWarning ? "Warning zone" : "Normal operating range",
                      pct: Math.min(state.fuel_temp / 1150, 1),
                      color: isCritical ? "#ff3b3b" : isWarning ? "#fbbf24" : "var(--brand-accent)",
                    },
                    {
                      label: "Reactor Power",
                      value: `${(state.power * 100).toFixed(1)} %`,
                      target: "Target: 80–120%",
                      detail: state.power < 0.8 ? "Under-producing" : state.power > 1.2 ? "Over-power" : "On target",
                      pct: Math.min(state.power / 1.5, 1),
                      color: state.power < 0.8 || state.power > 1.2 ? "#fbbf24" : "var(--brand-accent)",
                    },
                    {
                      label: "Coolant Temperature",
                      value: `${state.coolant_temp.toFixed(1)} K`,
                      target: "Safe band: 280–320K",
                      detail: state.coolant_temp > 320 ? "Overheating — check flow" : "Normal",
                      pct: Math.min(state.coolant_temp / 350, 1),
                      color: state.coolant_temp > 320 ? "#fbbf24" : "#9ca3af",
                    },
                    {
                      label: "System Pressure",
                      value: `${state.pressure.toFixed(2)} bar`,
                      target: "Safe: 8–12 bar",
                      detail: state.pressure > 13 ? "High pressure — risk of leak" : state.pressure < 7 ? "Low pressure" : "Normal",
                      pct: Math.min((state.pressure - 5) / 11, 1),
                      color: state.pressure < 8 || state.pressure > 12 ? "#fbbf24" : "var(--brand-accent)",
                    },
                    {
                      label: "Coolant Flow",
                      value: `${(state.coolant_flow_actual ?? 0).toFixed(0)} kg/s`,
                      target: "Nominal: ~8,000 kg/s",
                      detail: (state.coolant_flow_actual ?? 0) < 4000 ? "⚠️ Low flow (LOFA risk)" : "Adequate flow",
                      pct: Math.min((state.coolant_flow_actual ?? 0) / 12000, 1),
                      color: (state.coolant_flow_actual ?? 0) < 4000 ? "#fbbf24" : "#9ca3af",
                    },
                    {
                      label: "Power Rate",
                      value: `${state.power_rate > 0 ? "+" : ""}${state.power_rate.toFixed(4)}/s`,
                      target: "Stable: < ±0.02/s",
                      detail: Math.abs(state.power_rate) > 0.05 ? "Rapid power change!" : "Stable rate",
                      pct: Math.min(Math.abs(state.power_rate) / 0.15, 1),
                      color: Math.abs(state.power_rate) > 0.05 ? "#fbbf24" : "#6b8fa8",
                    },
                  ].map((item, i) => (
                    <div
                      key={i}
                      className="rounded-lg px-3 py-2"
                      style={{
                        background: "rgba(0,0,0,0.3)",
                        borderLeft: `3px solid ${item.color}`,
                        borderTop: "1px solid rgba(255,255,255,0.03)",
                        borderRight: "1px solid rgba(255,255,255,0.03)",
                        borderBottom: "1px solid rgba(255,255,255,0.03)",
                      }}
                    >
                      <div className="flex justify-between items-baseline">
                        <span className="text-xs" style={{ color: "rgba(107,143,168,0.7)" }}>{item.label}</span>
                        <span
                          className="font-mono font-bold text-sm"
                          style={{
                            color: item.color,
                            textShadow: `0 0 6px ${item.color}60`,
                            transitionProperty: "color",
                            transitionDuration: "0.3s",
                            transitionTimingFunction: "ease",
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
                      <div className="flex justify-between mt-0.5">
                        <p className="text-xs" style={{ color: "rgba(107,143,168,0.4)", fontSize: "0.58rem" }}>
                          {item.target}
                        </p>
                        <p className="text-xs" style={{ color: `${item.color}90`, fontSize: "0.58rem" }}>
                          {item.detail}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* LSTM Fault Detector */}
            <div className="nuclear-panel p-4">
              <div className="flex items-center justify-between mb-3">
                <p className="section-label">LSTM Fault Detector</p>
                <span
                  className="text-xs font-mono uppercase"
                  style={{ color: faultAccent }}
                >
                  {fault?.status === "prediction_ready"
                    ? fault.predicted_state
                    : fault?.status === "error"
                    ? "Offline"
                    : "Buffering"}
                </span>
              </div>

              {!fault || fault.status === "insufficient_data" ? (
                <div
                  className="rounded-lg p-3 text-sm"
                  style={{
                    background: "rgba(0,0,0,0.3)",
                    border: "1px solid rgba(0,255,136,0.08)",
                    color: "rgba(160,216,232,0.8)",
                  }}
                >
                  {fault?.message ?? "Collecting sequence history for the first LSTM prediction."}
                </div>
              ) : fault.status === "error" ? (
                <div
                  className="rounded-lg p-3 text-sm"
                  style={{
                    background: "rgba(255,59,59,0.08)",
                    border: "1px solid rgba(255,59,59,0.25)",
                    color: "#ff9c9c",
                  }}
                >
                  {fault.message ?? "Fault detector unavailable."}
                </div>
              ) : (
                <div className="space-y-3">
                  <div
                    className="rounded-lg p-3"
                    style={{ background: "rgba(0,0,0,0.3)", border: `1px solid ${faultAccent}33` }}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs uppercase" style={{ color: "rgba(107,143,168,0.7)" }}>
                        Predicted State
                      </span>
                      <span className="font-mono font-bold" style={{ color: faultAccent }}>
                        {fault.predicted_state}
                      </span>
                    </div>
                    <div className="flex items-center justify-between mt-2 text-xs">
                      <span style={{ color: "rgba(107,143,168,0.7)" }}>Confidence</span>
                      <span style={{ color: "#a0d8e8" }}>
                        {fault.confidence !== undefined ? `${(fault.confidence * 100).toFixed(1)}%` : "N/A"}
                      </span>
                    </div>
                    <div className="flex items-center justify-between mt-1 text-xs">
                      <span style={{ color: "rgba(107,143,168,0.7)" }}>Risk Level</span>
                      <span style={{ color: faultAccent }}>
                        {(fault.risk_level ?? "low").toUpperCase()}
                      </span>
                    </div>
                  </div>

                  {fault.class_probabilities && (
                    <div className="space-y-2">
                      {(["Normal", "Scram", "LOFA"] as const).map((label) => (
                        <div key={label}>
                          <div className="flex justify-between text-xs mb-1 uppercase font-medium">
                            <span style={{ color: "rgba(107,143,168,0.7)" }}>{label}</span>
                            <span className="font-mono" style={{ color: "#a0d8e8" }}>
                              {((fault.class_probabilities?.[label] ?? 0) * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="h-1.5 rounded-full overflow-hidden bg-white/5">
                            <div
                              className="h-full rounded-full transition-all duration-300"
                              style={{
                                width: `${(fault.class_probabilities?.[label] ?? 0) * 100}%`,
                                background:
                                  label === "LOFA"
                                    ? "#ff6568"
                                    : label === "Scram"
                                    ? "#fbbf24"
                                    : "var(--brand-accent)",
                              }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  {fault.recommendations && fault.recommendations.length > 0 && (
                    <div
                      className="rounded-lg p-3 text-xs leading-relaxed"
                      style={{
                        background: "rgba(0,0,0,0.25)",
                        border: "1px solid rgba(255,255,255,0.04)",
                        color: "#a0d8e8",
                      }}
                    >
                      {fault.recommendations[0]}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Rate Indicators */}
            {state && (
              <div className="nuclear-panel p-4">
                <p className="section-label mb-2">Rate of Change Indicators</p>
                <div className="grid grid-cols-2 gap-2">
                  {[
                    {
                      label: "Power Rate",
                      value: state.power_rate,
                      color: Math.abs(state.power_rate) > 0.05 ? "#fbbf24" : "var(--brand-accent)",
                      unit: "/s",
                      desc: "How fast power is changing",
                    },
                    {
                      label: "Temp Rate",
                      value: state.temp_rate,
                      color: Math.abs(state.temp_rate) > 5 ? "#fbbf24" : "#9ca3af",
                      unit: "K/s",
                      desc: "How fast fuel temp rises",
                    },
                  ].map((item, i) => (
                    <div
                      key={i}
                      className="rounded-lg p-2.5 text-center"
                      style={{ background: "rgba(0,0,0,0.3)", border: "1px solid rgba(0,255,136,0.06)" }}
                    >
                      <p className="section-label" style={{ fontSize: "0.55rem" }}>{item.label}</p>
                      <p
                        className="font-mono text-sm font-bold mt-0.5"
                        style={{ color: item.color }}
                      >
                        {item.value > 0 ? "+" : ""}{item.value.toFixed(4)} {item.unit}
                      </p>
                      <p style={{ color: "rgba(107,143,168,0.4)", fontSize: "0.55rem" }}>{item.desc}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* ── Post-Simulation Mission Report ── */}
            {!store.is_running && store.metrics && store.metrics.episode_steps > 0 && (
              <>
                <MissionReport
                  metrics={store.metrics}
                  scenario={selectedScenario}
                  history={store.history || []}
                  isRunning={store.is_running}
                />
                <button
                  onClick={() => store.reset()}
                  className="w-full py-2.5 text-xs uppercase tracking-widest font-bold rounded-xl transition-all duration-200"
                  style={{
                    background: "rgba(0,255,136,0.06)",
                    border: "1px solid rgba(0,255,136,0.2)",
                    color: "var(--brand-accent)",
                  }}
                >
                  ↺ Acknowledge & Run New Scenario
                </button>
              </>
            )}

            {/* Event Log */}
            <div className="nuclear-panel p-4" style={{ flex: 1 }}>
              <EventLog
                events={store.events}
                onClear={() => store.clearEvents()}
              />
            </div>

          </aside>
        </div>
      </main>

      {/* FOOTER */}
      <footer
        className="mt-4 py-2 px-4 text-center text-xs"
        style={{
          background: "rgba(5,15,31,0.8)",
          borderTop: "1px solid rgba(0,255,136,0.08)",
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
