"use client";

import React, { useState } from "react";
import { Model, Scenario, Action } from "@/types/reactor";
import { Play, Square, RotateCcw, ChevronRight, Zap, AlertTriangle, Activity } from "lucide-react";

// ─────────────────────────────────────────────
// MODEL SELECTOR — Card-style selection
// ─────────────────────────────────────────────
interface ModelSelectorProps {
  models: Model[];
  selectedModel: string | null;
  onSelect: (modelId: string) => void;
  isLoading?: boolean;
  disabled?: boolean;
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  models,
  selectedModel,
  onSelect,
  isLoading,
  disabled,
}) => {
  const safeModels = models || [];

  return (
    <div>
      <p className="section-label mb-2">AI Controller Model</p>
      {isLoading ? (
        <div className="shimmer h-20 rounded-lg" />
      ) : safeModels.length === 0 ? (
        <div
          className="rounded-lg p-3 text-center text-xs"
          style={{ background: "rgba(255,59,59,0.1)", border: "1px solid rgba(255,59,59,0.3)", color: "#ff8080" }}
        >
          Backend offline — no models available
        </div>
      ) : (
        <div className="space-y-2">
          {safeModels.map((model) => {
            const isSelected = selectedModel === model.id;
            return (
              <button
                key={model.id}
                onClick={() => !disabled && onSelect(model.id)}
                disabled={disabled}
                className="w-full text-left rounded-lg p-3 transition-all duration-200"
                style={{
                  background: isSelected
                    ? "rgba(0,255,136,0.12)"
                    : "rgba(5,15,31,0.7)",
                  border: isSelected
                    ? "1px solid rgba(0,255,136,0.5)"
                    : "1px solid rgba(0,255,136,0.1)",
                  boxShadow: isSelected ? "0 0 12px rgba(0,255,136,0.15)" : "none",
                  cursor: disabled ? "not-allowed" : "pointer",
                  opacity: disabled ? 0.5 : 1,
                }}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div
                      className="h-2 w-2 rounded-full"
                      style={{
                        background: isSelected ? "var(--brand-accent)" : "rgba(0,255,136,0.3)",
                        boxShadow: isSelected ? "0 0 6px var(--brand-accent)" : "none",
                        transition: "all 0.2s",
                      }}
                    />
                    <span
                      className="text-sm font-semibold"
                      style={{ color: isSelected ? "var(--brand-accent)" : "#a0b8c8" }}
                    >
                      {model.name}
                    </span>
                  </div>
                  <span
                    className="text-xs font-bold px-2 py-0.5 rounded-full font-mono"
                    style={{
                      background: isSelected ? "rgba(0,255,136,0.2)" : "rgba(255,255,255,0.05)",
                      color: isSelected ? "var(--brand-accent)" : "#6b8fa8",
                      border: isSelected ? "1px solid rgba(0,255,136,0.4)" : "1px solid rgba(255,255,255,0.08)",
                    }}
                  >
                    {model.reward_per_step.toFixed(1)} r/s
                  </span>
                </div>
                {isSelected && model.description && (
                  <p className="mt-1.5 text-xs leading-relaxed" style={{ color: "rgba(0,255,136,0.6)" }}>
                    {model.description}
                  </p>
                )}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
};

// ─────────────────────────────────────────────
// SCENARIO SELECTOR — Card-style with difficulty
// ─────────────────────────────────────────────
interface ScenarioSelectorProps {
  scenarios: Scenario[];
  selectedScenario: string | null;
  onSelect: (scenarioId: string) => void;
  isLoading?: boolean;
  disabled?: boolean;
}

const DIFFICULTY_COLORS: Record<string, string> = {
  easy: "var(--brand-accent)",
  medium: "#fbbf24",
  hard: "#ff6d00",
  extreme: "#ff3b3b",
};

const SCENARIO_ICONS: Record<string, string> = {
  normal: "🔋",
  lofa: "🌊",
  rod_stuck: "🔒",
  power_ramp: "📈",
  sensor_noise: "📡",
};

export const ScenarioSelector: React.FC<ScenarioSelectorProps> = ({
  scenarios,
  selectedScenario,
  onSelect,
  isLoading,
  disabled,
}) => {
  const safeScenarios = scenarios || [];

  return (
    <div>
      <p className="section-label mb-2">Mission Scenario</p>
      {isLoading ? (
        <div className="shimmer h-32 rounded-lg" />
      ) : (
        <div className="space-y-1.5">
          {safeScenarios.map((scenario) => {
            const isSelected = selectedScenario === scenario.id;
            const diff = scenario.difficulty || "medium";
            const diffColor = DIFFICULTY_COLORS[diff] || "#6b8fa8";
            const icon = SCENARIO_ICONS[scenario.id] || "⚡";

            return (
              <button
                key={scenario.id}
                onClick={() => !disabled && onSelect(scenario.id)}
                disabled={disabled}
                className="w-full text-left rounded-lg px-3 py-2 transition-all duration-200"
                style={{
                  background: isSelected ? "rgba(0,255,136,0.1)" : "rgba(5,15,31,0.6)",
                  border: isSelected ? `1px solid ${diffColor}60` : "1px solid rgba(0,255,136,0.08)",
                  cursor: disabled ? "not-allowed" : "pointer",
                  opacity: disabled ? 0.5 : 1,
                }}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-base leading-none">{icon}</span>
                    <span
                      className="text-sm font-medium"
                      style={{ color: isSelected ? "#e2f0ff" : "#8aa0b0" }}
                    >
                      {scenario.name}
                    </span>
                  </div>
                  <span
                    className="text-xs font-bold uppercase tracking-wide"
                    style={{ color: diffColor, fontSize: "0.6rem" }}
                  >
                    {diff}
                  </span>
                </div>
                {isSelected && scenario.description && (
                  <p className="mt-1 text-xs leading-relaxed pl-6" style={{ color: "rgba(0,255,136,0.55)" }}>
                    {scenario.description}
                  </p>
                )}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
};

// ─────────────────────────────────────────────
// CONTROL BUTTONS
// ─────────────────────────────────────────────
interface ControlButtonsProps {
  isRunning: boolean;
  isPaused: boolean;
  isLoading: boolean;
  canStart: boolean;
  onStart: () => void;
  onStop: () => void;
  onReset?: () => void;
}

export const ControlButtons: React.FC<ControlButtonsProps> = ({
  isRunning,
  isPaused,
  isLoading,
  canStart,
  onStart,
  onStop,
  onReset,
}) => {
  return (
    <div className="space-y-2">
      {!isRunning ? (
        <button
          onClick={onStart}
          disabled={isLoading || !canStart}
          className="w-full rounded-lg py-3 font-bold text-sm transition-all duration-200 flex items-center justify-center gap-2"
          style={{
            background: canStart && !isLoading
              ? "linear-gradient(135deg, rgba(0,255,136,0.2), rgba(0,255,136,0.08))"
              : "rgba(255,255,255,0.04)",
            border: canStart && !isLoading
              ? "1px solid rgba(0,255,136,0.5)"
              : "1px solid rgba(255,255,255,0.08)",
            color: canStart && !isLoading ? "var(--brand-accent)" : "rgba(255,255,255,0.25)",
            cursor: canStart && !isLoading ? "pointer" : "not-allowed",
            boxShadow: canStart && !isLoading ? "0 0 20px rgba(0,255,136,0.15), inset 0 0 20px rgba(0,255,136,0.05)" : "none",
          }}
        >
          {isLoading ? (
            <>
              <div
                className="h-4 w-4 rounded-full border-2 border-t-transparent animate-spin"
                style={{ borderColor: "rgba(0,255,136,0.3)", borderTopColor: "var(--brand-accent)" }}
              />
              Initializing...
            </>
          ) : (
            <>
              <Play className="h-4 w-4" />
              Launch Simulation
            </>
          )}
        </button>
      ) : (
        <button
          onClick={onStop}
          className="w-full rounded-lg py-3 font-bold text-sm transition-all duration-200 flex items-center justify-center gap-2"
          style={{
            background: "linear-gradient(135deg, rgba(255,59,59,0.2), rgba(255,59,59,0.08))",
            border: "1px solid rgba(255,59,59,0.5)",
            color: "#ff6568",
            cursor: "pointer",
            boxShadow: "0 0 12px rgba(255,59,59,0.15)",
          }}
        >
          <Square className="h-4 w-4" />
          Stop Simulation
        </button>
      )}

      {onReset && (
        <button
          onClick={onReset}
          disabled={isLoading || isRunning}
          className="w-full rounded-lg py-2 text-xs font-semibold transition-all duration-200 flex items-center justify-center gap-2"
          style={{
            background: "rgba(255,255,255,0.03)",
            border: "1px solid rgba(255,255,255,0.08)",
            color: isRunning ? "rgba(255,255,255,0.2)" : "rgba(107,143,168,0.8)",
            cursor: isRunning ? "not-allowed" : "pointer",
          }}
        >
          <RotateCcw className="h-3 w-3" />
          Reset
        </button>
      )}
    </div>
  );
};

// ─────────────────────────────────────────────
// MANUAL CONTROL — Sliders
// ─────────────────────────────────────────────
interface ManualControlProps {
  onControlChange: (action: Action) => void;
  isEnabled: boolean;
}

export const ManualControl: React.FC<ManualControlProps> = ({
  onControlChange,
  isEnabled,
}) => {
  const [controlRod, setControlRod] = useState(0);
  const [coolantFlow, setCoolantFlow] = useState(0);

  const handleRodChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const v = parseFloat(e.target.value);
    setControlRod(v);
    onControlChange({ control_rod: v, coolant_flow: coolantFlow });
  };

  const handleFlowChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const v = parseFloat(e.target.value);
    setCoolantFlow(v);
    onControlChange({ control_rod: controlRod, coolant_flow: v });
  };

  return (
    <div style={{ opacity: isEnabled ? 1 : 0.4, pointerEvents: isEnabled ? "auto" : "none" }}>
      {/* Control Rod Slider */}
      <div className="mb-4">
        <div className="flex justify-between items-center mb-2">
          <span className="section-label">Control Rod</span>
          <span
            className="text-xs font-bold font-mono px-2 py-0.5 rounded"
            style={{
              background: Math.abs(controlRod) > 0.3 ? "rgba(0,255,136,0.15)" : "rgba(255,255,255,0.05)",
              color: controlRod > 0.2 ? "#ff6568" : controlRod < -0.2 ? "#9ca3af" : "#6b8fa8",
              border: "1px solid rgba(0,255,136,0.15)",
            }}
          >
            {controlRod > 0 ? "↓" : controlRod < 0 ? "↑" : "—"} {controlRod.toFixed(2)}
          </span>
        </div>
        <input
          type="range" min="-1" max="1" step="0.05"
          value={controlRod}
          onChange={handleRodChange}
          disabled={!isEnabled}
          className="w-full"
          style={{ accentColor: "var(--brand-accent)" }}
        />
        <div className="flex justify-between text-xs mt-1" style={{ color: "rgba(107,143,168,0.5)" }}>
          <span>↑ Withdraw</span>
          <span>0</span>
          <span>Insert ↓</span>
        </div>
      </div>

      {/* Coolant Flow Slider */}
      <div>
        <div className="flex justify-between items-center mb-2">
          <span className="section-label">Coolant Flow</span>
          <span
            className="text-xs font-bold font-mono px-2 py-0.5 rounded"
            style={{
              background: Math.abs(coolantFlow) > 0.3 ? "rgba(64,196,255,0.1)" : "rgba(255,255,255,0.05)",
              color: "#9ca3af",
              border: "1px solid rgba(64,196,255,0.2)",
            }}
          >
            {coolantFlow > 0 ? "+" : ""}{coolantFlow.toFixed(2)}
          </span>
        </div>
        <input
          type="range" min="-1" max="1" step="0.05"
          value={coolantFlow}
          onChange={handleFlowChange}
          disabled={!isEnabled}
          className="w-full"
          style={{ accentColor: "#9ca3af" }}
        />
        <div className="flex justify-between text-xs mt-1" style={{ color: "rgba(107,143,168,0.5)" }}>
          <span>← Reduce</span>
          <span>Normal</span>
          <span>Boost →</span>
        </div>
      </div>

      {!isEnabled && (
        <p className="text-xs mt-3 text-center" style={{ color: "rgba(107,143,168,0.5)" }}>
          Start simulation to enable manual override
        </p>
      )}
    </div>
  );
};

// ─────────────────────────────────────────────
// SIMULATION STATUS CHIP (used in header)
// ─────────────────────────────────────────────
export const SimulationStatus: React.FC<{
  isRunning: boolean;
  currentModel: string | null;
  episodeStep: number;
}> = ({ isRunning, currentModel, episodeStep }) => (
  <div className="flex items-center gap-2">
    {isRunning ? (
      <span
        className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold"
        style={{
          background: "rgba(0,255,136,0.15)",
          border: "1px solid rgba(0,255,136,0.4)",
          color: "var(--brand-accent)",
        }}
      >
        <span className="led-green" style={{ width: 6, height: 6 }} />
        RUNNING · Step {episodeStep}
      </span>
    ) : (
      <span
        className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold"
        style={{
          background: "rgba(107,143,168,0.1)",
          border: "1px solid rgba(107,143,168,0.2)",
          color: "#6b8fa8",
        }}
      >
        ◻ STANDBY
      </span>
    )}
  </div>
);


