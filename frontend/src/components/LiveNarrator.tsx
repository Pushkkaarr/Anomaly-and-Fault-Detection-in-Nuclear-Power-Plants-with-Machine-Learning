"use client";

import React, { useEffect, useState, useRef } from "react";
import { ReactorState, Action } from "@/types/reactor";

interface LiveNarratorProps {
  state: ReactorState | null;
  action: Action | null;
  scenario: string | null;
  isRunning: boolean;
  step: number;
}

interface NarratorLine {
  id: number;
  text: string;
  severity: "info" | "warning" | "critical" | "success" | "action";
  timestamp: number;
}

let _lineId = 0;

function buildNarratorLines(
  state: ReactorState,
  action: Action | null,
  scenario: string | null,
  step: number,
  prevState: ReactorState | null
): NarratorLine[] {
  const lines: NarratorLine[] = [];
  const now = Date.now();

  const addLine = (text: string, severity: NarratorLine["severity"]) => {
    lines.push({ id: _lineId++, text, severity, timestamp: now });
  };

  const fuelTemp = state.fuel_temp;
  const power = state.power * 100;
  const pressure = state.pressure;
  const coolantTemp = state.coolant_temp;

  // === Critical alerts first ===
  if (fuelTemp > 1100) {
    addLine(`🔥 CRITICAL: Fuel temp ${fuelTemp.toFixed(0)}K exceeds 1100K safety limit — core damage risk!`, "critical");
  } else if (fuelTemp > 1050) {
    addLine(`⚠️ Alert: Fuel temp ${fuelTemp.toFixed(0)}K approaching safety boundary at 1100K`, "warning");
  }

  if (pressure > 13.5) {
    addLine(`💨 High pressure: ${pressure.toFixed(2)} bar — primary coolant loop overpressure`, "critical");
  } else if (pressure > 12.5) {
    addLine(`💨 Pressure elevated: ${pressure.toFixed(2)} bar (limit: 13.5 bar)`, "warning");
  } else if (pressure < 7) {
    addLine(`💨 Low pressure: ${pressure.toFixed(2)} bar — possible coolant leak`, "warning");
  }

  if (power > 130) {
    addLine(`⚡ Power surge: ${power.toFixed(0)}% — far above nominal 100%`, "critical");
  } else if (power < 70) {
    addLine(`⚡ Power low: ${power.toFixed(0)}% — reactor under-producing`, "warning");
  }

  // === Scenario-specific narration ===
  if (scenario === "lofa" && coolantTemp > 320) {
    addLine(
      `🌊 LOFA Active: Coolant temp ${coolantTemp.toFixed(0)}K — reduced flow starving heat removal`,
      "warning"
    );
  }

  if (scenario === "power_ramp" && Math.abs(state.power_rate) > 0.06) {
    addLine(
      `📈 Ramp transient: Power changing at ${state.power_rate.toFixed(4)}/s — AI tracking demand`,
      "info"
    );
  }

  if (scenario === "rod_stuck") {
    addLine(`🔒 Rod Stuck: Control rod unresponsive — AI using coolant-only control`, "warning");
  }

  if (scenario === "sensor_noise") {
    addLine(`📡 Sensor Noise Active: Instrument readings have random perturbations`, "info");
  }

  // === AI action narration ===
  if (action) {
    const rod = action.control_rod;
    const flow = action.coolant_flow;

    if (Math.abs(rod) > 0.01) {
      const rodAction = rod > 0
        ? `inserting rods (deepening absorption) to reduce reactivity`
        : `withdrawing rods to increase neutron flux and power`;
      addLine(`🤖 AI → ${rodAction} [Δ${rod.toFixed(4)}]`, "action");
    }

    if (Math.abs(flow) > 0.01) {
      const flowAction = flow > 0
        ? `boosting coolant flow to increase heat removal`
        : `reducing coolant flow to raise coolant temperature`;
      addLine(`🤖 AI → ${flowAction} [${flow > 0 ? "+" : ""}${flow.toFixed(4)}]`, "action");
    }

    if (Math.abs(rod) <= 0.01 && Math.abs(flow) <= 0.01) {
      if (step % 10 === 0) {
        addLine(`🤖 AI → Holding steady — reactor in equilibrium, no adjustment needed`, "success");
      }
    }
  }

  // === Periodic status ===
  if (step % 20 === 0 && step > 0) {
    const allGood = fuelTemp < 960 && power >= 80 && power <= 120 && pressure >= 8 && pressure <= 12;
    if (allGood) {
      addLine(
        `✅ Step ${step}: All parameters nominal — Fuel ${fuelTemp.toFixed(0)}K, Power ${power.toFixed(0)}%, P ${pressure.toFixed(1)}bar`,
        "success"
      );
    }
  }

  // === Temperature trend from previous state ===
  if (prevState) {
    const tempDelta = fuelTemp - prevState.fuel_temp;
    if (tempDelta > 15) {
      addLine(`🌡️ Rapid temp rise: +${tempDelta.toFixed(1)}K this step — immediate intervention needed`, "warning");
    } else if (tempDelta < -15) {
      addLine(`🌡️ Temperature dropping: ${tempDelta.toFixed(1)}K this step — cooling effective`, "success");
    }
  }

  return lines;
}

export const LiveNarrator: React.FC<LiveNarratorProps> = ({
  state,
  action,
  scenario,
  isRunning,
  step,
}) => {
  const [lines, setLines] = useState<NarratorLine[]>([]);
  const prevStateRef = useRef<ReactorState | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!state || !isRunning) return;

    const newLines = buildNarratorLines(state, action, scenario, step, prevStateRef.current);
    prevStateRef.current = state;

    if (newLines.length > 0) {
      setLines((prev) => [...newLines, ...prev].slice(0, 40));
    }
  }, [state, action, step, scenario, isRunning]);

  // Clear on reset
  useEffect(() => {
    if (!isRunning) {
      prevStateRef.current = null;
    }
  }, [isRunning]);

  const severityColor: Record<NarratorLine["severity"], string> = {
    critical: "#ff3b3b",
    warning: "#fbbf24",
    info: "#a0d8e8",
    success: "#00ff88",
    action: "#9ca3af",
  };

  const severityBg: Record<NarratorLine["severity"], string> = {
    critical: "rgba(255,59,59,0.08)",
    warning: "rgba(251,191,36,0.05)",
    info: "rgba(160,216,232,0.04)",
    success: "rgba(0,255,136,0.04)",
    action: "rgba(255,255,255,0.02)",
  };

  return (
    <div
      className="rounded-xl overflow-hidden"
      style={{
        background: "rgba(3,10,22,0.95)",
        border: "1px solid rgba(0,255,136,0.1)",
      }}
    >
      <div
        className="flex items-center justify-between px-4 py-2.5"
        style={{ borderBottom: "1px solid rgba(255,255,255,0.05)" }}
      >
        <div className="flex items-center gap-2">
          <div
            className="w-2 h-2 rounded-full"
            style={{
              background: isRunning ? "#00ff88" : "rgba(107,143,168,0.3)",
              boxShadow: isRunning ? "0 0 6px #00ff88" : "none",
              animationName: isRunning ? "led-blink" : "none",
              animationDuration: "1.2s",
              animationIterationCount: "infinite",
            }}
          />
          <p
            className="text-xs font-bold uppercase tracking-widest"
            style={{ color: "rgba(107,143,168,0.7)" }}
          >
            Live System Narration
          </p>
        </div>
        {lines.length > 0 && (
          <button
            onClick={() => setLines([])}
            className="text-xs"
            style={{ color: "rgba(107,143,168,0.4)" }}
          >
            Clear
          </button>
        )}
      </div>

      <div
        className="overflow-y-auto p-3 space-y-1"
        style={{ maxHeight: "240px", scrollbarWidth: "thin" }}
      >
        {!isRunning && lines.length === 0 && (
          <p className="text-xs text-center py-4" style={{ color: "rgba(107,143,168,0.3)" }}>
            Narration begins when simulation starts
          </p>
        )}
        {lines.map((line, idx) => (
          <div
            key={line.id}
            className="flex items-start gap-2 rounded-lg px-2.5 py-1.5"
            style={{
              background: idx === 0 ? severityBg[line.severity] : "transparent",
              border: idx === 0 ? `1px solid ${severityColor[line.severity]}15` : "none",
            }}
          >
            <p
              className="text-xs leading-relaxed"
              style={{
                color: idx === 0 ? severityColor[line.severity] : `${severityColor[line.severity]}80`,
                fontSize: idx < 3 ? "0.7rem" : "0.65rem",
              }}
            >
              {line.text}
            </p>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
};

export default LiveNarrator;
