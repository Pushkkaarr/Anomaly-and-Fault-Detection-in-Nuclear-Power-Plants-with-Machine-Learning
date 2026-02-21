"use client";

import React, { useMemo } from "react";
import { ReactorState } from "@/types/reactor";

/**
 * ReactorVisualization Component
 * Visual representation of nuclear reactor with live control indicators
 */

interface ReactorVisualizationProps {
  state: ReactorState | null;
  rodPosition?: number; // -1 to 1 (retracted to inserted)
  coolantFlow?: number; // -1 to 1 (decreasing to increasing)
}

export const ReactorVisualization: React.FC<ReactorVisualizationProps> = ({
  state,
  rodPosition = 0,
  coolantFlow = 0,
}) => {
  if (!state) {
    return (
      <div className="flex h-full items-center justify-center rounded-lg border border-gray-700 bg-gray-800 p-8">
        <p className="text-center text-gray-400">Waiting for reactor state...</p>
      </div>
    );
  }

  // Calculate safety status colors
  const fuelTempPercent = Math.min(state.fuel_temp / 1100, 1);
  const pressurePercent = Math.min((state.pressure - 8) / 4, 1);
  const coolantTempPercent = Math.min(state.coolant_temp / 310, 1);

  const getTempColor = (percent: number) => {
    if (percent < 0.7) return "#10b981"; // Green
    if (percent < 0.85) return "#f59e0b"; // Amber
    return "#ef4444"; // Red
  };

  const fuelTempColor = getTempColor(fuelTempPercent);
  const coolantTempColor = getTempColor(coolantTempPercent);
  const pressureColor = getTempColor(pressurePercent);

  // Rod insertion visual position (SVG coordinates)
  const rodVisualPosition = 200 + ((1 - rodPosition) * 100) / 2; // 200-250 pixels

  return (
    <div className="rounded-lg border border-gray-700 bg-gray-800 p-6 overflow-hidden">
      {/* Title */}
      <div className="mb-6 flex items-center justify-between">
        <h3 className="text-lg font-bold text-white">🔥 Reactor Core Status</h3>
        <div className="flex items-center gap-2">
          <div className="h-3 w-3 animate-pulse rounded-full" style={{ backgroundColor: fuelTempColor }} />
          <span className="text-xs font-semibold text-gray-300">Live</span>
        </div>
      </div>

      {/* Main Reactor Visualization */}
      <svg viewBox="0 0 400 350" className="w-full h-64 mx-auto mb-6 drop-shadow-lg">
        {/* Reactor Vessel Background */}
        <defs>
          <linearGradient id="vesselGrad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" style={{ stopColor: "#1f2937", stopOpacity: 1 }} />
            <stop offset="100%" style={{ stopColor: "#0f172a", stopOpacity: 1 }} />
          </linearGradient>

          {/* Fuel heat gradient */}
          <radialGradient id="fuelGrad" cx="50%" cy="30%">
            <stop offset="0%" style={{ stopColor: fuelTempColor, stopOpacity: 0.9 }} />
            <stop offset="100%" style={{ stopColor: "#374151", stopOpacity: 0.3 }} />
          </radialGradient>

          {/* Coolant flow */}
          <linearGradient id="coolantGrad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" style={{ stopColor: "#60a5fa", stopOpacity: 0.8 }} />
            <stop offset="100%" style={{ stopColor: "#3b82f6", stopOpacity: 0.3 }} />
          </linearGradient>
        </defs>

        {/* Vessel Outline */}
        <rect x="80" y="30" width="240" height="260" fill="url(#vesselGrad)" stroke="#4b5563" strokeWidth="2" rx="8" />

        {/* Fuel Rod Zone (Core) */}
        <ellipse cx="200" cy="120" rx="70" ry="60" fill="url(#fuelGrad)" opacity={0.8} />

        {/* Core Indicator Text */}
        <text x="200" y="120" textAnchor="middle" dy=".3em" className="text-xs font-bold" fill="#fff" opacity="0.8">
          CORE
        </text>

        {/* Temperature Heat Ring */}
        <circle
          cx="200"
          cy="120"
          r={40 + fuelTempPercent * 30}
          fill="none"
          stroke={fuelTempColor}
          strokeWidth="2"
          opacity={0.6}
        />

        {/* Coolant Flow Lines (animated) */}
        <g opacity={Math.abs(coolantFlow) > 0.2 ? 0.7 : 0.3}>
          {/* Left inlet */}
          <line x1="80" y1="80" x2="100" y2="80" stroke={coolantTempColor} strokeWidth="3" />
          <line x1="80" y1="120" x2="100" y2="120" stroke={coolantTempColor} strokeWidth="3" />
          <line x1="80" y1="160" x2="100" y2="160" stroke={coolantTempColor} strokeWidth="3" />

          {/* Right outlet */}
          <line x1="300" y1="80" x2="320" y2="80" stroke={coolantTempColor} strokeWidth="3" />
          <line x1="300" y1="120" x2="320" y2="120" stroke={coolantTempColor} strokeWidth="3" />
          <line x1="300" y1="160" x2="320" y2="160" stroke={coolantTempColor} strokeWidth="3" />
        </g>

        {/* Control Rod (movable) */}
        <g>
          {/* Rod slider track */}
          <line x1="200" y1={rodVisualPosition - 40} x2="200" y2={rodVisualPosition + 80} stroke="#4b5563" strokeWidth="4" />

          {/* Rod position indicator */}
          <rect
            x="190"
            y={rodVisualPosition - 8}
            width="20"
            height="16"
            fill={rodPosition > 0.5 ? "#10b981" : rodPosition < -0.5 ? "#ef4444" : "#f59e0b"}
            stroke="#fff"
            strokeWidth="2"
            rx="2"
          />

          {/* Rod label and position */}
          <text x="220" y={rodVisualPosition + 3} className="text-xs font-bold" fill="#fff">
            {rodPosition > 0 ? "↓ INSERT" : rodPosition < 0 ? "↑ RETRACT" : "NEUTRAL"}
          </text>
        </g>

        {/* Power Output Indicator (bottom) */}
        <rect x="100" y="280" width="200" height="12" fill="#374151" stroke="#4b5563" strokeWidth="1" rx="4" />
        <rect
          x="100"
          y="280"
          width={200 * (state.power / 1.2)}
          height="12"
          fill={state.power > 0.8 ? "#10b981" : state.power < 0.4 ? "#ef4444" : "#f59e0b"}
          rx="4"
        />
        <text x="310" y="288" className="text-xs font-bold" fill="#fff">
          {(state.power * 100).toFixed(0)}%
        </text>
      </svg>

      {/* Parameters Display Grid */}
      <div className="grid grid-cols-4 gap-3 text-xs">
        {/* Fuel Temperature */}
        <div className="rounded-lg bg-gray-700 p-3 border-l-4" style={{ borderColor: fuelTempColor }}>
          <p className="text-gray-400 uppercase tracking-wide">Fuel Temp</p>
          <p className="text-lg font-bold" style={{ color: fuelTempColor }}>
            {state.fuel_temp.toFixed(0)}K
          </p>
          <p className="text-gray-500 text-xs mt-1">
            {fuelTempPercent < 0.7 ? "🟢 Safe" : fuelTempPercent < 0.85 ? "🟡 Caution" : "🔴 Critical"}
          </p>
        </div>

        {/* System Pressure */}
        <div className="rounded-lg bg-gray-700 p-3 border-l-4" style={{ borderColor: pressureColor }}>
          <p className="text-gray-400 uppercase tracking-wide">Pressure</p>
          <p className="text-lg font-bold" style={{ color: pressureColor }}>
            {state.pressure.toFixed(1)} bar
          </p>
          <p className="text-gray-500 text-xs mt-1">
            {state.pressure >= 8 && state.pressure <= 12 ? "🟢 Optimal" : "🟡 Adjust"}
          </p>
        </div>

        {/* Coolant Temperature */}
        <div className="rounded-lg bg-gray-700 p-3 border-l-4" style={{ borderColor: coolantTempColor }}>
          <p className="text-gray-400 uppercase tracking-wide">Coolant Temp</p>
          <p className="text-lg font-bold" style={{ color: coolantTempColor }}>
            {state.coolant_temp.toFixed(0)}K
          </p>
          <p className="text-gray-500 text-xs mt-1">
            {state.coolant_temp >= 280 && state.coolant_temp <= 310 ? "🟢 Stable" : "🟡 Monitor"}
          </p>
        </div>

        {/* Power Output */}
        <div className="rounded-lg bg-gray-700 p-3 border-l-4 border-blue-500">
          <p className="text-gray-400 uppercase tracking-wide">Power Out</p>
          <p className="text-lg font-bold text-blue-400">
            {state.power.toFixed(2)} MW
          </p>
          <p className="text-gray-500 text-xs mt-1">
            {state.power >= 0.8 && state.power <= 1.2 ? "🟢 Target" : "🟡 Off-target"}
          </p>
        </div>
      </div>

      {/* Control Status Footer */}
      <div className="mt-4 flex items-center justify-between text-xs text-gray-400 border-t border-gray-700 pt-4">
        <div>
          <span>Rod Position:</span>
          <span className="ml-2 font-bold text-white">
            {rodPosition > 0 ? "↓" : rodPosition < 0 ? "↑" : "—"} {(rodPosition * 100).toFixed(0)}%
          </span>
        </div>
        <div>
          <span>Coolant Flow:</span>
          <span className="ml-2 font-bold text-white">
            {coolantFlow > 0 ? "→ +↑" : coolantFlow < 0 ? "← -↓" : "→"} {(Math.abs(coolantFlow) * 100).toFixed(0)}%
          </span>
        </div>
        <div>
          <span>Simulation Time:</span>
          <span className="ml-2 font-bold text-white">{state.time.toFixed(1)}s</span>
        </div>
      </div>
    </div>
  );
};
