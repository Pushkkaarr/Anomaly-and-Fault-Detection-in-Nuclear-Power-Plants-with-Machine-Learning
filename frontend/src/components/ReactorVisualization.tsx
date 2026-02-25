"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";
import { ReactorState } from "@/types/reactor";

interface ReactorVisualizationProps {
  state: ReactorState | null;
  rodPosition?: number;  // -1 to 1
  coolantFlow?: number;  // -1 to 1
  isRunning?: boolean;
}

interface Particle {
  id: number;
  x: number;
  y: number;
  opacity: number;
  speed: number;
}

// Fuel assembly grid layout (hexagonal-ish 7×5)
const FUEL_ASSEMBLIES = (() => {
  const assemblies = [];
  const cols = 7;
  const rows = 5;
  const cx = 200;
  const cy = 155;
  const spacingX = 28;
  const spacingY = 26;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      // Skip corners to make it circular-ish
      const distFromCenter = Math.abs(c - 3) + Math.abs(r - 2);
      if (distFromCenter > 4) continue;

      const x = cx + (c - 3) * spacingX + (r % 2 === 1 ? spacingX / 2 : 0);
      const y = cy + (r - 2) * spacingY;
      // Weight for temperature distribution (center is hottest)
      const weight = 1 - distFromCenter / 5;
      assemblies.push({ x, y, weight, id: `${r}-${c}` });
    }
  }
  return assemblies;
})();

const getTempColor = (fuelTemp: number, weight: number): string => {
  const localTemp = fuelTemp * (0.7 + weight * 0.3);
  if (localTemp > 1100) return "#ff1a1a";
  if (localTemp > 1050) return "#ff6600";
  if (localTemp > 1000) return "#ff9900";
  if (localTemp > 950) return "#ffcc00";
  if (localTemp > 900) return "#aaff00";
  if (localTemp > 850) return "#00dd88";
  return "#00aacc";
};

const COOLANT_COLUMNS = [155, 180, 200, 220, 245];

export const ReactorVisualization: React.FC<ReactorVisualizationProps> = ({
  state,
  rodPosition = 0,
  coolantFlow = 0,
  isRunning = false,
}) => {
  const [particles, setParticles] = useState<Particle[]>([]);
  const [tick, setTick] = useState(0);
  const animFrameRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number>(0);
  const particleIdRef = useRef(0);

  // Animate coolant particles
  const animateParticles = useCallback((timestamp: number) => {
    if (timestamp - lastTimeRef.current < 40) {
      animFrameRef.current = requestAnimationFrame(animateParticles);
      return;
    }
    lastTimeRef.current = timestamp;

    const flowSpeed = 0.5 + Math.abs(coolantFlow) * 2.5;
    const spawnRate = isRunning ? (coolantFlow > -0.5 ? 0.45 : 0.2) : 0.1;

    setParticles(prev => {
      // Move existing particles upward
      let updated = prev
        .map(p => ({
          ...p,
          y: p.y - p.speed * flowSpeed,
          opacity: p.y < 100 ? p.opacity - 0.04 : p.opacity,
        }))
        .filter(p => p.y > 60 && p.opacity > 0.05);

      // Spawn new particles
      if (Math.random() < spawnRate && updated.length < 30) {
        const colX = COOLANT_COLUMNS[Math.floor(Math.random() * COOLANT_COLUMNS.length)];
        updated.push({
          id: particleIdRef.current++,
          x: colX + (Math.random() - 0.5) * 8,
          y: 240 + Math.random() * 20,
          opacity: 0.6 + Math.random() * 0.4,
          speed: 1.2 + Math.random() * 0.8,
        });
      }

      return updated;
    });

    setTick(t => t + 1);
    animFrameRef.current = requestAnimationFrame(animateParticles);
  }, [coolantFlow, isRunning]);

  useEffect(() => {
    animFrameRef.current = requestAnimationFrame(animateParticles);
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [animateParticles]);

  if (!state) {
    return (
      <div
        className="flex h-80 items-center justify-center rounded-lg"
        style={{ background: "rgba(2,8,18,0.8)", border: "1px solid rgba(0,212,255,0.12)" }}
      >
        <div className="text-center">
          <div className="text-5xl mb-4" style={{ opacity: 0.3 }}>⚛️</div>
          <p style={{ color: "rgba(0,212,255,0.5)", fontSize: 13 }}>
            Reactor offline — initialize simulation to begin
          </p>
        </div>
      </div>
    );
  }

  const fuelTempPercent = Math.min(state.fuel_temp / 1150, 1);
  const isCritical = state.fuel_temp > 1100;
  const isWarning = state.fuel_temp > 950 && !isCritical;

  // Control rod visual positions (3 rods)
  // rodPosition: -1 = fully inserted (rod covers core), +1 = fully withdrawn
  const rodInsertDepth = Math.max(0, Math.min(1, (1 - rodPosition) / 2));
  const rodY = 95 + rodInsertDepth * 80;

  // Glow radius based on temperature
  const glowRadius = 45 + fuelTempPercent * 35;

  return (
    <div
      className="relative rounded-lg overflow-hidden"
      style={{
        background: "rgba(2, 8, 18, 0.95)",
        border: `1px solid ${isCritical ? "rgba(255,59,59,0.6)" : "rgba(0,212,255,0.15)"}`,
        boxShadow: isCritical
          ? "0 0 20px rgba(255,59,59,0.3), inset 0 0 20px rgba(255,59,59,0.05)"
          : "0 0 10px rgba(0,212,255,0.08)",
        animation: isCritical ? "critical-pulse 1.2s ease-in-out infinite" : "none",
      }}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b" style={{ borderColor: "rgba(0,212,255,0.1)" }}>
        <div className="flex items-center gap-2">
          <div
            className="text-lg"
            style={{
              filter: isCritical ? "drop-shadow(0 0 8px #ff3b3b)" : "drop-shadow(0 0 6px #00d4ff)",
              animation: isRunning ? "fuel-pulse 1.5s ease-in-out infinite" : "none",
            }}
          >
            ⚛️
          </div>
          <span className="section-label">Reactor Core</span>
        </div>
        <div className="flex items-center gap-3">
          <div
            className={`${isCritical ? "led-red" : isWarning ? "led-yellow" : "led-green"}`}
          />
          <span
            className="text-xs font-bold"
            style={{
              color: isCritical ? "#ff3b3b" : isWarning ? "#ffd600" : "#00e676",
              fontFamily: "JetBrains Mono, monospace",
            }}
          >
            {isCritical ? "CRITICAL" : isWarning ? "CAUTION" : "NOMINAL"}
          </span>
        </div>
      </div>

      {/* Main SVG */}
      <svg viewBox="0 0 400 290" className="w-full" style={{ maxHeight: 290 }}>
        <defs>
          {/* Core glow gradient */}
          <radialGradient id="coreGlow" cx="50%" cy="45%">
            <stop offset="0%" stopColor={isCritical ? "#ff3b3b" : "#ff9900"} stopOpacity="0.4" />
            <stop offset="60%" stopColor={isCritical ? "#ff330020" : "#ff660010"} stopOpacity="0.2" />
            <stop offset="100%" stopColor="#020812" stopOpacity="0" />
          </radialGradient>

          {/* Coolant gradient */}
          <linearGradient id="coolantGrad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#40c4ff" stopOpacity="0.8" />
            <stop offset="100%" stopColor="#0d47a1" stopOpacity="0.3" />
          </linearGradient>

          {/* Vessel gradient */}
          <linearGradient id="vesselGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#0a1628" />
            <stop offset="100%" stopColor="#020812" />
          </linearGradient>

          {/* Control rod gradient */}
          <linearGradient id="rodGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#1a2a3a" />
            <stop offset="50%" stopColor="#2d4a5a" />
            <stop offset="100%" stopColor="#1a2a3a" />
          </linearGradient>

          {/* Scan line effect */}
          <filter id="glow">
            <feGaussianBlur stdDeviation="2" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* ─── Reactor Vessel ─── */}
        <rect
          x="70" y="60" width="260" height="200"
          rx="12" ry="12"
          fill="url(#vesselGrad)"
          stroke="rgba(0,212,255,0.15)"
          strokeWidth="1.5"
        />

        {/* Vessel inner lines */}
        {[80, 95, 110, 320, 305, 290].map((x, i) => (
          <line
            key={i}
            x1={i < 3 ? x : x} y1="60"
            x2={i < 3 ? x : x} y2="260"
            stroke={`rgba(0,212,255,0.04)`}
            strokeWidth="1"
          />
        ))}

        {/* ─── Core Glow ─── */}
        <ellipse
          cx="200" cy="155" rx={glowRadius} ry={glowRadius * 0.65}
          fill="url(#coreGlow)"
          style={{ animation: isRunning ? "fuel-pulse 2s ease-in-out infinite" : "none" }}
        />

        {/* ─── Fuel Assemblies ─── */}
        {FUEL_ASSEMBLIES.map((fa) => {
          const color = getTempColor(state.fuel_temp, fa.weight);
          const glowStrength = fa.weight * fuelTempPercent;
          return (
            <g key={fa.id}>
              {/* Glow behind */}
              <circle
                cx={fa.x} cy={fa.y} r={9}
                fill={color}
                opacity={glowStrength * 0.25}
                filter="url(#glow)"
              />
              {/* Fuel rod hexagon (approximated as circle) */}
              <circle
                cx={fa.x} cy={fa.y} r={7}
                fill={color}
                opacity={0.5 + fa.weight * 0.5}
                stroke={color}
                strokeWidth="0.5"
                style={{
                  filter: `drop-shadow(0 0 ${3 + fa.weight * 6}px ${color}80)`,
                  animationName: isRunning ? "fuel-pulse" : "none",
                  animationDuration: `${1.5 + fa.weight}s`,
                  animationTimingFunction: "ease-in-out",
                  animationIterationCount: "infinite",
                  animationDelay: `${fa.weight * 0.3}s`,
                }}
              />
              {/* Inner hot spot */}
              <circle
                cx={fa.x} cy={fa.y} r={3}
                fill="white"
                opacity={fa.weight * fuelTempPercent * 0.6}
              />
            </g>
          );
        })}

        {/* ─── Coolant Flow Lines ─── */}
        {COOLANT_COLUMNS.map((x, i) => (
          <g key={`flow-${i}`} opacity={0.15 + Math.abs(coolantFlow) * 0.3}>
            <line
              x1={x} y1="240" x2={x} y2="95"
              stroke="url(#coolantGrad)"
              strokeWidth={1 + Math.abs(coolantFlow) * 1.5}
              strokeDasharray="6,8"
              style={{
                animationName: "coolant-flow",
                animationDuration: `${2 - Math.abs(coolantFlow) * 0.8}s`,
                animationTimingFunction: "linear",
                animationIterationCount: "infinite",
                animationDelay: `${i * 0.3}s`,
              }}
            />
          </g>
        ))}

        {/* ─── Coolant Particles ─── */}
        {particles.map((p) => (
          <circle
            key={p.id}
            cx={p.x} cy={p.y} r={2}
            fill="#40c4ff"
            opacity={p.opacity}
            style={{ filter: "drop-shadow(0 0 3px #40c4ff)" }}
          />
        ))}

        {/* ─── Control Rods (3 rods) ─── */}
        {[155, 200, 245].map((x, i) => (
          <g key={`rod-${i}`}>
            {/* Rod track */}
            <line
              x1={x} y1="60" x2={x} y2="100"
              stroke="rgba(0,212,255,0.2)"
              strokeWidth="4"
            />
            {/* Rod body */}
            <rect
              x={x - 4} y={60}
              width={8}
              height={rodY - 60}
              fill="url(#rodGrad)"
              stroke="rgba(0,212,255,0.3)"
              strokeWidth="0.5"
              style={{ transition: "height 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94)" }}
            />
            {/* Rod tip */}
            <rect
              x={x - 5} y={rodY - 6}
              width={10} height={8}
              rx="2"
              fill={rodPosition < -0.2 ? "#ff3b3b" : rodPosition > 0.2 ? "#00e676" : "#ffd600"}
              style={{
                transition: "all 0.4s ease",
                filter: `drop-shadow(0 0 4px ${rodPosition < -0.2 ? "#ff3b3b" : rodPosition > 0.2 ? "#00e676" : "#ffd600"})`,
              }}
            />
            {/* "Enter core" indicator */}
            {rodY > 120 && (
              <line
                x1={x} y1={rodY} x2={x} y2={Math.min(200, rodY + 40)}
                stroke="rgba(0,212,255,0.15)"
                strokeWidth="2"
                strokeDasharray="3,4"
              />
            )}
          </g>
        ))}

        {/* ─── Coolant pipes (inlet/outlet) ─── */}
        {/* Inlet bottom */}
        <path
          d="M 100 260 Q 100 275 130 275 L 270 275 Q 300 275 300 260"
          fill="none" stroke="rgba(64,196,255,0.25)" strokeWidth="3"
        />
        {/* Outlet top */}
        <path
          d="M 115 60 Q 115 48 140 48 L 260 48 Q 285 48 285 60"
          fill="none" stroke="rgba(255,150,50,0.25)" strokeWidth="3"
        />
        {/* Inlet label */}
        <text x="200" y="285" textAnchor="middle" fontSize="8" fill="rgba(64,196,255,0.5)"
          fontFamily="JetBrains Mono, monospace">
          COOLANT IN
        </text>
        {/* Outlet label */}
        <text x="200" y="42" textAnchor="middle" fontSize="8" fill="rgba(255,150,50,0.5)"
          fontFamily="JetBrains Mono, monospace">
          COOLANT OUT
        </text>

        {/* ─── Power bar at bottom ─── */}
        <rect x="90" y="266" width="220" height="8" rx="4" fill="rgba(255,255,255,0.06)" />
        <rect
          x="90" y="266"
          width={Math.min(220, 220 * (state.power / 1.5))}
          height="8" rx="4"
          fill={state.power > 1.2 ? "#ff3b3b" : state.power > 0.8 ? "#00e676" : "#ffd600"}
          style={{
            transition: "width 0.3s ease",
            filter: `drop-shadow(0 0 4px ${state.power > 1.2 ? "#ff3b3b" : state.power > 0.8 ? "#00e676" : "#ffd600"})`,
          }}
        />
        <text x="318" y="274" fontSize="8.5" fill="rgba(255,255,255,0.6)"
          fontFamily="JetBrains Mono, monospace" textAnchor="start">
          {(state.power * 100).toFixed(0)}%
        </text>
        <text x="85" y="274" fontSize="8" fill="rgba(0,212,255,0.5)"
          fontFamily="JetBrains Mono, monospace" textAnchor="end">
          PWR
        </text>
      </svg>

      {/* ─── Bottom telemetry strip ─── */}
      <div
        className="grid grid-cols-4 gap-0 border-t"
        style={{ borderColor: "rgba(0,212,255,0.08)" }}
      >
        {[
          {
            label: "FUEL TEMP",
            value: `${state.fuel_temp.toFixed(0)}K`,
            color: isCritical ? "#ff3b3b" : isWarning ? "#ffd600" : "#00e676",
          },
          {
            label: "COOLANT",
            value: `${state.coolant_temp.toFixed(0)}K`,
            color: state.coolant_temp > 310 ? "#ffd600" : "#40c4ff",
          },
          {
            label: "PRESSURE",
            value: `${state.pressure.toFixed(1)} bar`,
            color: (state.pressure < 8 || state.pressure > 12) ? "#ffd600" : "#00e676",
          },
          {
            label: "SIM TIME",
            value: `${state.time.toFixed(1)}s`,
            color: "#6b8fa8",
          },
        ].map((item, i) => (
          <div
            key={i}
            className="flex flex-col items-center py-2 px-1"
            style={{
              borderRight: i < 3 ? "1px solid rgba(0,212,255,0.06)" : "none",
            }}
          >
            <span className="section-label" style={{ fontSize: "0.55rem" }}>{item.label}</span>
            <span
              className="font-mono font-bold mt-0.5"
              style={{
                color: item.color,
                fontSize: "0.75rem",
                textShadow: `0 0 8px ${item.color}60`,
                transition: "color 0.3s ease",
              }}
            >
              {item.value}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ReactorVisualization;
