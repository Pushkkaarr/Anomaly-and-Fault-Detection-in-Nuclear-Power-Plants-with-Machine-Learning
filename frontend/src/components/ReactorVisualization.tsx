"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";
import { ReactorState } from "@/types/reactor";

interface ReactorVisualizationProps {
  state: ReactorState | null;
  rodPosition?: number;  // last AI action: -1 (withdraw) to +1 (insert)
  coolantFlow?: number;  // last AI action: -1 (reduce) to +1 (boost)
  isRunning?: boolean;
}

interface Particle {
  id: number;
  x: number;
  y: number;
  opacity: number;
  speed: number;
  goingUp: boolean; // flow direction
}

// Fuel assembly grid layout (hexagonal-ish 7×5)
const FUEL_ASSEMBLIES = (() => {
  const assemblies = [];
  const cols = 7, rows = 5;
  const cx = 200, cy = 155;
  const spacingX = 28, spacingY = 26;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const dist = Math.abs(c - 3) + Math.abs(r - 2);
      if (dist > 4) continue;
      const x = cx + (c - 3) * spacingX + (r % 2 === 1 ? spacingX / 2 : 0);
      const y = cy + (r - 2) * spacingY;
      const weight = 1 - dist / 5;
      assemblies.push({ x, y, weight, id: `${r}-${c}` });
    }
  }
  return assemblies;
})();

const getTempColor = (fuelTemp: number, weight: number): string => {
  const t = fuelTemp * (0.7 + weight * 0.3);
  if (t > 1100) return "#ff1a1a";
  if (t > 1050) return "#ff6600";
  if (t > 1000) return "#ff9900";
  if (t > 950) return "#ffcc00";
  if (t > 900) return "#aaff00";
  if (t > 850) return "#00dd88";
  return "#00aacc";
};

const COOLANT_COLUMNS = [155, 175, 200, 225, 245];

// Map accumulate rod position [0,1] to SVG y coordinate
// 0 = fully withdrawn (top, y=62), 1 = fully inserted (deep in core, y=195)
const ROD_TOP = 62;
const ROD_BOTTOM = 195;

export const ReactorVisualization: React.FC<ReactorVisualizationProps> = ({
  state,
  rodPosition = 0,
  coolantFlow = 0,
  isRunning = false,
}) => {
  // ── Cumulative rod position (integrate step actions) ────────────────
  // rodPosition prop = last action (-1=withdraw, +1=insert)
  // We integrate to show where the rods physically ARE, not just the delta
  const cumulativeRodRef = useRef(0.5); // start halfway inserted
  const prevRunningRef = useRef(false);

  useEffect(() => {
    if (!isRunning) {
      // Reset to halfway when sim not running
      if (prevRunningRef.current) cumulativeRodRef.current = 0.5;
      prevRunningRef.current = false;
      return;
    }
    prevRunningRef.current = true;
    // Integrate: positive action = insert more (move toward core)
    // action is small (±0.001–0.1), scale it for visibility
    const SENSITIVITY = 0.4; // how much a single action moves the rod visually
    cumulativeRodRef.current = Math.max(0, Math.min(1,
      cumulativeRodRef.current + rodPosition * SENSITIVITY
    ));
  }, [rodPosition, isRunning]);

  const cumulativeRod = cumulativeRodRef.current;
  const rodY = ROD_TOP + cumulativeRod * (ROD_BOTTOM - ROD_TOP);

  // ── Particle system ─────────────────────────────────────────────────
  const [particles, setParticles] = useState<Particle[]>([]);
  const animFrameRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number>(0);
  const particleIdRef = useRef(0);

  const animateParticles = useCallback((timestamp: number) => {
    if (timestamp - lastTimeRef.current < 40) {
      animFrameRef.current = requestAnimationFrame(animateParticles);
      return;
    }
    lastTimeRef.current = timestamp;

    // Flow direction: positive coolantFlow = boost (particles go up faster)
    //                 negative coolantFlow = reduce (particles go up slower or reverse)
    const flowBoost = coolantFlow;              // -1 to +1
    const baseSpeed = isRunning ? 1.5 : 0.4;
    const speedMult = 1 + flowBoost;            // 0 (reduce) to 2 (boost)
    const goingUp = flowBoost >= -0.3;        // reverse if heavily reducing
    const spawnRate = isRunning
      ? Math.max(0.05, 0.5 + flowBoost * 0.4)    // more particles = high flow
      : 0.08;

    setParticles(prev => {
      let updated = prev
        .map(p => ({
          ...p,
          y: p.goingUp
            ? p.y - p.speed * baseSpeed * speedMult
            : p.y + p.speed * baseSpeed * 0.5,   // slow drift down when reducing
          opacity: (p.goingUp ? p.y < 90 : p.y > 265)
            ? p.opacity - 0.06
            : p.opacity,
        }))
        .filter(p => p.opacity > 0.05 && p.y > 55 && p.y < 285);

      if (Math.random() < spawnRate && updated.length < 35) {
        const colX = COOLANT_COLUMNS[Math.floor(Math.random() * COOLANT_COLUMNS.length)];
        updated.push({
          id: particleIdRef.current++,
          x: colX + (Math.random() - 0.5) * 10,
          // spawn from bottom when flowing up, from top when reversing
          y: goingUp ? 248 + Math.random() * 15 : 75 + Math.random() * 15,
          opacity: 0.5 + Math.random() * 0.5,
          speed: 0.9 + Math.random() * 0.8,
          goingUp,
        });
      }

      return updated;
    });

    animFrameRef.current = requestAnimationFrame(animateParticles);
  }, [coolantFlow, isRunning]);

  useEffect(() => {
    animFrameRef.current = requestAnimationFrame(animateParticles);
    return () => { if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current); };
  }, [animateParticles]);

  // ── Idle / offline screen ───────────────────────────────────────────
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
  const glowRadius = 45 + fuelTempPercent * 35;

  // Rod action indicator label
  const rodLabel = rodPosition > 0.01 ? `↓ INSERT +${rodPosition.toFixed(3)}`
    : rodPosition < -0.01 ? `↑ WITHDRAW ${rodPosition.toFixed(3)}`
      : "HOLD";
  const rodLabelColor = rodPosition > 0.01 ? "#ff5252"
    : rodPosition < -0.01 ? "#40c4ff"
      : "#ffd600";

  // Coolant action label
  const flowLabel = coolantFlow > 0.01 ? `↑ BOOST +${coolantFlow.toFixed(3)}`
    : coolantFlow < -0.01 ? `↓ REDUCE ${coolantFlow.toFixed(3)}`
      : "NOMINAL";
  const flowColor = coolantFlow > 0.01 ? "#00e676"
    : coolantFlow < -0.01 ? "#ffd600"
      : "#40c4ff";

  // Coolant flow line speed & opacity based on action
  const flowLineOpacity = 0.08 + Math.abs(coolantFlow) * 0.35;
  const flowLineDuration = Math.max(0.4, 2.5 - Math.abs(coolantFlow) * 2);

  return (
    <div
      className="relative rounded-lg overflow-hidden"
      style={{
        background: "rgba(2, 8, 18, 0.95)",
        border: `1px solid ${isCritical ? "rgba(255,59,59,0.6)" : "rgba(0,212,255,0.15)"}`,
        boxShadow: isCritical
          ? "0 0 20px rgba(255,59,59,0.3), inset 0 0 20px rgba(255,59,59,0.05)"
          : "0 0 10px rgba(0,212,255,0.08)",
        animationName: isCritical ? "critical-pulse" : "none",
        animationDuration: "1.2s",
        animationTimingFunction: "ease-in-out",
        animationIterationCount: "infinite",
      }}
    >
      {/* ── Header ── */}
      <div className="flex items-center justify-between px-4 py-2 border-b" style={{ borderColor: "rgba(0,212,255,0.1)" }}>
        <div className="flex items-center gap-2">
          <div className="text-lg" style={{
            filter: isCritical ? "drop-shadow(0 0 8px #ff3b3b)" : "drop-shadow(0 0 6px #00d4ff)",
            animationName: isRunning ? "fuel-pulse" : "none",
            animationDuration: "1.5s",
            animationTimingFunction: "ease-in-out",
            animationIterationCount: "infinite",
          }}>⚛️</div>
          <span className="section-label">Reactor Core</span>
        </div>
        <div className="flex items-center gap-3 text-xs font-mono">
          {/* Rod action readout */}
          <span style={{ color: rodLabelColor, fontSize: "0.6rem" }}>{rodLabel}</span>
          <span style={{ color: "rgba(0,212,255,0.2)" }}>|</span>
          {/* Flow action readout */}
          <span style={{ color: flowColor, fontSize: "0.6rem" }}>{flowLabel}</span>
          <span style={{ color: "rgba(0,212,255,0.2)" }}>|</span>
          <div className={`${isCritical ? "led-red" : isWarning ? "led-yellow" : "led-green"}`} />
          <span style={{ color: isCritical ? "#ff3b3b" : isWarning ? "#ffd600" : "#00e676", fontFamily: "JetBrains Mono, monospace", fontSize: "0.65rem" }}>
            {isCritical ? "CRITICAL" : isWarning ? "CAUTION" : "NOMINAL"}
          </span>
        </div>
      </div>

      {/* ── Main SVG ── */}
      <svg viewBox="0 0 400 295" className="w-full" style={{ maxHeight: 295 }}>
        <defs>
          <radialGradient id="coreGlow" cx="50%" cy="45%">
            <stop offset="0%" stopColor={isCritical ? "#ff3b3b" : "#ff9900"} stopOpacity="0.4" />
            <stop offset="60%" stopColor={isCritical ? "#ff330020" : "#ff660010"} stopOpacity="0.2" />
            <stop offset="100%" stopColor="#020812" stopOpacity="0" />
          </radialGradient>
          <linearGradient id="coolantGrad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#40c4ff" stopOpacity="0.9" />
            <stop offset="100%" stopColor="#0d47a1" stopOpacity="0.2" />
          </linearGradient>
          {/* Reversed coolant gradient for reduce-flow */}
          <linearGradient id="coolantGradRev" x1="0%" y1="100%" x2="0%" y2="0%">
            <stop offset="0%" stopColor="#ffd600" stopOpacity="0.9" />
            <stop offset="100%" stopColor="#ff6d00" stopOpacity="0.2" />
          </linearGradient>
          <linearGradient id="vesselGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#0a1628" />
            <stop offset="100%" stopColor="#020812" />
          </linearGradient>
          <linearGradient id="rodGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#1a2a3a" />
            <stop offset="50%" stopColor="#2d4a5a" />
            <stop offset="100%" stopColor="#1a2a3a" />
          </linearGradient>
          <filter id="glow">
            <feGaussianBlur stdDeviation="2" result="coloredBlur" />
            <feMerge><feMergeNode in="coloredBlur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>

        {/* Reactor Vessel */}
        <rect x="70" y="63" width="260" height="200" rx="12" ry="12"
          fill="url(#vesselGrad)" stroke="rgba(0,212,255,0.15)" strokeWidth="1.5" />

        {/* Core Glow */}
        <ellipse cx="200" cy="158" rx={glowRadius} ry={glowRadius * 0.65}
          fill="url(#coreGlow)"
          style={{
            animationName: isRunning ? "fuel-pulse" : "none",
            animationDuration: "2s",
            animationTimingFunction: "ease-in-out",
            animationIterationCount: "infinite",
          }} />

        {/* ── Fuel Assemblies ── */}
        {FUEL_ASSEMBLIES.map(fa => {
          const color = getTempColor(state.fuel_temp, fa.weight);
          return (
            <g key={fa.id}>
              <circle cx={fa.x} cy={fa.y} r={9} fill={color}
                opacity={fa.weight * fuelTempPercent * 0.25} filter="url(#glow)" />
              <circle cx={fa.x} cy={fa.y} r={7} fill={color}
                opacity={0.5 + fa.weight * 0.5}
                stroke={color} strokeWidth="0.5"
                style={{
                  filter: `drop-shadow(0 0 ${3 + fa.weight * 6}px ${color}80)`,
                  animationName: isRunning ? "fuel-pulse" : "none",
                  animationDuration: `${1.5 + fa.weight}s`,
                  animationTimingFunction: "ease-in-out",
                  animationIterationCount: "infinite",
                  animationDelay: `${fa.weight * 0.3}s`,
                }} />
              <circle cx={fa.x} cy={fa.y} r={3} fill="white"
                opacity={fa.weight * fuelTempPercent * 0.6} />
            </g>
          );
        })}

        {/* ── Coolant Flow Lines ── react to coolantFlow action ── */}
        {COOLANT_COLUMNS.map((x, i) => {
          const isReducing = coolantFlow < -0.03;
          return (
            <g key={`flow-${i}`} opacity={flowLineOpacity}>
              <line
                x1={x} y1={isReducing ? 73 : 248}
                x2={x} y2={isReducing ? 248 : 73}
                stroke={`url(#${isReducing ? "coolantGradRev" : "coolantGrad"})`}
                strokeWidth={1 + Math.abs(coolantFlow) * 2}
                strokeDasharray="5,8"
                style={{
                  animationName: "coolant-flow",
                  animationDuration: `${flowLineDuration}s`,
                  animationTimingFunction: "linear",
                  animationIterationCount: "infinite",
                  animationDelay: `${i * 0.25}s`,
                }}
              />
            </g>
          );
        })}

        {/* ── Coolant Particles ── direction-aware ── */}
        {particles.map(p => (
          <circle key={p.id} cx={p.x} cy={p.y} r={2.5}
            fill={p.goingUp ? "#40c4ff" : "#ffd600"}
            opacity={p.opacity}
            style={{ filter: `drop-shadow(0 0 3px ${p.goingUp ? "#40c4ff" : "#ffd600"})` }}
          />
        ))}

        {/* ── Control Rods (3 rods) — position driven by cumulative action ── */}
        {/* Track lines (extend above vessel) */}
        {[155, 200, 245].map((x, i) => (
          <g key={`rod-${i}`}>
            {/* Above-vessel track */}
            <line x1={x} y1="30" x2={x} y2={ROD_TOP}
              stroke="rgba(0,212,255,0.15)" strokeWidth="3" />
            {/* Rod body — height = how far inserted */}
            <rect
              x={x - 4} y={30}
              width={8} height={Math.max(0, rodY - 30)}
              fill="url(#rodGrad)"
              stroke={rodPosition > 0.01 ? "rgba(255,82,82,0.5)" : rodPosition < -0.01 ? "rgba(64,196,255,0.5)" : "rgba(0,212,255,0.2)"}
              strokeWidth="0.8"
              style={{ transition: "height 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94)" }}
            />
            {/* Rod tip — glows based on action */}
            <rect
              x={x - 5} y={rodY - 5}
              width={10} height={10} rx="2"
              fill={rodPosition > 0.01 ? "#ff5252" : rodPosition < -0.01 ? "#40c4ff" : "#ffd600"}
              style={{
                transition: "fill 0.3s ease",
                filter: `drop-shadow(0 0 5px ${rodPosition > 0.01 ? "#ff5252" : rodPosition < -0.01 ? "#40c4ff" : "#ffd600"})`,
              }}
            />
            {/* Depth indicator line into core */}
            {rodY > ROD_TOP + 10 && (
              <line x1={x} y1={rodY} x2={x} y2={Math.min(220, rodY + 45)}
                stroke="rgba(64,196,255,0.12)" strokeWidth="2" strokeDasharray="3,5" />
            )}
          </g>
        ))}

        {/* Rod position label */}
        <text x="200" y="58" textAnchor="middle" fontSize="7"
          fill={rodLabelColor} fontFamily="JetBrains Mono, monospace" opacity="0.8">
          RODS: {(cumulativeRod * 100).toFixed(0)}% INSERTED
        </text>

        {/* ── Coolant pipes (inlet/outlet) ── */}
        <path d="M 100 263 Q 100 278 130 278 L 270 278 Q 300 278 300 263"
          fill="none" stroke={`rgba(64,196,255,${0.15 + Math.max(0, coolantFlow) * 0.4})`} strokeWidth="3" />
        <path d="M 115 63 Q 115 51 140 51 L 260 51 Q 285 51 285 63"
          fill="none" stroke="rgba(255,150,50,0.25)" strokeWidth="3" />
        <text x="200" y="288" textAnchor="middle" fontSize="7.5"
          fill={`rgba(64,196,255,${0.4 + Math.max(0, coolantFlow) * 0.5})`}
          fontFamily="JetBrains Mono, monospace">
          COOLANT IN {coolantFlow > 0.01 ? "↑↑" : coolantFlow < -0.01 ? "↓↓" : ""}
        </text>
        <text x="200" y="46" textAnchor="middle" fontSize="7.5"
          fill="rgba(255,150,50,0.5)" fontFamily="JetBrains Mono, monospace">
          COOLANT OUT
        </text>

        {/* ── Power bar ── */}
        <rect x="90" y="268" width="220" height="8" rx="4" fill="rgba(255,255,255,0.06)" />
        <rect
          x="90" y="268"
          width={Math.min(220, 220 * (state.power / 1.5))} height="8" rx="4"
          fill={state.power > 1.2 ? "#ff3b3b" : state.power > 0.8 ? "#00e676" : "#ffd600"}
          style={{
            transition: "width 0.3s ease",
            filter: `drop-shadow(0 0 4px ${state.power > 1.2 ? "#ff3b3b" : state.power > 0.8 ? "#00e676" : "#ffd600"})`,
          }}
        />
        <text x="318" y="276" fontSize="8.5" fill="rgba(255,255,255,0.6)"
          fontFamily="JetBrains Mono, monospace" textAnchor="start">
          {(state.power * 100).toFixed(0)}%
        </text>
        <text x="85" y="276" fontSize="7.5" fill="rgba(0,212,255,0.5)"
          fontFamily="JetBrains Mono, monospace" textAnchor="end">PWR</text>
      </svg>

      {/* ── Bottom telemetry strip ── */}
      <div className="grid grid-cols-4 gap-0 border-t" style={{ borderColor: "rgba(0,212,255,0.08)" }}>
        {[
          { label: "FUEL TEMP", value: `${state.fuel_temp.toFixed(0)}K`, color: isCritical ? "#ff3b3b" : isWarning ? "#ffd600" : "#00e676" },
          { label: "COOLANT", value: `${state.coolant_temp.toFixed(0)}K`, color: state.coolant_temp > 310 ? "#ffd600" : "#40c4ff" },
          { label: "PRESSURE", value: `${state.pressure.toFixed(1)} bar`, color: (state.pressure < 8 || state.pressure > 12) ? "#ffd600" : "#00e676" },
          { label: "SIM TIME", value: `${state.time.toFixed(1)}s`, color: "#6b8fa8" },
        ].map((item, i) => (
          <div key={i} className="flex flex-col items-center py-2 px-1"
            style={{ borderRight: i < 3 ? "1px solid rgba(0,212,255,0.06)" : "none" }}>
            <span className="section-label" style={{ fontSize: "0.55rem" }}>{item.label}</span>
            <span className="font-mono font-bold mt-0.5"
              style={{ color: item.color, fontSize: "0.75rem", textShadow: `0 0 8px ${item.color}60`, transition: "color 0.3s ease" }}>
              {item.value}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ReactorVisualization;
