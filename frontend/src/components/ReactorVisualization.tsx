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
  yCurrent: number;   // current visual Y (from bottom)
  yStart: number;     // spawn Y (bottom of vessel)
  yEnd: number;       // exit Y (top of vessel)
  opacity: number;
  speed: number;      // px per frame
  hue: number;        // color shifts as particle heats up
  lane: number;       // column index 0–4
}

// Fuel assembly grid layout
const FUEL_ASSEMBLIES = (() => {
  const assemblies = [];
  const cols = 7, rows = 5;
  const cx = 250, cy = 195;
  const spacingX = 32, spacingY = 30;
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
  if (t > 950)  return "#ffcc00";
  if (t > 900)  return "#aaff00";
  if (t > 850)  return "#00dd88";
  return "#00aacc";
};

// 5 coolant lane x-positions inside the vessel
const LANE_X = [185, 215, 250, 285, 315];
const VESSEL_TOP_Y = 80;     // particles exit here
const VESSEL_BOTTOM_Y = 320; // particles spawn here

// Rod geometry
const ROD_TOP_Y  = 25;   // top anchor (above vessel)
const ROD_CORE_Y = 240;  // when fully inserted (bottom of core)
const ROD_PARK_Y = 78;   // just above vessel when fully withdrawn

export const ReactorVisualization: React.FC<ReactorVisualizationProps> = ({
  state,
  rodPosition = 0,
  coolantFlow  = 0,
  isRunning    = false,
}) => {
  // ── Cumulative rod position (0 = fully withdrawn, 1 = fully inserted) ─────
  const cumulativeRodRef = useRef(0.35);
  const prevRunningRef   = useRef(false);

  useEffect(() => {
    if (!isRunning) {
      if (prevRunningRef.current) cumulativeRodRef.current = 0.35;
      prevRunningRef.current = false;
      return;
    }
    prevRunningRef.current = true;
    const SENSITIVITY = 0.35;
    cumulativeRodRef.current = Math.max(0, Math.min(1,
      cumulativeRodRef.current + rodPosition * SENSITIVITY
    ));
  }, [rodPosition, isRunning]);

  const cumulRod = cumulativeRodRef.current;
  // Y-tip of the rod in SVG coords
  const rodTipY = ROD_PARK_Y + cumulRod * (ROD_CORE_Y - ROD_PARK_Y);

  // ── Particle system (upward flowing coolant) ─────────────────────────────
  const [particles, setParticles] = useState<Particle[]>([]);
  const animFrameRef  = useRef<number | null>(null);
  const lastTimeRef   = useRef<number>(0);
  const particleIdRef = useRef(0);

  // Base speed from actual coolant flow value (if available)
  const actualFlowNorm = state?.coolant_flow_actual
    ? Math.min(state.coolant_flow_actual / 8000, 1.5)
    : 0.8;

  const animateParticles = useCallback(
    (timestamp: number) => {
      // Throttle to ~45fps for performance
      if (timestamp - lastTimeRef.current < 22) {
        animFrameRef.current = requestAnimationFrame(animateParticles);
        return;
      }
      lastTimeRef.current = timestamp;

      // Compute speed from flow action and running state
      // Action is -1 to +1. We shift it to be a multiplier.
      const actionMultiplier = 1.0 + coolantFlow * 0.8; 
      const speedMultiplier = isRunning ? (0.5 + actualFlowNorm * 2.5 * actionMultiplier) : 0;
      
      // Spawn rate also depends on flow
      const spawnRate = isRunning ? Math.min(0.95, 0.4 + actualFlowNorm * 0.3) : 0;

      setParticles(prev => {
        // Move all particles upward
        let updated = prev.map(p => {
          // Particles speed up as they heat up in the core
          const inCoreZone = p.yCurrent > 150 && p.yCurrent < 260;
          const heatSpeedBoost = inCoreZone ? 1.2 : 1.0;
          
          return {
            ...p,
            yCurrent: p.yCurrent - p.speed * speedMultiplier * heatSpeedBoost,
            // Fade out as they exit through top of vessel
            opacity: p.yCurrent < VESSEL_TOP_Y + 30
              ? Math.max(0, p.opacity - 0.1)
              : p.opacity,
          };
        }).filter(p => p.opacity > 0.02 && p.yCurrent > VESSEL_TOP_Y - 40);

        // Spawn new particles at the bottom
        // Max particles also scales with flow
        const maxParticles = 30 + Math.floor(actualFlowNorm * 50);
        
        if (Math.random() < spawnRate && updated.length < maxParticles) {
          const lane = Math.floor(Math.random() * LANE_X.length);
          updated.push({
            id:       particleIdRef.current++,
            x:        LANE_X[lane] + (Math.random() - 0.5) * 16,
            yCurrent: VESSEL_BOTTOM_Y + Math.random() * 10,
            yStart:   VESSEL_BOTTOM_Y,
            yEnd:     VESSEL_TOP_Y,
            opacity:  0.4 + Math.random() * 0.6,
            speed:    0.8 + Math.random() * 1.2,
            hue:      190,
            lane,
          });
        }

        return updated;
      });

      animFrameRef.current = requestAnimationFrame(animateParticles);
    },
    [coolantFlow, isRunning, actualFlowNorm]
  );

  useEffect(() => {
    animFrameRef.current = requestAnimationFrame(animateParticles);
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [animateParticles]);

  if (!state) {
    return (
      <div
        className="flex items-center justify-center rounded-xl"
        style={{
          background: "rgba(2,8,18,0.8)",
          border: "1px solid rgba(0,255,136,0.12)",
          minHeight: 420,
        }}
      >
        <div className="text-center">
          <div className="text-6xl mb-4" style={{ opacity: 0.25 }}>⚛️</div>
          <p style={{ color: "rgba(0,255,136,0.5)", fontSize: 14 }}>
            Reactor offline — select scenario and launch simulation
          </p>
        </div>
      </div>
    );
  }

  const fuelTempPercent    = Math.min(state.fuel_temp / 1300, 1);
  const coolantTempPercent = Math.min((state.coolant_temp - 273) / 100, 1);
  const isCritical         = state.fuel_temp > 1100;
  const isWarning          = state.fuel_temp > 950 && !isCritical;
  const glowRadius         = 55 + fuelTempPercent * 55;

  const bulkFluidColor = `hsla(${195 - coolantTempPercent * 18}, 100%, ${32 + coolantTempPercent * 18}%, ${0.10 + coolantTempPercent * 0.12})`;

  // Rod action indicator
  const rodLabel =
    rodPosition > 0.005
      ? `↓ INSERT  ${rodPosition.toFixed(3)}`
      : rodPosition < -0.005
      ? `↑ WITHDRAW ${rodPosition.toFixed(3)}`
      : "STABLE";
  const rodLabelColor =
    rodPosition > 0.005 ? "#fb2c36" : rodPosition < -0.005 ? "#9ca3af" : "#fbbf24";

  const flowLabel =
    coolantFlow > 0.005
      ? `↑ BOOST +${coolantFlow.toFixed(3)}`
      : coolantFlow < -0.005
      ? `↓ REDUCE ${coolantFlow.toFixed(3)}`
      : "STEADY";
  const flowColor =
    coolantFlow > 0.005 ? "var(--brand-accent)" : coolantFlow < -0.005 ? "#fbbf24" : "#9ca3af";

  return (
    <div
      className="relative rounded-xl overflow-hidden"
      style={{
        background: "rgba(1, 4, 10, 0.98)",
        border: `1px solid ${isCritical ? "rgba(255,59,59,0.7)" : "rgba(0,255,136,0.2)"}`,
        boxShadow: isCritical
          ? "0 0 50px rgba(255,59,59,0.3), inset 0 0 40px rgba(255,59,59,0.05)"
          : "0 0 24px rgba(0,255,136,0.06)",
        transitionProperty: "border-color, box-shadow",
        transitionDuration: "400ms",
        transitionTimingFunction: "ease",
      }}
    >
      {/* ── Mission Complete Overlay ── */}
      {!isRunning && state.time > 0 && (
        <div
          className="absolute inset-0 z-20 flex flex-col items-center justify-center"
          style={{ background: "rgba(2,12,28,0.88)", backdropFilter: "blur(6px)" }}
        >
          <div className="text-5xl mb-3">🛡️</div>
          <h2
            className="text-2xl font-black tracking-[0.18em] uppercase mb-1"
            style={{ color: "var(--brand-accent)", textShadow: "0 0 24px rgba(0,255,136,0.7)" }}
          >
            Simulation Complete
          </h2>
          <p className="text-xs font-mono tracking-widest" style={{ color: "rgba(0,255,136,0.5)" }}>
            FINAL LOG AT T+{state.time.toFixed(3)}S
          </p>
        </div>
      )}

      {/* ── Header ── */}
      <div
        className="flex items-center justify-between px-4 py-2.5 border-b"
        style={{ borderColor: "rgba(0,255,136,0.1)" }}
      >
        <div className="flex items-center gap-2">
          <div
            className="text-xl"
            style={{
              filter: isCritical
                ? "drop-shadow(0 0 12px #ff3b3b)"
                : "drop-shadow(0 0 8px var(--brand-accent))",
              animationName: isRunning ? "fuel-pulse" : "none",
              animationDuration: "1.5s",
              animationTimingFunction: "ease-in-out",
              animationIterationCount: "infinite",
            }}
          >
            ⚛️
          </div>
          <span
            className="text-xs font-bold tracking-widest uppercase"
            style={{ color: "#a0d8e8" }}
          >
            Primary Reactor Core
          </span>
        </div>
        <div className="flex items-center gap-3 font-mono">
          <span style={{ color: rodLabelColor, fontSize: "0.62rem" }}>{rodLabel}</span>
          <div className="w-px h-3 bg-white/10" />
          <span style={{ color: flowColor, fontSize: "0.62rem" }}>{flowLabel}</span>
          <div className="w-px h-3 bg-white/10" />
          <div
            className={isCritical ? "led-red" : isWarning ? "led-yellow" : "led-green"}
            style={{ width: 7, height: 7 }}
          />
          <span
            className="text-xs font-bold"
            style={{
              color: isCritical ? "#ff3b3b" : isWarning ? "#fbbf24" : "var(--brand-accent)",
            }}
          >
            {isCritical ? "MELTDOWN RISK" : isWarning ? "THERMAL ALERT" : "STABLE"}
          </span>
        </div>
      </div>

      {/* ── Main SVG (bigger viewBox) ── */}
      <svg
        viewBox="0 0 500 400"
        className="w-full"
        style={{ minHeight: 360 }}
      >
        <defs>
          {/* Core heat halo */}
          <radialGradient id="rv2-heat" cx="50%" cy="52%">
            <stop offset="0%"   stopColor={isCritical ? "#ff1111" : "#ffaa00"} stopOpacity="0.55" />
            <stop offset="65%"  stopColor={isCritical ? "#ff3300" : "#ff6600"} stopOpacity="0.12" />
            <stop offset="100%" stopColor="#000000" stopOpacity="0" />
          </radialGradient>
          {/* Coolant upward gradient */}
          <linearGradient id="rv2-coolant" x1="0%" y1="100%" x2="0%" y2="0%">
            <stop offset="0%"   stopColor="#004e92" stopOpacity="0.9" />
            <stop offset="45%"  stopColor="#00c4ff" stopOpacity="1" />
            <stop offset="100%" stopColor="#ffffff"  stopOpacity="0.9" />
          </linearGradient>
          {/* Rod gradient */}
          <linearGradient id="rv2-rod" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%"   stopColor="#1a2a3a" />
            <stop offset="50%"  stopColor="#2d4a6a" />
            <stop offset="100%" stopColor="#1a2a3a" />
          </linearGradient>
          {/* Glow filter */}
          <filter id="rv2-glow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="2.5" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          {/* Particle glow filter */}
          <filter id="rv2-pglow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="1.5" result="b" />
            <feMerge>
              <feMergeNode in="b" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          {/* Vessel sheen */}
          <linearGradient id="rv2-vessel-sheen" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%"   stopColor="rgba(0,255,136,0.06)" />
            <stop offset="50%"  stopColor="rgba(0,255,136,0.01)" />
            <stop offset="100%" stopColor="rgba(0,255,136,0.06)" />
          </linearGradient>
        </defs>

        {/* ── Outer Containment Structure ── */}
        <rect x="85" y="68" width="330" height="280" rx="18" ry="18"
          fill="rgba(5,15,30,0.98)"
          stroke={isCritical ? "rgba(255,59,59,0.5)" : "rgba(0,255,136,0.18)"}
          strokeWidth="2.5"
        />
        {/* Sheen overlay */}
        <rect x="87" y="70" width="326" height="276" rx="16" ry="16"
          fill="url(#rv2-vessel-sheen)"
        />

        {/* ── Bulk Fluid Overlay ── */}
        <rect x="87" y="70" width="326" height="276" rx="16" ry="16"
          fill={bulkFluidColor}
          style={{
            transitionProperty: "fill",
            transitionDuration: "0.6s",
            transitionTimingFunction: "ease",
          }}
        />

        {/* ── Core Heat Glow ── */}
        <ellipse cx="250" cy="205" rx={glowRadius} ry={glowRadius * 0.65}
          fill="url(#rv2-heat)"
          style={{
            animationName: isRunning ? "fuel-pulse" : "none",
            animationDuration: "2.2s",
            animationTimingFunction: "ease-in-out",
            animationIterationCount: "infinite",
          }}
        />

        {/* ── Fuel Assemblies ── */}
        {FUEL_ASSEMBLIES.map(fa => {
          const color = getTempColor(state.fuel_temp, fa.weight);
          return (
            <g key={fa.id} filter="url(#rv2-glow)">
              {/* Outer glow halo */}
              <circle cx={fa.x} cy={fa.y} r={12} fill={color}
                opacity={fa.weight * fuelTempPercent * 0.35}
              />
              {/* Main fuel pin */}
              <circle cx={fa.x} cy={fa.y} r={9.5} fill={color}
                opacity={0.65 + fa.weight * 0.35}
                stroke={color} strokeWidth="1.5"
                style={{
                  filter: `drop-shadow(0 0 ${5 + fa.weight * 10}px ${color})`,
                  animationName: isRunning ? "fuel-pulse" : "none",
                  animationDuration: `${1.2 + fa.weight}s`,
                  animationTimingFunction: "ease-in-out",
                  animationIterationCount: "infinite",
                  animationDelay: `${fa.weight * 0.5}s`,
                }}
              />
              {/* Hot center */}
              <circle cx={fa.x} cy={fa.y} r={4} fill="white"
                opacity={fa.weight * fuelTempPercent * 0.85}
              />
            </g>
          );
        })}

        {/* ── Coolant Lane Guide Lines (subtle) ── */}
        {LANE_X.map((x, i) => (
          <line key={`lane-${i}`}
            x1={x} y1={VESSEL_BOTTOM_Y}
            x2={x} y2={VESSEL_TOP_Y}
            stroke="rgba(0,196,255,0.06)"
            strokeWidth="6"
            strokeLinecap="round"
          />
        ))}

        {/* ── Coolant Particles (flowing UPWARD) ── */}
        {particles.map(p => {
          // Color shifts: blue (cool bottom) → white (hot middle zone) → slightly orange (heated)
          const progress = 1 - (p.yCurrent - VESSEL_TOP_Y) / (VESSEL_BOTTOM_Y - VESSEL_TOP_Y);
          const inCore = p.yCurrent > 160 && p.yCurrent < 250;
          const pColor = inCore
            ? `rgba(255,${200 + Math.floor((1 - progress) * 55)},${100 + Math.floor((1 - progress) * 100)},${p.opacity})`
            : `rgba(${60 + Math.floor(progress * 180)},${180 + Math.floor(progress * 75)},255,${p.opacity})`;

          return (
            <circle
              key={p.id}
              cx={p.x}
              cy={p.yCurrent}
              r={3}
              fill={pColor}
              filter="url(#rv2-pglow)"
            />
          );
        })}

        {/* ── Control Rods (3 rods) ── */}
        {[192, 250, 308].map((x, i) => {
          const tipY = rodTipY;
          const rodLen = tipY - ROD_TOP_Y;
          const rodColor =
            rodPosition > 0.005 ? "#fb2c36"
            : rodPosition < -0.005 ? "#9ca3af"
            : "rgba(0,255,136,0.5)";
          return (
            <g key={`rod-${i}`}>
              {/* Drive mechanism (above vessel) */}
              <rect
                x={x - 5}   y={ROD_TOP_Y}
                width={10}   height={Math.max(1, rodLen)}
                fill="url(#rv2-rod)"
                stroke={rodColor}
                strokeWidth="1.5" rx="3"
                style={{
                  transitionProperty: "height",
                  transitionDuration: "800ms",
                  transitionTimingFunction: "cubic-bezier(0.34,1.56,0.64,1)",
                }}
              />
              {/* Rod tip indicator */}
              <rect
                x={x - 7}   y={tipY - 7}
                width={14}   height={14} rx="4"
                fill={rodColor}
                style={{
                  filter: `drop-shadow(0 0 10px ${rodColor})`,
                  transitionProperty: "y, fill",
                  transitionDuration: "800ms",
                  transitionTimingFunction: "cubic-bezier(0.34,1.56,0.64,1)",
                }}
              />
              {/* Absorber section below tip (in core) */}
              <rect
                x={x - 4}   y={tipY + 7}
                width={8}   height={Math.max(0, ROD_CORE_Y - tipY - 7)} rx="2"
                fill="rgba(160,196,230,0.15)"
                stroke="rgba(160,196,230,0.08)"
                strokeWidth="1"
              />
            </g>
          );
        })}

        {/* ── Inlet Pipe (bottom) ── */}
        {/* Horizontal bottom pipe */}
        <path d="M 135 338 L 365 338" fill="none" stroke="#004e84" strokeWidth="8" strokeLinecap="round" opacity="0.5" />
        {/* Left elbow into vessel */}
        <path d="M 112 323 Q 112 338 135 338" fill="none" stroke="#00c4ff" strokeWidth="5" opacity={0.3 + Math.abs(coolantFlow) * 0.4} />
        {/* Right elbow */}
        <path d="M 388 323 Q 388 338 365 338" fill="none" stroke="#00c4ff" strokeWidth="5" opacity={0.3 + Math.abs(coolantFlow) * 0.4} />

        {/* ── Outlet Pipe (top) ── */}
        <path d="M 148 62 L 352 62" fill="none" stroke="#b04040" strokeWidth="8" strokeLinecap="round" opacity="0.4" />
        <path d="M 128 78 Q 128 62 148 62" fill="none" stroke="#ff8f00" strokeWidth="5" opacity="0.5" />
        <path d="M 372 78 Q 372 62 352 62" fill="none" stroke="#ff8f00" strokeWidth="5" opacity="0.5" />

        {/* ── Labels with Dynamic Intensity ── */}
        <g transform="translate(250, 392)">
           <text textAnchor="middle" fontSize="10" fontWeight="bold"
            fill={actualFlowNorm < 0.5 ? "#fbbf24" : "#6b8fa8"} 
            fontFamily="JetBrains Mono, monospace" opacity={isRunning ? 1 : 0.4}
          >
            PRIMARY COOLANT INLET  {isRunning ? (actualFlowNorm > 1.0 ? "▶▶▶▶" : actualFlowNorm > 0.5 ? "▶▶▶" : "▶") : ""}
          </text>
        </g>
        <g transform="translate(250, 52)">
          <text textAnchor="middle" fontSize="10" fontWeight="bold"
            fill={actualFlowNorm > 1.2 ? "#ff3b3b" : "#ff8f00"} 
            fontFamily="JetBrains Mono, monospace" opacity={isRunning ? 1 : 0.4}
          >
            {isRunning ? (actualFlowNorm > 1.0 ? "▶▶▶▶" : actualFlowNorm > 0.5 ? "▶▶▶" : "▶") : ""}  PRIMARY COOLANT OUTLET
          </text>
        </g>

        {/* ── Power Strip (bottom of vessel interior) ── */}
        <g transform="translate(105, 326)">
          <rect width="290" height="12" rx="6" fill="rgba(255,255,255,0.04)" />
          <rect
            width={Math.min(290, 290 * (state.power / 1.5))}
            height="12" rx="6"
            fill={state.power > 1.2 ? "#ff3b3b" : state.power > 0.8 ? "var(--brand-accent)" : "#fbbf24"}
            style={{
              transitionProperty: "width",
              transitionDuration: "0.5s",
              transitionTimingFunction: "cubic-bezier(0.175,0.885,0.32,1.275)",
              filter: `drop-shadow(0 0 10px ${state.power > 1.2 ? "#ff3b3b" : "var(--brand-accent)"})`,
            }}
          />
          <text x={Math.min(300, 290 * (state.power / 1.5) + 6)} y="9.5"
            fontSize="9" fill="rgba(255,255,255,0.7)" fontWeight="bold"
            fontFamily="JetBrains Mono"
          >
            {(state.power * 100).toFixed(3)}%
          </text>
        </g>

        {/* ── Rod Position Legend ── */}
        <g transform="translate(402, 90)">
          <rect width="88" height="200" rx="8" fill="rgba(0,0,0,0.5)" stroke="rgba(255,255,255,0.04)" strokeWidth="1" />
          <text x="44" y="16" textAnchor="middle" fontSize="8" fill="rgba(107,143,168,0.6)" fontFamily="JetBrains Mono" fontWeight="bold">
            ROD POSITION
          </text>
          {/* Track */}
          <rect x="39" y="22" width="10" height="165" rx="5" fill="rgba(255,255,255,0.05)" />
          {/* Rod position indicator */}
          <rect
            x="36" y={22 + cumulRod * 155}
            width="16" height="10" rx="3"
            fill={rodPosition > 0.005 ? "#fb2c36" : rodPosition < -0.005 ? "#9ca3af" : "#fbbf24"}
            style={{
              transitionProperty: "y",
              transitionDuration: "800ms",
              transitionTimingFunction: "cubic-bezier(0.34,1.56,0.64,1)",
              filter: `drop-shadow(0 0 6px ${rodPosition > 0.005 ? "#fb2c36" : "#fbbf24"})`,
            }}
          />
          <text x="44" y="195" textAnchor="middle" fontSize="7.5" fill="rgba(107,143,168,0.5)" fontFamily="JetBrains Mono">
            {(cumulRod * 100).toFixed(3)}% IN
          </text>
        </g>
      </svg>

      {/* ── Bottom telemetry strip ── */}
      <div
        className="grid grid-cols-4 gap-0 border-t"
        style={{ borderColor: "rgba(0,255,136,0.08)" }}
      >
        {[
          {
            label: "FUEL TEMP",
            value: `${state.fuel_temp.toFixed(3)}K`,
            color: isCritical ? "#ff3b3b" : isWarning ? "#fbbf24" : "var(--brand-accent)",
          },
          {
            label: "COOLANT",
            value: `${state.coolant_temp.toFixed(3)}K`,
            color: state.coolant_temp > 320 ? "#fbbf24" : "#9ca3af",
          },
          {
            label: "PRESSURE",
            value: `${state.pressure.toFixed(3)} bar`,
            color: state.pressure < 8 || state.pressure > 12 ? "#fbbf24" : "var(--brand-accent)",
          },
          {
            label: "SIM TIME",
            value: `${state.time.toFixed(3)}s`,
            color: "#6b8fa8",
          },
        ].map((item, i) => (
          <div
            key={i}
            className="flex flex-col items-center py-2.5 px-1"
            style={{
              borderRight: i < 3 ? "1px solid rgba(0,255,136,0.06)" : "none",
            }}
          >
            <span
              className="font-bold uppercase tracking-widest"
              style={{ color: "rgba(107,143,168,0.5)", fontSize: "0.52rem" }}
            >
              {item.label}
            </span>
            <span
              className="font-mono font-bold mt-0.5"
              style={{
                color: item.color,
                fontSize: "0.8rem",
                textShadow: `0 0 8px ${item.color}60`,
                transitionProperty: "color",
                transitionDuration: "0.3s",
                transitionTimingFunction: "ease",
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
