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

  const animateParticles = useCallback(
    (timestamp: number) => {
      if (timestamp - lastTimeRef.current < 25) {
        animFrameRef.current = requestAnimationFrame(animateParticles);
        return;
      }
      lastTimeRef.current = timestamp;

      // Coolant flow physics
      const flowEffect = Math.max(-1, Math.min(1, coolantFlow * 12));
      const baseSpeed = isRunning ? 2.4 : 0.0;
      const speedMult = 1 + flowEffect * 0.8;
      const spawnChance = isRunning ? Math.max(0.15, 0.6 + flowEffect * 0.4) : 0;

      setParticles((prev) => {
        let updated = prev
          .map((p) => {
            // Add slight turbulence (horizontal wiggle)
            const wiggle = Math.sin(p.y * 0.05 + timestamp * 0.01) * 0.5;
            
            return {
              ...p,
              x: p.x + wiggle * (isRunning ? 1 : 0),
              y: p.y - p.speed * baseSpeed * speedMult,
              // Fade out only once they go DEEP into the outlet pipe
              opacity:
                p.y < 50
                  ? p.opacity - 0.12
                  : isRunning
                  ? p.opacity
                  : p.opacity - 0.08,
            };
          })
          .filter((p) => p.opacity > 0.03 && p.y > 35 && p.y < 290);

        if (Math.random() < spawnChance && updated.length < 50) {
          const colX = COOLANT_COLUMNS[Math.floor(Math.random() * COOLANT_COLUMNS.length)];
          updated.push({
            id: particleIdRef.current++,
            x: colX + (Math.random() - 0.5) * 14,
            y: 275, // Spawn EARLIER in the inlet pipe
            opacity: 0.6 + Math.random() * 0.4,
            speed: 0.7 + Math.random() * 0.9,
            goingUp: true,
          });
        }

        return updated;
      });
      animFrameRef.current = requestAnimationFrame(animateParticles);
    },
    [coolantFlow, isRunning]
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

  const fuelTempPercent = Math.min(state.fuel_temp / 1300, 1);
  const coolantTempPercent = Math.min((state.coolant_temp - 273) / 100, 1); 
  
  const isCritical = state.fuel_temp > 1100;
  const isWarning = state.fuel_temp > 950 && !isCritical;
  const glowRadius = 45 + fuelTempPercent * 45;

  const bulkFluidColor = `hsla(${195 - coolantTempPercent * 15}, 100%, ${35 + coolantTempPercent * 15}%, ${0.12 + coolantTempPercent * 0.1})`;

  // Rod action indicator label
  const rodLabel =
    rodPosition > 0.005
      ? `↓ INSERT +${rodPosition.toFixed(4)}`
      : rodPosition < -0.005
      ? `↑ WITHDRAW ${rodPosition.toFixed(4)}`
      : "STABLE";
  const rodLabelColor =
    rodPosition > 0.005 ? "#ff5252" : rodPosition < -0.005 ? "#40c4ff" : "#ffd600";

  // Coolant action label
  const flowLabel =
    coolantFlow > 0.005
      ? `↑ BOOST +${coolantFlow.toFixed(4)}`
      : coolantFlow < -0.005
      ? `↓ REDUCE ${coolantFlow.toFixed(4)}`
      : "STEADY";
  const flowColor =
    coolantFlow > 0.005 ? "#00e676" : coolantFlow < -0.005 ? "#ffd600" : "#40c4ff";

  // Coolant flow line animation speed & opacity
  const flowLineOpacity = 0.12 + Math.abs(coolantFlow) * 0.5;
  const flowLineDuration = Math.max(0.3, 2.0 - Math.abs(coolantFlow) * 1.8);

  return (
    <div
      className="relative rounded-lg overflow-hidden"
      style={{
        background: "rgba(1, 4, 10, 0.98)",
        border: `1px solid ${isCritical ? "rgba(255,59,59,0.7)" : "rgba(0,212,255,0.25)"}`,
        boxShadow: isCritical
          ? "0 0 40px rgba(255,59,59,0.3), inset 0 0 30px rgba(255,59,59,0.05)"
          : "0 0 20px rgba(0,212,255,0.08)",
        animationName: isCritical ? "critical-pulse" : "none",
        animationDuration: "0.8s",
        animationIterationCount: "infinite",
        transitionProperty: "background, border-color, box-shadow",
        transitionDuration: "300ms",
        transitionTimingFunction: "ease",
      }}
    >
      {/* ── MISSION COMPLETE OVERLAY ── */}
      {!isRunning && state.time > 0 && (
        <div 
          className="absolute inset-0 z-20 flex flex-col items-center justify-center animate-in fade-in zoom-in duration-500"
          style={{ background: "rgba(2,12,28,0.88)", backdropFilter: "blur(5px)" }}
        >
          <div className="text-5xl mb-3">🛡️</div>
          <h2 className="text-2xl font-black tracking-[0.2em] uppercase mb-1" style={{ color: "#00d4ff", textShadow: "0 0 20px #00d4ff80" }}>
            Operation Secured
          </h2>
          <p className="text-[0.65rem] font-mono tracking-widest" style={{ color: "rgba(0,212,255,0.6)" }}>
            FINAL LOG AT T+{state.time.toFixed(2)}S
          </p>
        </div>
      )}

      {/* ── Header ── */}
      <div className="flex items-center justify-between px-4 py-2.5 border-b" style={{ borderColor: "rgba(0,212,255,0.12)" }}>
        <div className="flex items-center gap-2">
          <div className="text-xl" style={{
            filter: isCritical ? "drop-shadow(0 0 10px #ff3b3b)" : "drop-shadow(0 0 8px #00d4ff)",
            animationName: isRunning ? "fuel-pulse" : "none",
            animationDuration: "1.5s",
            animationTimingFunction: "ease-in-out",
            animationIterationCount: "infinite",
          }}>⚛️</div>
          <span className="section-label tracking-widest text-[#a0d8e8]">System Vitalization</span>
        </div>
        <div className="flex items-center gap-3 font-mono">
          <span style={{ color: rodLabelColor, fontSize: "0.62rem" }}>{rodLabel}</span>
          <div className="w-[1px] h-3 bg-white/5" />
          <span style={{ color: flowColor, fontSize: "0.62rem" }}>{flowLabel}</span>
          <div className="w-[1px] h-3 bg-white/5" />
          <div className={`${isCritical ? "led-red" : isWarning ? "led-yellow" : "led-green"}`} />
          <span style={{ color: isCritical ? "#ff3b3b" : isWarning ? "#ffd600" : "#00e676", fontSize: "0.65rem", fontWeight: "bold" }}>
            {isCritical ? "MELTDOWN RISK" : isWarning ? "THERMAL ALERT" : "STABLE"}
          </span>
        </div>
      </div>

      {/* ── Main SVG ── */}
      <svg viewBox="0 0 400 300" className="w-full" style={{ maxHeight: 300 }}>
        <defs>
          <radialGradient id="highHeat" cx="50%" cy="45%">
            <stop offset="0%"   stopColor={isCritical ? "#ff1111" : "#ffaa00"} stopOpacity="0.5" />
            <stop offset="70%"  stopColor={isCritical ? "#ff3300" : "#ff6600"} stopOpacity="0.1" />
            <stop offset="100%" stopColor="#000000" stopOpacity="0" />
          </radialGradient>
          <linearGradient id="coolantFlowGrad" x1="0%" y1="100%" x2="0%" y2="0%">
            <stop offset="0%"   stopColor="#004e92" stopOpacity="0.8" />
            <stop offset="50%"  stopColor="#00d4ff" stopOpacity="1" />
            <stop offset="100%" stopColor="#ffffff" stopOpacity="0.8" />
          </linearGradient>
          <filter id="fluidGlow">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
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

        {/* Reactor Vessel Boundary */}
        <rect x="70" y="63" width="260" height="200" rx="14" ry="14"
          fill="rgba(5, 15, 30, 0.95)" stroke="rgba(0,212,255,0.2)" strokeWidth="2" />

        {/* ── BULK FLUID OVERLAY (THE "INSIDE" FEEL) ── */}
        <rect x="72" y="65" width="256" height="196" rx="12" ry="12"
          fill={bulkFluidColor}
          style={{ 
            transitionProperty: "fill",
            transitionDuration: "0.5s",
            transitionTimingFunction: "ease",
            animationName: isRunning ? "fluid-pulse" : "none",
            animationDuration: "4s",
            animationTimingFunction: "ease-in-out",
            animationIterationCount: "infinite"
          }} />

        {/* Core Heat Glow */}
        <ellipse cx="200" cy="158" rx={glowRadius} ry={glowRadius * 0.7}
          fill="url(#highHeat)"
          style={{ 
            animationName: isRunning ? "fuel-pulse" : "none",
            animationDuration: "2s",
            animationTimingFunction: "ease-in-out",
            animationIterationCount: "infinite"
          }} />

        {/* ── Fuel Assemblies ── */}
        {FUEL_ASSEMBLIES.map(fa => {
          const color = getTempColor(state.fuel_temp, fa.weight);
          return (
            <g key={fa.id}>
              <circle cx={fa.x} cy={fa.y} r={9} fill={color}
                opacity={fa.weight * fuelTempPercent * 0.3} filter="url(#glow)" />
              <circle cx={fa.x} cy={fa.y} r={7.5} fill={color}
                opacity={0.6 + fa.weight * 0.4}
                stroke={color} strokeWidth="1"
                style={{
                  filter: `drop-shadow(0 0 ${4 + fa.weight * 8}px ${color})`,
                  animationName: isRunning ? "fuel-pulse" : "none",
                  animationDuration: `${1.2 + fa.weight}s`,
                  animationTimingFunction: "ease-in-out",
                  animationIterationCount: "infinite",
                  animationDelay: `${fa.weight * 0.5}s`,
                }} />
              <circle cx={fa.x} cy={fa.y} r={3.5} fill="white"
                opacity={fa.weight * fuelTempPercent * 0.8} />
            </g>
          );
        })}

        {/* ── Coolant Flow Columns ── */}
        {COOLANT_COLUMNS.map((x, i) => (
          <g key={`flow-${i}`} opacity={flowLineOpacity}>
            <line x1={x} y1={270} x2={x} y2={50}
              stroke="url(#coolantFlowGrad)"
              strokeWidth={1.5 + Math.abs(coolantFlow) * 2.5}
              strokeDasharray="10,15"
              style={{
                animationName: isRunning ? "coolant-flow" : "none",
                animationDuration: `${flowLineDuration}s`,
                animationTimingFunction: "linear",
                animationIterationCount: "infinite",
                animationDelay: `${i * 0.3}s`,
              }}
            />
          </g>
        ))}

        {/* ── Coolant Particles with Thermal Interpolation ── */}
        {particles.map(p => {
          // Calculate if particle is in the "hot zone" (middle of core)
          const distToCore = Math.abs(p.y - 158);
          const heatRatio = Math.max(0, 1 - distToCore / 80);
          const pColor = heatRatio > 0.4 ? "#ffffff" : "#40c4ff";
          
          return (
            <circle key={p.id} cx={p.x} cy={p.y} r={2.8}
              fill={pColor}
              opacity={p.opacity}
              style={{ 
                filter: `drop-shadow(0 0 ${2 + heatRatio * 4}px ${pColor})`,
                transition: "fill 0.2s ease"
              }}
            />
          );
        })}

        {/* ── Control Rods (3 rods) ── */}
        {[155, 200, 245].map((x, i) => (
          <g key={`rod-${i}`}>
            <line x1={x} y1="20" x2={x} y2={ROD_TOP}
              stroke="rgba(0,212,255,0.2)" strokeWidth="4" strokeLinecap="round" />
            <rect
              x={x - 4.5} y={20}
              width={9} height={Math.max(0, rodY - 20)}
              fill="url(#rodGrad)"
              stroke={rodPosition > 0.005 ? "#ff5252" : rodPosition < -0.005 ? "#40c4ff" : "rgba(0,212,255,0.4)"}
              strokeWidth="1.2"
              rx="2"
              style={{ transition: "height 0.6s cubic-bezier(0.34, 1.56, 0.64, 1)" }}
            />
            <rect
              x={x - 6} y={rodY - 6}
              width={12} height={12} rx="3"
              fill={rodPosition > 0.005 ? "#ff5252" : rodPosition < -0.005 ? "#40c4ff" : "#ffd600"}
              style={{
                filter: `drop-shadow(0 0 10px ${rodPosition > 0.005 ? "#ff5252" : rodPosition < -0.005 ? "#40c4ff" : "#ffd600"})`,
                transition: "all 0.3s ease"
              }}
            />
          </g>
        ))}

        {/* ── Coolant Plumbing (In/Out) ── */}
        {/* Inlet Pipe */}
        <path d="M 120 278 L 280 278" fill="none" stroke="#004e92" strokeWidth="6" strokeLinecap="round" opacity="0.4" />
        <path d="M 100 263 Q 100 278 120 278" fill="none" stroke="#00d4ff" strokeWidth="4" opacity={0.3 + flowLineOpacity} />
        <path d="M 300 263 Q 300 278 280 278" fill="none" stroke="#00d4ff" strokeWidth="4" opacity={0.3 + flowLineOpacity} />
        
        {/* Outlet Pipe */}
        <path d="M 130 50 L 270 50" fill="none" stroke="#d32f2f" strokeWidth="6" strokeLinecap="round" opacity="0.3" />
        <path d="M 115 63 Q 115 50 135 50" fill="none" stroke="#ff8f00" strokeWidth="4" opacity="0.4" />
        <path d="M 285 63 Q 285 50 265 50" fill="none" stroke="#ff8f00" strokeWidth="4" opacity="0.4" />

        <text x="200" y="292" textAnchor="middle" fontSize="9" fontWeight="bold"
          fill="#40c4ff" fontFamily="JetBrains Mono, monospace" style={{ opacity: 0.8 }}>
          PRIMARY INLET {isRunning ? ">>>" : ""}
        </text>
        <text x="200" y="40" textAnchor="middle" fontSize="9" fontWeight="bold"
          fill="#ff8f00" fontFamily="JetBrains Mono, monospace" style={{ opacity: 0.8 }}>
          PRIMARY OUTLET {isRunning ? ">>>" : ""}
        </text>

        {/* ── Power Performance Strip ── */}
        <g transform="translate(90, 275)">
          <rect width="220" height="10" rx="5" fill="rgba(255,255,255,0.05)" />
          <rect
            width={Math.min(220, 220 * (state.power / 1.5))} height="10" rx="5"
            fill={state.power > 1.2 ? "#ff3b3b" : state.power > 0.8 ? "#00e676" : "#ffd600"}
            style={{ 
              transitionProperty: "width",
              transitionDuration: "0.4s",
              transitionTimingFunction: "cubic-bezier(0.175, 0.885, 0.32, 1.275)",
              filter: `drop-shadow(0 0 8px ${state.power > 1.2 ? "#ff3b3b" : state.power > 0.8 ? "#00e676" : "#ffd600"})`
            }}
          />
          <text x="230" y="8" fontSize="10" fill="white/80" fontWeight="bold" fontFamily="JetBrains Mono">
            {(state.power * 100).toFixed(0)}%
          </text>
        </g>
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
              style={{ 
                color: item.color, 
                fontSize: "0.75rem", 
                textShadow: `0 0 8px ${item.color}60`, 
                transitionProperty: "color",
                transitionDuration: "0.3s",
                transitionTimingFunction: "ease"
              }}>
              {item.value}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ReactorVisualization;
