"use client";

import React from "react";
import {
    ComposedChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
    ReferenceLine,
} from "recharts";
import { ReactorState } from "@/types/reactor";

interface LiveGraphsProps {
    history: ReactorState[];
    isRunning: boolean;
}

// Safety thresholds for reference lines
const FUEL_TEMP_WARN = 950;
const POWER_WARN_HI = 120; // % (1.2 × 100)
const POWER_WARN_LO = 80;  // %

const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload?.length) return null;
    return (
        <div
            style={{
                background: "rgba(2,8,18,0.97)",
                border: "1px solid rgba(0,212,255,0.25)",
                borderRadius: 6,
                padding: "8px 12px",
                fontSize: 11,
                fontFamily: "'JetBrains Mono', monospace",
                minWidth: 170,
            }}
        >
            <p style={{ color: "rgba(0,212,255,0.55)", marginBottom: 5 }}>
                t = {typeof label === "number" ? label.toFixed(1) : label}s
            </p>
            {payload.map((p: any) => (
                <p key={p.dataKey} style={{ color: p.color, margin: "2px 0" }}>
                    {p.name}: {typeof p.value === "number" ? p.value.toFixed(2) : p.value}
                    {p.name.includes("Temp") ? " K" : p.name.includes("Power") ? " %" : ""}
                </p>
            ))}
        </div>
    );
};

const AXIS_STYLE = {
    fontSize: 9,
    fill: "rgba(107,143,168,0.7)",
    fontFamily: "'JetBrains Mono', monospace",
};

export const LiveGraphs: React.FC<LiveGraphsProps> = ({ history, isRunning }) => {
    if (!history || history.length === 0) {
        return (
            <div
                className="flex h-44 items-center justify-center rounded-lg"
                style={{ background: "rgba(0,0,0,0.2)", border: "1px dashed rgba(0,212,255,0.1)" }}
            >
                <div className="text-center">
                    <div className="text-2xl mb-2 opacity-20">📈</div>
                    <p className="text-xs" style={{ color: "rgba(0,212,255,0.35)" }}>
                        {isRunning ? "Collecting data…" : "Start simulation to view live graphs"}
                    </p>
                </div>
            </div>
        );
    }

    // Build chart data — power as % so it fits a sensible 0–150% secondary axis
    const chartData = history.map((s) => ({
        time: parseFloat(s.time.toFixed(1)),
        fuelTemp: parseFloat(s.fuel_temp.toFixed(1)),
        coolantTemp: parseFloat(s.coolant_temp.toFixed(1)),
        powerPct: parseFloat((s.power * 100).toFixed(1)),
    }));

    // Compute tight domains from actual data (+/- 5% margin)
    const minFuel = Math.max(0, Math.min(...chartData.map(d => d.fuelTemp)) - 10);
    const maxFuel = Math.max(...chartData.map(d => d.fuelTemp)) + 10;
    const minCoolant = Math.max(0, Math.min(...chartData.map(d => d.coolantTemp)) - 5);
    const maxCoolant = Math.max(...chartData.map(d => d.coolantTemp)) + 5;
    const minPwr = Math.max(0, Math.min(...chartData.map(d => d.powerPct)) - 5);
    const maxPwr = Math.max(...chartData.map(d => d.powerPct)) + 5;

    // Use widest temp range so both temp lines share the same left axis
    const leftMin = Math.min(minFuel, minCoolant);
    const leftMax = Math.max(maxFuel, maxCoolant);

    return (
        <div className="w-full space-y-1">
            {/* TEMPERATURE chart */}
            <p className="section-label" style={{ fontSize: "0.58rem" }}>
                Temperature (K) — last {history.length} samples
            </p>
            <ResponsiveContainer width="100%" height={130}>
                <ComposedChart data={chartData} margin={{ top: 4, right: 12, bottom: 0, left: -8 }}>
                    <CartesianGrid strokeDasharray="2 4" stroke="rgba(0,212,255,0.05)" />
                    <XAxis
                        dataKey="time"
                        tick={AXIS_STYLE}
                        tickFormatter={(v) => `${Number(v).toFixed(0)}s`}
                        stroke="rgba(0,212,255,0.1)"
                        tickLine={false}
                        interval="preserveStartEnd"
                    />
                    <YAxis
                        domain={[leftMin, leftMax]}
                        tick={AXIS_STYLE}
                        stroke="rgba(0,212,255,0.1)"
                        tickLine={false}
                        width={42}
                        tickFormatter={(v) => `${v.toFixed(0)}K`}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    {/* Fuel temp warning line */}
                    {leftMax > FUEL_TEMP_WARN && (
                        <ReferenceLine
                            y={FUEL_TEMP_WARN}
                            stroke="rgba(255,59,59,0.35)"
                            strokeDasharray="4 3"
                            label={{ value: "⚠ 950K", fill: "rgba(255,59,59,0.6)", fontSize: 8, position: "right" }}
                        />
                    )}
                    <Line
                        type="monotone" dataKey="fuelTemp" name="Fuel Temp"
                        stroke="#ff5252" strokeWidth={1.5} dot={false}
                        isAnimationActive={false}
                    />
                    <Line
                        type="monotone" dataKey="coolantTemp" name="Coolant Temp"
                        stroke="#40c4ff" strokeWidth={1.5} dot={false}
                        isAnimationActive={false}
                    />
                    <Legend
                        wrapperStyle={{ fontSize: 9, color: "rgba(107,143,168,0.8)", paddingTop: 2 }}
                        formatter={(v) => <span style={{ color: v === "Fuel Temp" ? "#ff5252" : "#40c4ff" }}>{v}</span>}
                    />
                </ComposedChart>
            </ResponsiveContainer>

            {/* POWER chart */}
            <p className="section-label" style={{ fontSize: "0.58rem", marginTop: 8 }}>
                Reactor Power (%)
            </p>
            <ResponsiveContainer width="100%" height={90}>
                <ComposedChart data={chartData} margin={{ top: 4, right: 12, bottom: 0, left: -8 }}>
                    <CartesianGrid strokeDasharray="2 4" stroke="rgba(0,212,255,0.05)" />
                    <XAxis
                        dataKey="time"
                        tick={AXIS_STYLE}
                        tickFormatter={(v) => `${Number(v).toFixed(0)}s`}
                        stroke="rgba(0,212,255,0.1)"
                        tickLine={false}
                        interval="preserveStartEnd"
                    />
                    <YAxis
                        domain={[Math.max(0, minPwr), maxPwr]}
                        tick={AXIS_STYLE}
                        stroke="rgba(0,212,255,0.1)"
                        tickLine={false}
                        width={42}
                        tickFormatter={(v) => `${v.toFixed(0)}%`}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <ReferenceLine y={POWER_WARN_HI} stroke="rgba(255,59,59,0.3)" strokeDasharray="4 3" />
                    <ReferenceLine y={POWER_WARN_LO} stroke="rgba(255,214,0,0.3)" strokeDasharray="4 3" />
                    <Line
                        type="monotone" dataKey="powerPct" name="Power %"
                        stroke="#69ff47" strokeWidth={1.5} dot={false}
                        isAnimationActive={false}
                    />
                    <Legend
                        wrapperStyle={{ fontSize: 9, color: "rgba(107,143,168,0.8)", paddingTop: 2 }}
                        formatter={() => <span style={{ color: "#69ff47" }}>Power %</span>}
                    />
                </ComposedChart>
            </ResponsiveContainer>

            {/* Live readout row */}
            {history.length > 0 && (
                <div className="flex gap-3 pt-1 text-xs font-mono">
                    {[
                        { label: "Fuel", val: `${history[history.length - 1].fuel_temp.toFixed(1)}K`, color: "#ff5252" },
                        { label: "Coolant", val: `${history[history.length - 1].coolant_temp.toFixed(1)}K`, color: "#40c4ff" },
                        { label: "Power", val: `${(history[history.length - 1].power * 100).toFixed(1)}%`, color: "#69ff47" },
                    ].map(item => (
                        <span key={item.label} style={{ color: item.color }}>
                            {item.label}: <strong>{item.val}</strong>
                        </span>
                    ))}
                </div>
            )}
        </div>
    );
};

export default LiveGraphs;
