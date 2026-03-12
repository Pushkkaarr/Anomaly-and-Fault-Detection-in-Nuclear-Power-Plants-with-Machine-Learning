"use client";

import React, { useMemo } from "react";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ReferenceLine,
    ResponsiveContainer,
} from "recharts";
import { ReactorState } from "@/types/reactor";

interface LiveGraphsProps {
    history: ReactorState[];
    isRunning: boolean;
}

const MONO = "'JetBrains Mono', monospace";
const AXIS_TICK = { fontSize: 9, fill: "rgba(107,143,168,0.7)", fontFamily: MONO };

// Compact tooltip
const MiniTooltip = ({ active, payload, label, unit }: any) => {
    if (!active || !payload?.length) return null;
    return (
        <div style={{
            background: "rgba(2,8,18,0.97)",
            border: "1px solid rgba(0,212,255,0.2)",
            borderRadius: 5,
            padding: "5px 10px",
            fontSize: 10,
            fontFamily: MONO,
        }}>
            <p style={{ color: "rgba(0,212,255,0.5)", margin: 0 }}>t={Number(label).toFixed(1)}s</p>
            <p style={{ color: payload[0].color, margin: 0, fontWeight: "bold" }}>
                {Number(payload[0].value).toFixed(2)}{unit}
            </p>
        </div>
    );
};

// Compute tight domain with ±margin (still ensures range > minRange so flat lines show)
function tightDomain(values: number[], minRange = 0.5, margin = 0.15): [number, number] {
    if (!values.length) return [0, 1];
    const lo = Math.min(...values);
    const hi = Math.max(...values);
    const range = Math.max(hi - lo, minRange);
    const pad = range * margin;
    return [lo - pad, hi + pad];
}

// One small self-contained metric chart
interface MetricChartProps {
    data: { t: number; v: number }[];
    color: string;
    label: string;
    unit: string;
    minRange?: number;
    refLines?: { y: number; color: string; label: string }[];
    height?: number;
}
const MetricChart: React.FC<MetricChartProps> = ({
    data, color, label, unit, minRange = 0.5, refLines = [], height = 90,
}) => {
    const domain = useMemo(() => tightDomain(data.map(d => d.v), minRange), [data, minRange]);
    const latest = data[data.length - 1]?.v;

    return (
        <div className="w-full">
            <div className="flex items-baseline justify-between mb-0.5 px-1">
                <p style={{ fontSize: "0.58rem", color: "rgba(107,143,168,0.7)", fontFamily: MONO, textTransform: "uppercase", letterSpacing: "0.08em" }}>
                    {label}
                </p>
                <span style={{ fontSize: "0.72rem", color, fontFamily: MONO, fontWeight: 700, textShadow: `0 0 8px ${color}60` }}>
                    {latest !== undefined ? `${latest.toFixed(latest > 100 ? 1 : 3)}${unit}` : "—"}
                </span>
            </div>
            <ResponsiveContainer width="100%" height={height}>
                <LineChart data={data} margin={{ top: 3, right: 8, bottom: 0, left: -10 }}>
                    <CartesianGrid strokeDasharray="2 5" stroke="rgba(0,212,255,0.04)" />
                    <XAxis
                        dataKey="t"
                        tick={AXIS_TICK}
                        strokeWidth={0}
                        tickFormatter={v => `${Number(v).toFixed(0)}s`}
                        interval="preserveStartEnd"
                        minTickGap={30}
                    />
                    <YAxis
                        domain={domain}
                        tick={AXIS_TICK}
                        strokeWidth={0}
                        tickFormatter={v => {
                            const n = Number(v);
                            return n > 999 ? `${(n / 1000).toFixed(2)}k` : n.toFixed(n > 10 ? 1 : 3);
                        }}
                        width={38}
                    />
                    <Tooltip content={<MiniTooltip unit={unit} />} />
                    {refLines.map((r, i) => (
                        <ReferenceLine key={i} y={r.y}
                            stroke={r.color} strokeDasharray="4 3" strokeOpacity={0.5}
                            label={{ value: r.label, fill: r.color, fontSize: 7.5, position: "right" }}
                        />
                    ))}
                    <Line
                        type="monotone" dataKey="v"
                        stroke={color} strokeWidth={1.8} dot={false}
                        isAnimationActive={false}
                    />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
};

export const LiveGraphs: React.FC<LiveGraphsProps> = ({ history, isRunning }) => {
    if (!history || history.length < 2) {
        return (
            <div className="flex h-44 items-center justify-center rounded-lg"
                style={{ background: "rgba(0,0,0,0.2)", border: "1px dashed rgba(0,212,255,0.1)" }}>
                <div className="text-center">
                    <div className="text-2xl mb-2 opacity-20">📈</div>
                    <p className="text-xs" style={{ color: "rgba(0,212,255,0.35)" }}>
                        {isRunning ? "Collecting data…" : "Start simulation to view live graphs"}
                    </p>
                </div>
            </div>
        );
    }

    // Build separate data series
    const fuelData = history.map(s => ({ t: s.time, v: s.fuel_temp }));
    const coolantData = history.map(s => ({ t: s.time, v: s.coolant_temp }));
    const powerData = history.map(s => ({ t: s.time, v: s.power * 100 }));
    const pressureData = history.map(s => ({ t: s.time, v: s.pressure }));

    return (
        <div className="w-full space-y-3">
            {/* FUEL TEMP — tight axis so ±5K changes are visible */}
            <MetricChart
                data={fuelData}
                color="#ff5252"
                label="Fuel Temperature"
                unit="K"
                minRange={2}           // show chart even if only 2K range (AI is precise)
                height={85}
                refLines={[{ y: 950, color: "#ff5252", label: "⚠950K" }]}
            />

            {/* COOLANT TEMP */}
            <MetricChart
                data={coolantData}
                color="#40c4ff"
                label="Coolant Temperature"
                unit="K"
                minRange={0.5}
                height={75}
                refLines={[{ y: 310, color: "#ffd600", label: "⚠310K" }]}
            />

            {/* POWER — show as % */}
            <MetricChart
                data={powerData}
                color="#69ff47"
                label="Reactor Power"
                unit="%"
                minRange={0.2}
                height={75}
                refLines={[
                    { y: 120, color: "#ff5252", label: "120%" },
                    { y: 80, color: "#ffd600", label: "80%" },
                ]}
            />

            {/* PRESSURE */}
            <MetricChart
                data={pressureData}
                color="#e040fb"
                label="System Pressure"
                unit=" bar"
                minRange={0.05}
                height={65}
                refLines={[
                    { y: 12, color: "#ff5252", label: "12 bar" },
                    { y: 8, color: "#ffd600", label: "8 bar" },
                ]}
            />
        </div>
    );
};

export default LiveGraphs;
