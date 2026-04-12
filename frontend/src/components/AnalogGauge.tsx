"use client";

import React, { useRef, useEffect } from "react";

interface AnalogGaugeProps {
    value: number;        // Current value
    min: number;          // Minimum value
    max: number;          // Maximum value
    label: string;        // Display label
    unit: string;         // Unit string
    safeMin?: number;     // Safe zone min
    safeMax?: number;     // Safe zone max
    warnMax?: number;     // Warning threshold
    size?: number;        // SVG size (default 140)
}

export const AnalogGauge: React.FC<AnalogGaugeProps> = ({
    value,
    min,
    max,
    label,
    unit,
    safeMin,
    safeMax,
    warnMax,
    size = 140,
}) => {
    const clampedValue = Math.max(min, Math.min(max, value));
    const percent = (clampedValue - min) / (max - min);

    // Arc from -225deg to 45deg (270 degrees total)
    const startAngle = -225;
    const totalAngle = 270;
    const needleAngle = startAngle + percent * totalAngle;

    const cx = size / 2;
    const cy = size / 2;
    const r = size * 0.38;
    const trackR = size * 0.42;

    const polarToCart = (angle: number, radius: number) => {
        const rad = ((angle - 90) * Math.PI) / 180;
        return {
            x: cx + radius * Math.cos(rad),
            y: cy + radius * Math.sin(rad),
        };
    };

    const describeArc = (startA: number, endA: number, radius: number) => {
        const start = polarToCart(startA, radius);
        const end = polarToCart(endA, radius);
        const largeArc = endA - startA > 180 ? 1 : 0;
        return `M ${start.x} ${start.y} A ${radius} ${radius} 0 ${largeArc} 1 ${end.x} ${end.y}`;
    };

    // Determine status color
    const getStatusColor = () => {
        if (warnMax !== undefined && clampedValue > warnMax) return "#ff3b3b";
        if (safeMax !== undefined && clampedValue > safeMax) return "#fbbf24";
        if (safeMin !== undefined && clampedValue < safeMin) return "#fbbf24";
        return "var(--brand-accent)";
    };

    const statusColor = getStatusColor();

    // Needle endpoint
    const needleLength = r * 0.85;
    const needleEnd = polarToCart(needleAngle, needleLength);
    const needleBase1 = polarToCart(needleAngle + 90, 4);
    const needleBase2 = polarToCart(needleAngle - 90, 4);

    // Safe zone arc
    const safeStartPercent = safeMin !== undefined ? (safeMin - min) / (max - min) : 0;
    const safeEndPercent = safeMax !== undefined ? (safeMax - min) / (max - min) : 1;
    const safeStartAngle = startAngle + safeStartPercent * totalAngle;
    const safeEndAngle = startAngle + safeEndPercent * totalAngle;

    // Warn zone arc
    const warnStartAngle = safeMax !== undefined
        ? startAngle + safeEndPercent * totalAngle
        : startAngle + 0.75 * totalAngle;
    const warnEndAngle = startAngle + totalAngle;

    const displayValue = typeof value === "number"
        ? value >= 10 ? value.toFixed(1) : value.toFixed(2)
        : "0";

    return (
        <div className="flex flex-col items-center gap-1">
            <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
                {/* Outer ring */}
                <circle
                    cx={cx} cy={cy} r={trackR + 3}
                    fill="none"
                    stroke="rgba(0,255,136,0.08)"
                    strokeWidth="1"
                />

                {/* Background track */}
                <path
                    d={describeArc(startAngle, startAngle + totalAngle, trackR)}
                    fill="none"
                    stroke="rgba(255,255,255,0.06)"
                    strokeWidth={size * 0.055}
                    strokeLinecap="round"
                />

                {/* Safe zone arc */}
                {safeMax !== undefined && (
                    <path
                        d={describeArc(safeStartAngle, safeEndAngle, trackR)}
                        fill="none"
                        stroke="rgba(0,255,136,0.35)"
                        strokeWidth={size * 0.055}
                        strokeLinecap="butt"
                    />
                )}

                {/* Warning zone arc */}
                <path
                    d={describeArc(warnStartAngle, warnEndAngle, trackR)}
                    fill="none"
                    stroke="rgba(255,59,59,0.3)"
                    strokeWidth={size * 0.055}
                    strokeLinecap="round"
                />

                {/* Active value arc */}
                <path
                    d={describeArc(startAngle, startAngle + percent * totalAngle, trackR)}
                    fill="none"
                    stroke={statusColor}
                    strokeWidth={size * 0.055}
                    strokeLinecap="round"
                    style={{
                        filter: `drop-shadow(0 0 4px ${statusColor}80)`,
                        transition: "all 0.3s ease",
                    }}
                />

                {/* Tick marks */}
                {[0, 0.25, 0.5, 0.75, 1].map((t) => {
                    const tickAngle = startAngle + t * totalAngle;
                    const inner = polarToCart(tickAngle, trackR - size * 0.07);
                    const outer = polarToCart(tickAngle, trackR + size * 0.02);
                    return (
                        <line
                            key={t}
                            x1={inner.x} y1={inner.y}
                            x2={outer.x} y2={outer.y}
                            stroke="rgba(255,255,255,0.25)"
                            strokeWidth="1.5"
                        />
                    );
                })}

                {/* Center hub */}
                <circle
                    cx={cx} cy={cy} r={size * 0.05}
                    fill="#0d1f38"
                    stroke={statusColor}
                    strokeWidth="1.5"
                />

                {/* Needle */}
                <polygon
                    points={`${needleEnd.x},${needleEnd.y} ${needleBase1.x},${needleBase1.y} ${needleBase2.x},${needleBase2.y}`}
                    fill={statusColor}
                    opacity="0.9"
                    style={{
                        filter: `drop-shadow(0 0 3px ${statusColor})`,
                        transition: "all 0.35s cubic-bezier(0.25, 0.46, 0.45, 0.94)",
                    }}
                />

                {/* Value text */}
                <text
                    x={cx} y={cy + r * 0.55}
                    textAnchor="middle"
                    fontSize={size * 0.13}
                    fontWeight="700"
                    fontFamily="'JetBrains Mono', monospace"
                    fill={statusColor}
                    style={{ filter: `drop-shadow(0 0 4px ${statusColor}80)` }}
                >
                    {displayValue}
                </text>

                {/* Unit text */}
                <text
                    x={cx} y={cy + r * 0.75}
                    textAnchor="middle"
                    fontSize={size * 0.075}
                    fontFamily="Inter, sans-serif"
                    fill="rgba(255,255,255,0.45)"
                >
                    {unit}
                </text>
            </svg>

            <p className="text-xs font-semibold tracking-widest uppercase" style={{ color: "rgba(0,255,136,0.7)" }}>
                {label}
            </p>
        </div>
    );
};

export default AnalogGauge;


