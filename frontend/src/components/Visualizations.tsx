"use client";

import React from "react";
import { ReactorState } from "@/types/reactor";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui";

/**
 * Gauges component - displays reactor parameters as circular gauges
 */
interface GaugeProps {
  label: string;
  value: number;
  unit: string;
  min: number;
  max: number;
  warning?: number;
  critical?: number;
}

export const Gauge: React.FC<GaugeProps> = ({
  label,
  value,
  unit,
  min,
  max,
  warning,
  critical,
}) => {
  const percentage = ((value - min) / (max - min)) * 100;
  const clampedPercentage = Math.max(0, Math.min(100, percentage));

  let fillColor = "#3b82f6"; // blue
  if (critical && value >= critical) {
    fillColor = "#dc2626"; // red
  } else if (warning && value >= warning) {
    fillColor = "#f59e0b"; // amber
  } else if (value < min + (max - min) * 0.3) {
    fillColor = "#10b981"; // green
  }

  const circumference = 2 * Math.PI * 45;
  const strokeDashoffset = circumference - (clampedPercentage / 100) * circumference;

  return (
    <div className="flex flex-col items-center justify-center">
      <svg width="120" height="120" className="transform -rotate-90">
        {/* Background circle */}
        <circle
          cx="60"
          cy="60"
          r="45"
          fill="none"
          stroke="#e5e7eb"
          strokeWidth="8"
        />
        {/* Progress circle */}
        <circle
          cx="60"
          cy="60"
          r="45"
          fill="none"
          stroke={fillColor}
          strokeWidth="8"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          className="transition-all duration-300"
        />
        {/* Center text */}
        <text
          x="60"
          y="55"
          textAnchor="middle"
          className="text-sm font-bold fill-gray-900"
          dominantBaseline="middle"
        >
          {value.toFixed(3)}
        </text>
        <text
          x="60"
          y="70"
          textAnchor="middle"
          className="text-xs fill-gray-600"
          dominantBaseline="middle"
        >
          {unit}
        </text>
      </svg>
      <p className="mt-2 text-sm font-medium text-gray-700">{label}</p>
    </div>
  );
};

/**
 * Gauges Panel - displays all main reactor parameters
 */
export const GaugesPanel: React.FC<{ state: ReactorState | null }> = ({
  state,
}) => {
  if (!state) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Reactor Parameters</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-gray-500">No data yet</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Reactor Parameters</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-6 py-4">
          <Gauge
            label="Power"
            value={state.power}
            unit="MW"
            min={0}
            max={150}
            warning={120}
            critical={140}
          />
          <Gauge
            label="Fuel Temp"
            value={state.fuel_temp}
            unit="K"
            min={273}
            max={1000}
            warning={800}
            critical={900}
          />
          <Gauge
            label="Coolant Temp"
            value={state.coolant_temp}
            unit="K"
            min={273}
            max={600}
            warning={550}
            critical={580}
          />
          <Gauge
            label="Pressure"
            value={state.pressure}
            unit="bar"
            min={0}
            max={160}
            warning={150}
            critical={155}
          />
        </div>
      </CardContent>
    </Card>
  );
};

/**
 * Control Rods Visualization
 */
interface ControlRodsProps {
  power: number;
  precursors: number;
}

export const ControlRods: React.FC<ControlRodsProps> = ({ power, precursors }) => {
  return (
    <div className="flex h-48 items-center justify-center gap-8 rounded-lg border border-gray-200 bg-gray-50 p-6">
      <div className="flex flex-col items-center">
        <div className="mb-2 text-xs font-semibold text-gray-600">POWER</div>
        <div className="h-32 w-8 rounded border-2 border-gray-400 bg-gray-100">
          <div
            className="h-full w-full rounded bg-linear-to-t from-blue-500 to-blue-400 transition-all"
            style={{ height: `${Math.min(100, power)}%` }}
          />
        </div>
        <div className="mt-2 text-sm font-bold text-gray-900">
          {power.toFixed(3)} MW
        </div>
      </div>

      <div className="flex flex-col items-center">
        <div className="mb-2 text-xs font-semibold text-gray-600">
          PRECURSORS
        </div>
        <div className="h-32 w-8 rounded border-2 border-gray-400 bg-gray-100">
          <div
            className="h-full w-full rounded bg-linear-to-t from-green-500 to-green-400 transition-all"
            style={{ height: `${Math.min(100, precursors)}%` }}
          />
        </div>
        <div className="mt-2 text-sm font-bold text-gray-900">
          {precursors.toFixed(3)}%
        </div>
      </div>
    </div>
  );
};

/**
 * Temperature Heatmap
 */
interface TemperatureHeatmapProps {
  fuelTemp: number;
  coolantTemp: number;
  pressure: number;
}

export const TemperatureHeatmap: React.FC<TemperatureHeatmapProps> = ({
  fuelTemp,
  coolantTemp,
  pressure,
}) => {
  // Normalize temperatures to 0-100 scale for heatmap
  const fuelNorm = Math.min(100, (fuelTemp / 1000) * 100);
  const coolantNorm = Math.min(100, (coolantTemp / 600) * 100);
  const pressureNorm = Math.min(100, (pressure / 160) * 100);

  const getHeatmapColor = (value: number) => {
    if (value < 30) return "bg-blue-400";
    if (value < 50) return "bg-green-400";
    if (value < 70) return "bg-yellow-400";
    if (value < 85) return "bg-orange-400";
    return "bg-red-500";
  };

  return (
    <div className="space-y-4">
      <div>
        <div className="mb-1 flex justify-between text-xs font-semibold text-gray-600">
          <span>Fuel Temperature</span>
          <span>{fuelTemp.toFixed(3)}K</span>
        </div>
        <div className="h-6 overflow-hidden rounded-lg bg-gray-100">
          <div
            className={`h-full transition-all ${getHeatmapColor(fuelNorm)}`}
            style={{ width: `${fuelNorm}%` }}
          />
        </div>
      </div>

      <div>
        <div className="mb-1 flex justify-between text-xs font-semibold text-gray-600">
          <span>Coolant Temperature</span>
          <span>{coolantTemp.toFixed(3)}K</span>
        </div>
        <div className="h-6 overflow-hidden rounded-lg bg-gray-100">
          <div
            className={`h-full transition-all ${getHeatmapColor(coolantNorm)}`}
            style={{ width: `${coolantNorm}%` }}
          />
        </div>
      </div>

      <div>
        <div className="mb-1 flex justify-between text-xs font-semibold text-gray-600">
          <span>System Pressure</span>
          <span>{pressure.toFixed(3)} bar</span>
        </div>
        <div className="h-6 overflow-hidden rounded-lg bg-gray-100">
          <div
            className={`h-full transition-all ${getHeatmapColor(pressureNorm)}`}
            style={{ width: `${pressureNorm}%` }}
          />
        </div>
      </div>
    </div>
  );
};

/**
 * Real-time Graph Container (placeholder for recharts integration)
 */
export const LiveGraph: React.FC<{
  title: string;
  data: { time: number; value: number }[];
  yUnit: string;
  yLabel: string;
  color?: string;
}> = ({ title, data, yUnit, yLabel, color = "#3b82f6" }) => {
  const isEmpty = !data || data.length === 0;

  if (isEmpty) {
    return (
      <Card className="col-span-1">
        <CardHeader>
          <CardTitle className="text-base">{title}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex h-48 items-center justify-center text-sm text-gray-400">
            Waiting for data...
          </div>
        </CardContent>
      </Card>
    );
  }

  const latestValue = data[data.length - 1]?.value || 0;
  const minValue = Math.min(...data.map((d) => d.value));
  const maxValue = Math.max(...data.map((d) => d.value));

  return (
    <Card className="col-span-1">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">{title}</CardTitle>
          <div className="text-right">
            <div className="text-2xl font-bold" style={{ color }}>
              {latestValue.toFixed(3)}
            </div>
            <div className="text-xs text-gray-500">{yUnit}</div>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="h-32 space-y-2">
          <div className="flex justify-between text-xs text-gray-500">
            <span>{yLabel}</span>
            <div>
              Min: {minValue.toFixed(3)} | Max: {maxValue.toFixed(3)}
            </div>
          </div>
          <div className="h-24 rounded border border-gray-200 bg-gray-50 p-2">
            <svg
              className="h-full w-full"
              viewBox={`0 0 ${data.length} 100`}
              preserveAspectRatio="none"
            >
              <polyline
                points={data
                  .map(
                    (d, i) =>
                      `${i},${100 - ((d.value - minValue) / (maxValue - minValue)) * 100}`
                  )
                  .join(" ")}
                fill="none"
                stroke={color}
                strokeWidth="0.5"
                vectorEffect="non-scaling-stroke"
              />
            </svg>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
