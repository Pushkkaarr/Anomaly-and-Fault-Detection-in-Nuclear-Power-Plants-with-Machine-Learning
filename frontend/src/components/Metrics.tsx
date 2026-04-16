"use client";

import React from "react";
import { SimulationEvent, SimulationMetrics } from "@/types/reactor";
import {
  AlertCircle, CheckCircle, AlertTriangle, Info, Trash2,
} from "lucide-react";

interface EventLogProps {
  events: SimulationEvent[];
  onClear?: () => void;
}

const EVENT_STYLES = {
  critical: {
    bg: "rgba(255,59,59,0.1)",
    border: "rgba(255,59,59,0.4)",
    icon: <AlertCircle className="h-3.5 w-3.5" />,
    iconColor: "#fb2c36",
    textColor: "#ffaaaa",
    badge: "rgba(255,59,59,0.25)",
  },
  warning: {
    bg: "rgba(255,214,0,0.08)",
    border: "rgba(255,214,0,0.35)",
    icon: <AlertTriangle className="h-3.5 w-3.5" />,
    iconColor: "#fbbf24",
    textColor: "#ffe680",
    badge: "rgba(255,214,0,0.2)",
  },
  success: {
    bg: "rgba(0,255,136,0.08)",
    border: "rgba(0,255,136,0.35)",
    icon: <CheckCircle className="h-3.5 w-3.5" />,
    iconColor: "var(--brand-accent)",
    textColor: "#aaefcc",
    badge: "rgba(0,255,136,0.2)",
  },
  info: {
    bg: "rgba(0,255,136,0.06)",
    border: "rgba(0,255,136,0.25)",
    icon: <Info className="h-3.5 w-3.5" />,
    iconColor: "var(--brand-accent)",
    textColor: "#a0d8e8",
    badge: "rgba(0,255,136,0.15)",
  },
};

export const EventLog: React.FC<EventLogProps> = ({ events, onClear }) => {
  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <p className="section-label">Event Stream</p>
        {onClear && events.length > 0 && (
          <button
            onClick={onClear}
            className="flex items-center gap-1 text-xs px-2 py-0.5 rounded transition-opacity hover:opacity-80"
            style={{ color: "rgba(107,143,168,0.7)", border: "1px solid rgba(107,143,168,0.15)" }}
          >
            <Trash2 className="h-3 w-3" />
            Clear
          </button>
        )}
      </div>

      <div className="space-y-1.5 overflow-y-auto" style={{ maxHeight: "220px" }}>
        {events.length === 0 ? (
          <div className="py-6 text-center">
            <p className="text-xs" style={{ color: "rgba(107,143,168,0.4)" }}>
              Awaiting events...
            </p>
          </div>
        ) : (
          events.map((event) => {
            const style = EVENT_STYLES[event.type] || EVENT_STYLES.info;
            return (
              <div
                key={event.id}
                className="flex gap-2 rounded-lg px-2.5 py-2"
                style={{
                  background: style.bg,
                  border: `1px solid ${style.border}`,
                }}
              >
                <span style={{ color: style.iconColor, flexShrink: 0, marginTop: 1 }}>
                  {style.icon}
                </span>
                <div className="flex-1 min-w-0">
                  <p
                    className="text-xs leading-tight"
                    style={{ color: style.textColor }}
                  >
                    {event.message}
                  </p>
                  <p
                    className="text-xs mt-0.5 font-mono"
                    style={{ color: "rgba(107,143,168,0.5)", fontSize: "0.6rem" }}
                  >
                    t={event.timestamp.toFixed(3)}s
                  </p>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};

interface MetricsSummaryProps {
  metrics: SimulationMetrics | null;
  isRunning: boolean;
}

export const MetricsSummary: React.FC<MetricsSummaryProps> = ({ metrics, isRunning }) => {
  if (!metrics) {
    return (
      <p className="text-xs text-center py-4" style={{ color: "rgba(107,143,168,0.5)" }}>
        {isRunning ? "Collecting metrics..." : "Run a simulation to see summary"}
      </p>
    );
  }

  const items = [
    { label: "Total Reward", value: metrics.total_reward.toFixed(3), unit: "pts", highlight: true },
    { label: "Steps", value: metrics.episode_steps.toString(), unit: "" },
    { label: "Duration", value: metrics.episode_duration.toFixed(3), unit: "s" },
    { label: "Peak Fuel Temp", value: metrics.max_fuel_temp.toFixed(3), unit: "K" },
    { label: "Peak Coolant", value: metrics.max_coolant_temp.toFixed(3), unit: "K" },
    { label: "Avg Pressure", value: metrics.avg_pressure.toFixed(3), unit: "bar" },
  ];

  return (
    <div className="grid grid-cols-2 gap-2">
      {items.map((item, i) => (
        <div
          key={i}
          className="rounded-lg px-3 py-2"
          style={{
            background: item.highlight ? "rgba(0,255,136,0.08)" : "rgba(255,255,255,0.03)",
            border: item.highlight ? "1px solid rgba(0,255,136,0.2)" : "1px solid rgba(255,255,255,0.05)",
          }}
        >
          <p className="section-label" style={{ fontSize: "0.58rem" }}>{item.label}</p>
          <p
            className="text-base font-bold font-mono mt-0.5"
            style={{ color: item.highlight ? "var(--brand-accent)" : "#a0b8c8" }}
          >
            {item.value}
            {item.unit && (
              <span className="text-xs font-normal ml-1" style={{ color: "rgba(107,143,168,0.6)" }}>
                {item.unit}
              </span>
            )}
          </p>
        </div>
      ))}
    </div>
  );
};

export const ScoreCard: React.FC<{
  title: string;
  value: number | string;
  unit?: string;
  color?: string;
}> = ({ title, value, unit, color = "var(--brand-accent)" }) => (
  <div
    className="rounded-lg px-3 py-2"
    style={{ background: "rgba(5,15,31,0.8)", border: `1px solid ${color}25` }}
  >
    <p className="section-label" style={{ fontSize: "0.58rem" }}>{title}</p>
    <p className="text-lg font-bold font-mono mt-0.5" style={{ color }}>
      {typeof value === "number" ? value.toFixed(3) : value}
      {unit && <span className="text-xs font-normal ml-1" style={{ color: "rgba(107,143,168,0.5)" }}>{unit}</span>}
    </p>
  </div>
);


