"use client";

import React from "react";
import { SimulationEvent, SimulationMetrics } from "@/types/reactor";
import { Card, CardContent, CardHeader, CardTitle, Badge } from "@/components/ui";
import {
  AlertCircle,
  CheckCircle,
  AlertTriangle,
  Info,
  Trash2,
} from "lucide-react";

/**
 * Event Log Component - displays simulation events in real-time
 */
interface EventLogProps {
  events: SimulationEvent[];
  onClear?: () => void;
  maxHeight?: string;
}

export const EventLog: React.FC<EventLogProps> = ({
  events,
  onClear,
  maxHeight = "max-h-80",
}) => {
  const getIconComponent = (type: SimulationEvent["type"]) => {
    const size = "h-4 w-4";
    switch (type) {
      case "critical":
        return <AlertCircle className={size} />;
      case "warning":
        return <AlertTriangle className={size} />;
      case "success":
        return <CheckCircle className={size} />;
      case "info":
      default:
        return <Info className={size} />;
    }
  };

  const getEventColor = (type: SimulationEvent["type"]) => {
    switch (type) {
      case "critical":
        return "border-red-200 bg-red-50";
      case "warning":
        return "border-yellow-200 bg-yellow-50";
      case "success":
        return "border-green-200 bg-green-50";
      case "info":
      default:
        return "border-blue-200 bg-blue-50";
    }
  };

  const getBadgeVariant = (
    type: SimulationEvent["type"]
  ): "danger" | "warning" | "success" | "info" => {
    switch (type) {
      case "critical":
        return "danger";
      case "warning":
        return "warning";
      case "success":
        return "success";
      case "info":
      default:
        return "info";
    }
  };

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">Event Log</CardTitle>
        {onClear && events.length > 0 && (
          <button
            onClick={onClear}
            className="inline-flex items-center gap-1 rounded px-2 py-1 text-xs text-gray-500 hover:bg-gray-100"
          >
            <Trash2 className="h-3 w-3" />
          </button>
        )}
      </CardHeader>
      <CardContent>
        <div className={`space-y-2 overflow-y-auto ${maxHeight}`}>
          {events.length === 0 ? (
            <p className="py-4 text-center text-sm text-gray-400">
              No events yet
            </p>
          ) : (
            events.map((event) => (
              <div
                key={event.id}
                className={`flex gap-3 rounded-lg border px-3 py-2 text-sm ${getEventColor(event.type)}`}
              >
                <div className={`mt-0.5 shrink-0 text-${event.type}`}>
                  {getIconComponent(event.type)}
                </div>
                <div className="flex-1">
                  <div className="flex items-start justify-between gap-2">
                    <p className="font-medium text-gray-900">
                      {event.message}
                    </p>
                    <Badge variant={getBadgeVariant(event.type)} className="text-xs">
                      {event.type}
                    </Badge>
                  </div>
                  <p className="mt-1 text-xs text-gray-600">
                    t = {event.timestamp.toFixed(1)}s
                  </p>
                </div>
              </div>
            ))
          )}
        </div>
      </CardContent>
    </Card>
  );
};

/**
 * Metrics Summary Component
 */
interface MetricsSummaryProps {
  metrics: SimulationMetrics | null;
  isRunning: boolean;
}

export const MetricsSummary: React.FC<MetricsSummaryProps> = ({
  metrics,
  isRunning,
}) => {
  if (!metrics) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Summary</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-gray-500">
            {isRunning ? "Collecting metrics..." : "No metrics available yet"}
          </p>
        </CardContent>
      </Card>
    );
  }

  const metricItems = [
    {
      label: "Total Reward",
      value: metrics.total_reward.toFixed(2),
      unit: "pts",
      highlight: true,
    },
    {
      label: "Steps Taken",
      value: metrics.episode_steps.toString(),
      unit: "steps",
    },
    {
      label: "Duration",
      value: metrics.episode_duration.toFixed(1),
      unit: "s",
    },
    {
      label: "Max Fuel Temp",
      value: metrics.max_fuel_temp.toFixed(0),
      unit: "K",
    },
    {
      label: "Max Coolant Temp",
      value: metrics.max_coolant_temp.toFixed(0),
      unit: "K",
    },
    {
      label: "Avg Pressure",
      value: metrics.avg_pressure.toFixed(1),
      unit: "bar",
    },
    {
      label: "Power Change Rate",
      value: metrics.power_change_rate.toFixed(2),
      unit: "/s",
    },
    {
      label: "Safety Events",
      value: metrics.safety_events.toString(),
      unit: "count",
    },
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Summary Statistics</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4">
          {metricItems.map((item, idx) => (
            <div
              key={idx}
              className={`rounded-lg p-3 ${item.highlight ? "bg-blue-50" : "bg-gray-50"}`}
            >
              <p className="text-xs text-gray-600">{item.label}</p>
              <p
                className={`mt-1 text-xl font-bold ${item.highlight ? "text-blue-900" : "text-gray-900"}`}
              >
                {item.value}
                <span className="text-xs font-normal text-gray-500">
                  {" "}
                  {item.unit}
                </span>
              </p>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

/**
 * Score Card - displays a single metric prominently
 */
interface ScoreCardProps {
  title: string;
  value: number | string;
  unit?: string;
  icon?: React.ReactNode;
  variant?: "default" | "success" | "warning" | "danger";
  trend?: "up" | "down" | "stable";
}

export const ScoreCard: React.FC<ScoreCardProps> = ({
  title,
  value,
  unit,
  icon,
  variant = "default",
  trend,
}) => {
  const bgColors = {
    default: "bg-blue-50",
    success: "bg-green-50",
    warning: "bg-yellow-50",
    danger: "bg-red-50",
  };

  const textColors = {
    default: "text-blue-900",
    success: "text-green-900",
    warning: "text-yellow-900",
    danger: "text-red-900",
  };

  return (
    <div className={`rounded-lg px-4 py-3 ${bgColors[variant]}`}>
      <div className="flex items-center justify-between">
        <div>
          <p className={`text-xs font-medium ${textColors[variant]}`}>
            {title}
          </p>
          <p className="mt-1 flex items-baseline gap-1 text-2xl font-bold">
            <span className={textColors[variant]}>
              {typeof value === "number" ? value.toFixed(1) : value}
            </span>
            {unit && (
              <span className="text-sm text-gray-600 font-normal">{unit}</span>
            )}
          </p>
        </div>
        {icon && <div className="h-8 w-8 opacity-50">{icon}</div>}
      </div>
      {trend && (
        <p className="mt-2 text-xs text-gray-600">
          Trend: {trend === "up" ? "📈" : trend === "down" ? "📉" : "➡️"}
        </p>
      )}
    </div>
  );
};

/**
 * Performance Comparison - side by side metrics when comparing two models
 */
interface PerformanceComparisonProps {
  model1Name: string;
  model1Metrics: SimulationMetrics | null;
  model2Name: string;
  model2Metrics: SimulationMetrics | null;
}

export const PerformanceComparison: React.FC<PerformanceComparisonProps> = ({
  model1Name,
  model1Metrics,
  model2Name,
  model2Metrics,
}) => {
  if (!model1Metrics || !model2Metrics) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Model Comparison</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-gray-500">
            Waiting for both model results...
          </p>
        </CardContent>
      </Card>
    );
  }

  const metrics = [
    {
      label: "Total Reward",
      get: (m: SimulationMetrics) => m.total_reward,
      format: (v: number) => v.toFixed(2),
    },
    {
      label: "Steps",
      get: (m: SimulationMetrics) => m.episode_steps,
      format: (v: number) => v.toString(),
    },
    {
      label: "Duration",
      get: (m: SimulationMetrics) => m.episode_duration,
      format: (v: number) => v.toFixed(1) + "s",
    },
    {
      label: "Max Fuel Temp",
      get: (m: SimulationMetrics) => m.max_fuel_temp,
      format: (v: number) => v.toFixed(0) + "K",
    },
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Model Comparison</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {metrics.map((metric) => {
            const v1 = metric.get(model1Metrics);
            const v2 = metric.get(model2Metrics);
            const isWinnerM1 =
              metric.label === "Total Reward" ? v1 > v2 : v1 < v2;

            return (
              <div key={metric.label} className="space-y-1">
                <p className="text-xs font-semibold text-gray-700">
                  {metric.label}
                </p>
                <div className="flex gap-2">
                  <div
                    className={`flex-1 rounded-lg px-3 py-2 text-sm font-semibold ${isWinnerM1 ? "bg-green-100 text-green-900" : "bg-gray-100 text-gray-900"}`}
                  >
                    {model1Name}: {metric.format(v1)}
                  </div>
                  <div
                    className={`flex-1 rounded-lg px-3 py-2 text-sm font-semibold ${!isWinnerM1 ? "bg-green-100 text-green-900" : "bg-gray-100 text-gray-900"}`}
                  >
                    {model2Name}: {metric.format(v2)}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
};
