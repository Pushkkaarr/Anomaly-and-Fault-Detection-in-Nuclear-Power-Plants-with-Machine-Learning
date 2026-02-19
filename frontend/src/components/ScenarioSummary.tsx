"use client";

import React from "react";
import { SimulationMetrics } from "@/types/reactor";
import { Card, CardContent, CardHeader, CardTitle, Badge } from "@/components/ui";
import { CheckCircle, AlertCircle, TrendingUp } from "lucide-react";

/**
 * Scenario Summary Component
 * Displays results when simulation completes
 */
interface ScenarioSummaryProps {
  scenario: string | null;
  metrics: SimulationMetrics | null;
  fuelTempMax?: number;
  fuelTempCurrent?: number;
  episodeSteps?: number;
}

export const ScenarioSummary: React.FC<ScenarioSummaryProps> = ({
  scenario,
  metrics,
  fuelTempMax = 0,
  fuelTempCurrent = 0,
  episodeSteps = 0,
}) => {
  if (!metrics || !scenario) return null;

  const successCriteria = {
    tempControl: fuelTempCurrent <= 1100, // Fuel temp below 1100K
    stableControl: episodeSteps >= 150, // Ran at least 150 steps
    rewardEarned: metrics.total_reward > 0, // Positive reward
  };

  const successCount = Object.values(successCriteria).filter(Boolean).length;
  const totalCriteria = Object.keys(successCriteria).length;
  const isSuccess = successCount === totalCriteria;

  const scenarioNames: Record<string, string> = {
    normal: "Normal Operation",
    lofa: "Loss of Flow Accident",
    rod_malfunction: "Control Rod Stuck",
    power_ramp: "Power Demand Ramp",
  };

  return (
    <Card className={isSuccess ? "border-green-200 bg-green-50" : "border-yellow-200 bg-yellow-50"}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {isSuccess ? (
              <CheckCircle className="h-5 w-5 text-green-600" />
            ) : (
              <AlertCircle className="h-5 w-5 text-yellow-600" />
            )}
            <CardTitle className="text-base">Scenario Complete</CardTitle>
          </div>
          <Badge variant={isSuccess ? "success" : "warning"}>
            {successCount}/{totalCriteria} ✓
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Scenario Info */}
        <div className="rounded-lg bg-white/50 px-3 py-2">
          <p className="text-xs text-gray-600">Scenario</p>
          <p className="text-sm font-semibold text-gray-900">
            {scenarioNames[scenario] || scenario}
          </p>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-2 gap-2">
          {/* Episodes Steps */}
          <div className="rounded-lg bg-white/50 px-3 py-2">
            <p className="text-xs text-gray-600">Steps Completed</p>
            <p className="text-lg font-bold text-blue-700">
              {episodeSteps}/{200}
            </p>
          </div>

          {/* Total Reward */}
          <div className="rounded-lg bg-white/50 px-3 py-2">
            <p className="text-xs text-gray-600">Total Reward</p>
            <p className={`text-lg font-bold ${metrics.total_reward > 0 ? "text-green-700" : "text-red-700"}`}>
              {metrics.total_reward.toFixed(2)}
            </p>
          </div>

          {/* Max Fuel Temp */}
          <div className="rounded-lg bg-white/50 px-3 py-2">
            <p className="text-xs text-gray-600">Max Fuel Temp</p>
            <p className={`text-lg font-bold ${fuelTempMax <= 1100 ? "text-green-700" : "text-red-700"}`}>
              {fuelTempMax.toFixed(1)}K
            </p>
          </div>

          {/* Final Fuel Temp */}
          <div className="rounded-lg bg-white/50 px-3 py-2">
            <p className="text-xs text-gray-600">Final Fuel Temp</p>
            <p className={`text-lg font-bold ${fuelTempCurrent <= 1100 ? "text-green-700" : "text-red-700"}`}>
              {fuelTempCurrent.toFixed(1)}K
            </p>
          </div>
        </div>

        {/* Success Criteria */}
        <div className="space-y-2 border-t pt-3">
          <p className="text-xs font-semibold text-gray-700">Success Criteria</p>
          <div className="space-y-1">
            {Object.entries(successCriteria).map(([key, passed]) => (
              <div key={key} className="flex items-center gap-2 text-xs">
                <div
                  className={`h-3 w-3 rounded-full ${
                    passed ? "bg-green-500" : "bg-red-500"
                  }`}
                />
                <span className="text-gray-700">
                  {key === "tempControl"
                    ? "Temperature Controlled (≤ 1100K)"
                    : key === "stableControl"
                    ? "Stable Control (≥ 150 steps)"
                    : "Positive Reward Earned"}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Result Message */}
        <div className={`rounded-lg border-l-4 px-3 py-2 ${
          isSuccess
            ? "border-green-500 bg-green-100"
            : "border-yellow-500 bg-yellow-100"
        }`}>
          <p className="text-sm font-semibold">
            {isSuccess
              ? "✓ Scenario Handled Successfully!"
              : "⚠ Scenario Partially Completed"}
          </p>
          <p className="text-xs text-gray-700 mt-1">
            {isSuccess
              ? "The AI model successfully controlled the reactor through the scenario."
              : "The AI model completed the scenario but some criteria were not met."}
          </p>
        </div>
      </CardContent>
    </Card>
  );
};

export default ScenarioSummary;
