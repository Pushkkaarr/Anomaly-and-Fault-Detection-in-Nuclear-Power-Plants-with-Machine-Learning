"use client";

import React from "react";
import { Action } from "@/types/reactor";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui";
import { Zap } from "lucide-react";

/**
 * AI Action Display Component
 * Shows the last control action taken by the AI model
 */
interface AIActionDisplayProps {
  action: Action | null;
  isAutomatic: boolean;
  modelName?: string;
  currentStep?: number;
  maxSteps?: number;
}

export const AIActionDisplay: React.FC<AIActionDisplayProps> = ({
  action,
  isAutomatic,
  modelName = "AI Model",
  currentStep = 0,
  maxSteps = 200,
}) => {
  if (!action) {
    return null;
  }

  const progressPercent = maxSteps > 0 ? (currentStep / maxSteps) * 100 : 0;

  return (
    <Card className="border-blue-200 bg-blue-50">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-sm">
            <Zap className="h-4 w-4 text-yellow-600" />
            AI Last Action
          </CardTitle>
          <span className="text-xs font-semibold text-blue-700">
            {modelName}
          </span>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Progress Bar */}
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-gray-600">
            <span>Simulation Progress</span>
            <span className="font-mono">
              {currentStep}/{maxSteps}
            </span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-gray-200">
            <div
              className="h-full bg-gradient-to-r from-blue-400 to-blue-600 transition-all duration-300"
              style={{
                width: `${progressPercent}%`,
              }}
            />
          </div>
        </div>

        {/* Control Rod */}
        <div className="space-y-1">
          <div className="flex justify-between text-xs">
            <span className="font-semibold text-gray-700">Control Rod</span>
            <span className="font-mono text-blue-700">
              {action.control_rod.toFixed(4)}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <div className="h-2 w-full overflow-hidden rounded-full bg-gray-200">
              <div
                className="h-full bg-blue-600 transition-all"
                style={{
                  width: `${((action.control_rod + 1) / 2) * 100}%`,
                }}
              />
            </div>
          </div>
          <div className="flex justify-between text-xs text-gray-500">
            <span>-1.0</span>
            <span>0.0</span>
            <span>+1.0</span>
          </div>
          <p className="text-xs text-gray-600 mt-1">
            {action.control_rod > 0.05
              ? "↓ Inserting rods (reduce power)"
              : action.control_rod < -0.05
              ? "↑ Retracting rods (increase power)"
              : "≈ Neutral"}
          </p>
        </div>

        {/* Coolant Flow */}
        <div className="space-y-1 border-t pt-3">
          <div className="flex justify-between text-xs">
            <span className="font-semibold text-gray-700">Coolant Flow</span>
            <span className="font-mono text-green-700">
              {action.coolant_flow.toFixed(4)}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <div className="h-2 w-full overflow-hidden rounded-full bg-gray-200">
              <div
                className="h-full bg-green-600 transition-all"
                style={{
                  width: `${((action.coolant_flow + 1) / 2) * 100}%`,
                }}
              />
            </div>
          </div>
          <div className="flex justify-between text-xs text-gray-500">
            <span>-1.0</span>
            <span>0.0</span>
            <span>+1.0</span>
          </div>
          <p className="text-xs text-gray-600 mt-1">
            {action.coolant_flow > 0.05
              ? "⬆ Increasing flow (better cooling)"
              : action.coolant_flow < -0.05
              ? "⬇ Decreasing flow"
              : "≈ Normal flow"}
          </p>
        </div>

        {/* Model Status */}
        {isAutomatic && (
          <div className="border-t pt-2 mt-2">
            <div className="flex items-center gap-2">
              <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
              <p className="text-xs text-gray-600">
                AI model making autonomous decisions
              </p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default AIActionDisplay;
