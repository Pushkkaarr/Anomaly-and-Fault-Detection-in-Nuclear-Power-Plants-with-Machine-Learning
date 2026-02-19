"use client";

import React from "react";
import { ReactorState } from "@/types/reactor";
import { Alert } from "@/components/ui";

/**
 * Safety-Focused Parameter Display
 * Professional nuclear engineering dashboard
 */
interface SafetyParametersProps {
  state: ReactorState | null;
  lastActionTime?: number;
}

export const SafetyParameters: React.FC<SafetyParametersProps> = ({
  state,
  lastActionTime,
}) => {
  if (!state) return null;

  // Safety zones for nuclear reactor
  const fuelTempSafe = state.fuel_temp <= 1100;
  const pressureSafe = state.pressure >= 8 && state.pressure <= 12;
  const coolantTempSafe = state.coolant_temp >= 280 && state.coolant_temp <= 310;
  const powerSafe = state.power >= 0.8 && state.power <= 1.2;

  const allSafe = fuelTempSafe && pressureSafe && coolantTempSafe && powerSafe;

  return (
    <div className="space-y-4">
      {/* Safety Status Banner */}
      {!allSafe && (
        <Alert
          type={fuelTempSafe && pressureSafe ? "warning" : "error"}
          title={allSafe ? "All Systems Safe" : "Safety Alert"}
          className="border-red-300 bg-red-50"
        >
          <p className="text-sm">
            {!fuelTempSafe && "⚠️ Fuel temperature out of range (>1100K) "}
            {!pressureSafe && "⚠️ System pressure out of range "}
            {!coolantTempSafe && "⚠️ Coolant temperature out of range "}
            {!powerSafe && "⚠️ Reactor power out of range"}
          </p>
        </Alert>
      )}

      {/* Primary Safety Parameters Grid */}
      <div className="grid grid-cols-4 gap-3">
        {/* Fuel Temperature - CRITICAL */}
        <div className={`rounded-lg border-2 p-4 ${
          fuelTempSafe
            ? "border-green-300 bg-green-50"
            : "border-red-300 bg-red-50 animate-pulse"
        }`}>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs font-semibold text-gray-600 uppercase tracking-wider">
                Fuel Temperature
              </p>
              <p className="mt-2 text-3xl font-bold text-gray-900">
                {state.fuel_temp.toFixed(1)}
              </p>
              <p className="text-xs text-gray-500">K (Kelvin)</p>
            </div>
            <div className="text-right">
              <div className={`h-12 w-12 rounded-full flex items-center justify-center ${
                fuelTempSafe ? "bg-green-100" : "bg-red-100"
              }`}>
                <span className={fuelTempSafe ? "text-green-700" : "text-red-700"}>
                  {fuelTempSafe ? "✓" : "⚠"}
                </span>
              </div>
            </div>
          </div>
          <div className="mt-3">
            <div className="flex justify-between text-xs text-gray-600">
              <span>Safe Zone</span>
              <span>≤1100K</span>
            </div>
            <div className="mt-1 h-2 overflow-hidden rounded-full bg-gray-200">
              <div
                className={`h-full ${fuelTempSafe ? "bg-green-500" : "bg-red-500"}`}
                style={{
                  width: `${Math.min((state.fuel_temp / 1200) * 100, 100)}%`,
                }}
              />
            </div>
          </div>
        </div>

        {/* System Pressure */}
        <div className={`rounded-lg border-2 p-4 ${
          pressureSafe
            ? "border-blue-300 bg-blue-50"
            : "border-yellow-300 bg-yellow-50"
        }`}>
          <div>
            <p className="text-xs font-semibold text-gray-600 uppercase tracking-wider">
              System Pressure
            </p>
            <p className="mt-2 text-3xl font-bold text-gray-900">
              {state.pressure.toFixed(1)}
            </p>
            <p className="text-xs text-gray-500">bar</p>
          </div>
          <div className="mt-3">
            <div className="flex justify-between text-xs text-gray-600">
              <span>Safe Zone</span>
              <span>8-12 bar</span>
            </div>
            <div className="mt-1 h-2 overflow-hidden rounded-full bg-gray-200">
              <div
                className={`h-full ${pressureSafe ? "bg-blue-500" : "bg-yellow-500"}`}
                style={{
                  width: `${Math.min((state.pressure / 15) * 100, 100)}%`,
                }}
              />
            </div>
          </div>
        </div>

        {/* Coolant Temperature */}
        <div className={`rounded-lg border-2 p-4 ${
          coolantTempSafe
            ? "border-cyan-300 bg-cyan-50"
            : "border-orange-300 bg-orange-50"
        }`}>
          <div>
            <p className="text-xs font-semibold text-gray-600 uppercase tracking-wider">
              Coolant Temperature
            </p>
            <p className="mt-2 text-3xl font-bold text-gray-900">
              {state.coolant_temp.toFixed(1)}
            </p>
            <p className="text-xs text-gray-500">K</p>
          </div>
          <div className="mt-3">
            <div className="flex justify-between text-xs text-gray-600">
              <span>Safe Zone</span>
              <span>280-310K</span>
            </div>
            <div className="mt-1 h-2 overflow-hidden rounded-full bg-gray-200">
              <div
                className={`h-full ${coolantTempSafe ? "bg-cyan-500" : "bg-orange-500"}`}
                style={{
                  width: `${Math.min((state.coolant_temp / 330) * 100, 100)}%`,
                }}
              />
            </div>
          </div>
        </div>

        {/* Reactor Power */}
        <div className={`rounded-lg border-2 p-4 ${
          powerSafe
            ? "border-purple-300 bg-purple-50"
            : "border-pink-300 bg-pink-50"
        }`}>
          <div>
            <p className="text-xs font-semibold text-gray-600 uppercase tracking-wider">
              Reactor Power
            </p>
            <p className="mt-2 text-3xl font-bold text-gray-900">
              {state.power.toFixed(2)}
            </p>
            <p className="text-xs text-gray-500">MW</p>
          </div>
          <div className="mt-3">
            <div className="flex justify-between text-xs text-gray-600">
              <span>Safe Zone</span>
              <span>0.8-1.2 MW</span>
            </div>
            <div className="mt-1 h-2 overflow-hidden rounded-full bg-gray-200">
              <div
                className={`h-full ${powerSafe ? "bg-purple-500" : "bg-pink-500"}`}
                style={{
                  width: `${Math.min((state.power / 1.5) * 100, 100)}%`,
                }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Secondary Parameters */}
      <div className="grid grid-cols-3 gap-3">
        {/* Power Rate of Change */}
        <div className="rounded-lg border border-gray-300 bg-white p-4">
          <p className="text-xs font-semibold text-gray-600 uppercase tracking-wider">
            Power Rate of Change
          </p>
          <p className="mt-2 text-2xl font-bold text-gray-900">
            {state.power_rate > 0 ? "+" : ""}{state.power_rate.toFixed(4)}
          </p>
          <p className="text-xs text-gray-500">MW/s</p>
          <p className="mt-2 text-xs text-gray-600">
            {Math.abs(state.power_rate) < 0.01
              ? "🟢 Stable"
              : state.power_rate > 0
              ? "📈 Increasing"
              : "📉 Decreasing"}
          </p>
        </div>

        {/* Temperature Rate of Change */}
        <div className="rounded-lg border border-gray-300 bg-white p-4">
          <p className="text-xs font-semibold text-gray-600 uppercase tracking-wider">
            Temperature Rate
          </p>
          <p className="mt-2 text-2xl font-bold text-gray-900">
            {state.temp_rate > 0 ? "+" : ""}{state.temp_rate.toFixed(4)}
          </p>
          <p className="text-xs text-gray-500">K/s</p>
          <p className="mt-2 text-xs text-gray-600">
            {Math.abs(state.temp_rate) < 1
              ? "🟢 Controlled"
              : state.temp_rate > 0
              ? "⬆️ Rising"
              : "⬇️ Falling"}
          </p>
        </div>

        {/* Simulation Time */}
        <div className="rounded-lg border border-gray-300 bg-white p-4">
          <p className="text-xs font-semibold text-gray-600 uppercase tracking-wider">
            Simulation Time
          </p>
          <p className="mt-2 text-2xl font-bold text-gray-900">
            {state.time.toFixed(1)}s
          </p>
          <p className="text-xs text-gray-500">elapsed</p>
          <p className="mt-2 text-xs text-gray-600">
            {lastActionTime && `Last update: ${(Date.now() - lastActionTime)}ms ago`}
          </p>
        </div>
      </div>

      {/* Explanation Panel */}
      <div className="rounded-lg border border-amber-200 bg-amber-50 p-4">
        <p className="text-xs font-bold text-amber-900 uppercase">📋 Parameter Guide</p>
        <div className="mt-3 grid grid-cols-4 gap-4 text-xs text-gray-700">
          <div>
            <p className="font-semibold">Fuel Temperature</p>
            <p className="mt-1">Critical safety parameter. Must stay below 1100K to prevent fuel damage.</p>
          </div>
          <div>
            <p className="font-semibold">System Pressure</p>
            <p className="mt-1">Primary coolant pressure. Normal range: 8-12 bar. Controls coolant circulation.</p>
          </div>
          <div>
            <p className="font-semibold">Coolant Temperature</p>
            <p className="mt-1">Heat removal efficiency indicator. Safe range: 280-310K. Higher = better cooling.</p>
          </div>
          <div>
            <p className="font-semibold">Reactor Power</p>
            <p className="mt-1">Current thermal power output. Target: 1.0 MW nominal. Range: 0.8-1.2 MW safe.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SafetyParameters;
