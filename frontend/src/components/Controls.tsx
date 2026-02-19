"use client";

import React, { useState } from "react";
import { Model, Scenario, Action } from "@/types/reactor";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  Button,
  Select,
  Alert,
  Spinner,
} from "@/components/ui";
import { Play, Square, Pause, RotateCcw } from "lucide-react";

/**
 * Model Selection Component
 */
interface ModelSelectorProps {
  models: Model[];
  selectedModel: string | null;
  onSelect: (modelId: string) => void;
  isLoading?: boolean;
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  models,
  selectedModel,
  onSelect,
  isLoading,
}) => {
  // Handle null/undefined models
  const safeModels = models || [];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Choose Model</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <Select
          value={selectedModel || ""}
          onChange={(e: React.ChangeEvent<HTMLSelectElement>) => onSelect(e.target.value)}
          disabled={isLoading || safeModels.length === 0}
        >
          <option value="">{safeModels.length === 0 ? "No models available" : "Select a model..."}</option>
          {safeModels.map((model) => (
            <option key={model.id} value={model.id}>
              {model.name} ({model.reward_per_step.toFixed(1)} reward/step)
            </option>
          ))}
        </Select>

        {selectedModel && safeModels.length > 0 && (
          <div className="rounded-lg bg-blue-50 p-3">
            <p className="text-xs text-gray-600">
              {safeModels.find((m) => m.id === selectedModel)?.description}
            </p>
            <p className="mt-2 text-xs text-gray-600">
              Training Steps:{" "}
              <span className="font-semibold">
                {safeModels.find((m) => m.id === selectedModel)?.training_steps}
              </span>
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

/**
 * Scenario Selection Component
 */
interface ScenarioSelectorProps {
  scenarios: Scenario[];
  selectedScenario: string | null;
  onSelect: (scenarioId: string) => void;
  isLoading?: boolean;
}

export const ScenarioSelector: React.FC<ScenarioSelectorProps> = ({
  scenarios,
  selectedScenario,
  onSelect,
  isLoading,
}) => {
  // Handle null/undefined scenarios
  const safeScenarios = scenarios || [];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Choose Scenario</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <Select
          value={selectedScenario || ""}
          onChange={(e: React.ChangeEvent<HTMLSelectElement>) => onSelect(e.target.value)}
          disabled={isLoading || safeScenarios.length === 0}
        >
          <option value="">{safeScenarios.length === 0 ? "No scenarios available" : "Select a scenario..."}</option>
          {safeScenarios.map((scenario) => (
            <option key={scenario.id} value={scenario.id}>
              {scenario.name}
            </option>
          ))}
        </Select>

        {selectedScenario && safeScenarios.length > 0 && (
          <div className="rounded-lg bg-green-50 p-3">
            <p className="text-xs text-gray-600">
              {safeScenarios.find((s) => s.id === selectedScenario)?.description}
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

/**
 * Simulation Control Buttons
 */
interface ControlButtonsProps {
  isRunning: boolean;
  isPaused: boolean;
  isLoading: boolean;
  canStart: boolean;
  onStart: () => void;
  onStop: () => void;
  onPause?: () => void;
  onResume?: () => void;
  onReset?: () => void;
}

export const ControlButtons: React.FC<ControlButtonsProps> = ({
  isRunning,
  isPaused,
  isLoading,
  canStart,
  onStart,
  onStop,
  onPause,
  onResume,
  onReset,
}) => {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Controls</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <Button
          variant="primary"
          size="lg"
          className="w-full"
          onClick={onStart}
          disabled={isLoading || isRunning || !canStart}
        >
          {isLoading ? <Spinner size="sm" /> : <Play className="mr-2 h-4 w-4" />}
          Start Simulation
        </Button>

        {isRunning && (
          <>
            <Button
              variant="secondary"
              size="md"
              className="w-full"
              onClick={onPause}
              disabled={isPaused || isLoading}
            >
              <Pause className="mr-2 h-4 w-4" />
              Pause
            </Button>

            <Button
              variant="danger"
              size="md"
              className="w-full"
              onClick={onStop}
            >
              <Square className="mr-2 h-4 w-4" />
              Stop
            </Button>
          </>
        )}

        {onReset && (
          <Button
            variant="ghost"
            size="md"
            className="w-full"
            onClick={onReset}
            disabled={isLoading || isRunning}
          >
            <RotateCcw className="mr-2 h-4 w-4" />
            Reset
          </Button>
        )}
      </CardContent>
    </Card>
  );
};

/**
 * Manual Control Sliders
 */
interface ManualControlProps {
  onControlChange: (action: Action) => void;
  isEnabled: boolean;
}

export const ManualControl: React.FC<ManualControlProps> = ({
  onControlChange,
  isEnabled,
}) => {
  const [controlRod, setControlRod] = useState(0);
  const [coolantFlow, setCoolantFlow] = useState(0);

  const handleControlRodChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseFloat(e.target.value);
    setControlRod(value);
    onControlChange({ control_rod: value, coolant_flow: coolantFlow });
  };

  const handleCoolantFlowChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseFloat(e.target.value);
    setCoolantFlow(value);
    onControlChange({ control_rod: controlRod, coolant_flow: value });
  };

  return (
    <Card className={isEnabled ? "" : "opacity-50"}>
      <CardHeader>
        <CardTitle className="text-base">Manual Control</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6" onClick={(e) => !isEnabled && e.preventDefault()}>
        <div>
          <div className="mb-2 flex justify-between text-sm">
            <label className="font-semibold text-gray-700">
              Control Rod Position
            </label>
            <span className="rounded bg-blue-100 px-2 py-1 text-xs font-mono text-blue-900">
              {controlRod.toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            min="-1"
            max="1"
            step="0.05"
            value={controlRod}
            onChange={handleControlRodChange}
            disabled={!isEnabled}
            className="w-full"
          />
          <div className="mt-1 flex justify-between text-xs text-gray-500">
            <span>-1.0 (Retracted)</span>
            <span>0.0 (Center)</span>
            <span>1.0 (Inserted)</span>
          </div>
        </div>

        <div>
          <div className="mb-2 flex justify-between text-sm">
            <label className="font-semibold text-gray-700">
              Coolant Flow Rate
            </label>
            <span className="rounded bg-green-100 px-2 py-1 text-xs font-mono text-green-900">
              {coolantFlow.toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            min="-1"
            max="1"
            step="0.05"
            value={coolantFlow}
            onChange={handleCoolantFlowChange}
            disabled={!isEnabled}
            className="w-full"
          />
          <div className="mt-1 flex justify-between text-xs text-gray-500">
            <span>-1.0 (Decrease)</span>
            <span>0.0 (Normal)</span>
            <span>1.0 (Increase)</span>
          </div>
        </div>

        {!isEnabled && (
          <Alert type="info" className="py-2">
            <p className="text-xs">
              Manual control is only available during an active simulation.
            </p>
          </Alert>
        )}
      </CardContent>
    </Card>
  );
};

/**
 * Simulation Status Card
 */
interface SimulationStatusProps {
  isRunning: boolean;
  currentModel: string | null;
  currentScenario: string | null;
  episodeStep: number;
  totalReward?: number;
}

export const SimulationStatus: React.FC<SimulationStatusProps> = ({
  isRunning,
  currentModel,
  currentScenario,
  episodeStep,
  totalReward,
}) => {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Status</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 text-sm">
        <div className="flex justify-between">
          <span className="text-gray-600">State:</span>
          <span
            className={`font-semibold ${isRunning ? "text-green-600" : "text-gray-600"}`}
          >
            {isRunning ? "🟢 Running" : "⚪ Idle"}
          </span>
        </div>

        <div className="flex justify-between">
          <span className="text-gray-600">Model:</span>
          <span className="font-mono text-xs">
            {currentModel || "None"}
          </span>
        </div>

        <div className="flex justify-between">
          <span className="text-gray-600">Scenario:</span>
          <span className="font-mono text-xs">
            {currentScenario || "None"}
          </span>
        </div>

        <div className="flex justify-between">
          <span className="text-gray-600">Step:</span>
          <span className="font-semibold">{episodeStep}</span>
        </div>

        {totalReward !== undefined && (
          <div className="border-t border-gray-200 pt-2">
            <div className="flex justify-between">
              <span className="text-gray-600">Total Reward:</span>
              <span
                className={`font-bold ${totalReward >= 0 ? "text-green-600" : "text-red-600"}`}
              >
                {totalReward.toFixed(2)}
              </span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};
