/**
 * Type definitions for Nuclear Reactor Control System
 * Synced with Backend API response formats
 */

export interface ReactorState {
  power: number;       // Reactor power (normalized ~1.0)
  precursors: number;  // Delayed neutron precursor concentration
  fuel_temp: number;   // Fuel temperature (K)
  coolant_temp: number; // Coolant temperature (K)
  pressure: number;    // System pressure (bar)
  coolant_flow_actual?: number; // Actual physical coolant flow (kg/s)
  power_rate: number;  // Rate of power change
  temp_rate: number;   // Rate of temperature change
  time: number;        // Simulation time (s)
}

export interface Action {
  control_rod: number;  // Control rod position (-1.0 to 1.0)
  coolant_flow: number; // Coolant flow rate (-1.0 to 1.0)
}

export interface Model {
  id: string;
  name: string;
  description: string;
  training_steps: number;
  reward_per_step: number;
  status: string;
}

export interface Scenario {
  id: string;
  name: string;
  description: string;
  difficulty?: string;
  trigger_time?: number;
  parameters?: Record<string, unknown>;
}

export interface SimulationEvent {
  id: string;
  timestamp: number;
  type: "info" | "warning" | "critical" | "success";
  message: string;
  icon?: string;
}

export interface SimulationMetrics {
  total_reward: number;
  episode_steps: number;
  episode_duration: number;
  max_fuel_temp: number;
  max_coolant_temp: number;
  avg_pressure: number;
  power_change_rate: number;
  safety_events: number;
}

export interface FaultPrediction {
  status: "insufficient_data" | "prediction_ready" | "error";
  message?: string;
  predicted_class?: number;
  predicted_state?: "Normal" | "Scram" | "LOFA";
  confidence?: number;
  confidence_level?: "low" | "medium" | "high";
  class_probabilities?: {
    Normal: number;
    Scram: number;
    LOFA: number;
  };
  risk_level?: "low" | "medium" | "high" | "critical";
  recommendations?: string[];
}

export interface SimulationState {
  is_running: boolean;
  is_paused: boolean;
  current_model: string | null;
  current_scenario: string | null;
  reactor_state: ReactorState | null;
  metrics: SimulationMetrics | null;
  episode_step: number;
  events: SimulationEvent[];
  error_message: string | null;
  last_action: Action | null;
  fault_prediction: FaultPrediction | null;
  _event_counter?: number;
  history: ReactorState[];      // Rolling history for live graphs
  current_reward: number;
}

export interface HealthResponse {
  status: string;
  message: string;
  timestamp: string;
}

export interface StatusResponse {
  status: string;
  available_models: string[];
  available_scenarios: string[];
  is_simulation_running: boolean;
  last_simulation_model: string | null;
}

export interface StateResponse {
  reactor_state: ReactorState;
  episode_step: number;
  is_running: boolean;
}

export interface StepResponse {
  reactor_state: ReactorState;
  fault_prediction?: FaultPrediction;
  reward: number;
  done: boolean;
  episode_step: number;
  action?: Action;
  info?: Record<string, unknown>;
}

export interface SimulationSummary {
  total_reward: number;
  episode_steps: number;
  episode_duration: number;
  max_fuel_temp: number;
  max_coolant_temp: number;
  avg_pressure: number;
  initial_state: ReactorState;
  final_state: ReactorState;
  scenario_used: string;
  model_used: string;
}

export interface ApiError {
  error: string;
  message: string;
  status_code: number;
  details?: Record<string, unknown>;
}
