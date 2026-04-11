import { create } from "zustand";
import {
  SimulationState,
  ReactorState,
  SimulationEvent,
  SimulationMetrics,
  Action,
  FaultPrediction,
} from "@/types/reactor";

const INITIAL_SIMULATION_STATE: SimulationState = {
  is_running: false,
  is_paused: false,
  current_model: null,
  current_scenario: null,
  reactor_state: null,
  metrics: null,
  episode_step: 0,
  events: [],
  error_message: null,
  last_action: null,
  fault_prediction: null,
  _event_counter: 0,
  history: [],
  current_reward: 0,
};

const INITIAL_METRICS: SimulationMetrics = {
  total_reward: 0,
  episode_steps: 0,
  episode_duration: 0,
  max_fuel_temp: 0,
  max_coolant_temp: 0,
  avg_pressure: 0,
  power_change_rate: 0,
  safety_events: 0,
};

interface SimulationStore extends SimulationState {
  // State Updates
  setIsRunning: (isRunning: boolean) => void;
  setIsPaused: (isPaused: boolean) => void;
  setCurrentModel: (modelId: string | null) => void;
  setCurrentScenario: (scenarioId: string | null) => void;
  setReactorState: (state: ReactorState) => void;
  setMetrics: (metrics: SimulationMetrics) => void;
  setEpisodeStep: (step: number) => void;
  setErrorMessage: (message: string | null) => void;
  setLastAction: (action: Action | null) => void;
  setFaultPrediction: (prediction: FaultPrediction | null) => void;
  setCurrentReward: (reward: number) => void;

  // History for live graph (circular buffer of 80 points)
  addStateToHistory: (state: ReactorState) => void;
  clearHistory: () => void;

  // Event Management (circular buffer of 100 events)
  addEvent: (event: Omit<SimulationEvent, 'id'>) => void;
  clearEvents: () => void;
  getEventLog: () => SimulationEvent[];

  // Reset Entire State
  reset: () => void;
}

export const useSimulationStore = create<SimulationStore>((set, get) => ({
  ...INITIAL_SIMULATION_STATE,

  setIsRunning: (isRunning) => set({ is_running: isRunning }),
  setIsPaused: (isPaused) => set({ is_paused: isPaused }),
  setCurrentModel: (modelId) => set({ current_model: modelId }),
  setCurrentScenario: (scenarioId) => set({ current_scenario: scenarioId }),
  setReactorState: (state) => set({ reactor_state: state }),
  setMetrics: (metrics) => set({ metrics }),
  setEpisodeStep: (step) => set({ episode_step: step }),
  setErrorMessage: (message) => set({ error_message: message }),
  setLastAction: (action) => set({ last_action: action }),
  setFaultPrediction: (prediction) => set({ fault_prediction: prediction }),
  setCurrentReward: (reward) => set({ current_reward: reward }),

  addStateToHistory: (state) => {
    const currentHistory = get().history || [];
    const newHistory = [...currentHistory, state].slice(-80); // Keep last 80 points
    set({ history: newHistory });
  },

  clearHistory: () => set({ history: [] }),

  addEvent: (event) => {
    const currentEvents = get().events;
    const counter = get()._event_counter ?? 0;
    
    const newEvent: SimulationEvent = {
      ...event,
      id: `event-${counter}`,
    };
    
    const newEvents = [newEvent, ...currentEvents].slice(0, 100);
    set({ events: newEvents, _event_counter: counter + 1 });
  },

  clearEvents: () => set({ events: [] }),

  getEventLog: () => get().events,

  reset: () => {
    set({
      ...INITIAL_SIMULATION_STATE,
      metrics: INITIAL_METRICS,
      history: [],
      current_reward: 0,
      _event_counter: 0,
    });
  },
}));

export function useSimulation() {
  return useSimulationStore();
}
