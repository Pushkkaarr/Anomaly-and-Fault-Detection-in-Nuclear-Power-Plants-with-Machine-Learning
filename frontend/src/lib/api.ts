import axios, { AxiosError } from "axios";
import { Action, FaultPrediction, Model, ReactorState, Scenario } from "@/types/reactor";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

type ApiEnvelope<T> = {
  success: boolean;
  status_code: number;
  message: string;
  data: T;
  errors?: string[];
};

type SimulationPayload = {
  reactor_state: ReactorState;
  fault_prediction?: FaultPrediction;
  reward?: number;
  done?: boolean;
  episode_step: number;
  action?: Action;
  is_running?: boolean;
  message?: string;
  model_id?: string;
  scenario_id?: string;
};

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000,
});

function unwrap<T>(response: { data: ApiEnvelope<T> }): T {
  return response.data.data;
}

export function getErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError<ApiEnvelope<unknown>>;
    return axiosError.response?.data?.message || axiosError.message || "Request failed";
  }

  if (error instanceof Error) {
    return error.message;
  }

  return "Unknown error";
}

export const apiClient = {
  async checkHealth(): Promise<{ status: string }> {
    const response = await api.get<ApiEnvelope<{ status: string }>>("/health");
    return unwrap(response);
  },

  async getModels(): Promise<Model[]> {
    const response = await api.get<ApiEnvelope<{ models: Model[] }>>("/models");
    return unwrap(response).models;
  },

  async loadModel(modelId: string): Promise<Model> {
    const response = await api.post<ApiEnvelope<Model>>(`/models/${modelId}/load`);
    return unwrap(response);
  },

  async getScenarios(): Promise<Scenario[]> {
    const response = await api.get<ApiEnvelope<{ scenarios: Scenario[] }>>("/scenarios");
    return unwrap(response).scenarios;
  },

  async startSimulation(modelId: string, scenarioId: string): Promise<SimulationPayload> {
    const response = await api.post<ApiEnvelope<SimulationPayload>>("/simulation/start", {
      model_id: modelId,
      scenario_id: scenarioId,
    });
    return unwrap(response);
  },

  async stepSimulation(): Promise<SimulationPayload> {
    const response = await api.post<ApiEnvelope<SimulationPayload>>("/simulation/step");
    return unwrap(response);
  },

  async manualAction(action: Action): Promise<SimulationPayload> {
    const response = await api.post<ApiEnvelope<SimulationPayload>>("/simulation/action", {
      rod_action: action.control_rod,
      flow_action: action.coolant_flow,
    });
    return unwrap(response);
  },

  async stopSimulation(): Promise<{ summary: Record<string, number>; step: number }> {
    const response = await api.post<ApiEnvelope<{ summary: Record<string, number>; step: number }>>("/simulation/stop");
    return unwrap(response);
  },

  async resetSimulation(): Promise<SimulationPayload> {
    const response = await api.post<ApiEnvelope<SimulationPayload>>("/simulation/reset");
    return unwrap(response);
  },
};
