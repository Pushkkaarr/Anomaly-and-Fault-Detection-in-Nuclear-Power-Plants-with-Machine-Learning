import { io, Socket } from "socket.io-client";

export type WebSocketMessage = {
  type: string;
  data?: unknown;
  timestamp?: string;
};

export type WebSocketConfig = {
  url?: string;
};

type Handler = (payload: any) => void;

export class WebSocketService {
  private socket: Socket | null = null;
  private handlers = new Map<string, Set<Handler>>();
  private config: WebSocketConfig;

  constructor(config: WebSocketConfig = {}) {
    this.config = config;
  }

  async connect(): Promise<void> {
    if (this.socket?.connected) {
      return;
    }

    const url = this.config.url || "http://localhost:8000";
    this.socket = io(url, {
      path: "/api/ws",
      transports: ["websocket", "polling"],
      autoConnect: true,
    });

    this.socket.onAny((event, payload) => {
      this.emitLocal(event, payload);
      this.emitLocal("*", payload);
    });

    await new Promise<void>((resolve, reject) => {
      this.socket?.once("connect", () => resolve());
      this.socket?.once("connect_error", (error) => reject(error));
    });
  }

  send(type: string, data: any): void {
    this.socket?.emit(type, data);
  }

  on(type: string, handler: Handler): () => void {
    const existing = this.handlers.get(type) || new Set<Handler>();
    existing.add(handler);
    this.handlers.set(type, existing);

    return () => {
      const current = this.handlers.get(type);
      current?.delete(handler);
      if (current && current.size === 0) {
        this.handlers.delete(type);
      }
    };
  }

  isConnected(): boolean {
    return !!this.socket?.connected;
  }

  getStatus() {
    return {
      connected: this.isConnected(),
      url: this.config.url || "http://localhost:8000",
      reconnectAttempts: 0,
    };
  }

  private emitLocal(type: string, payload: any): void {
    const handlers = this.handlers.get(type);
    if (!handlers) {
      return;
    }

    handlers.forEach((handler) => handler(payload));
  }
}

let singleton: WebSocketService | null = null;

export function getWebSocketService(config?: WebSocketConfig): WebSocketService {
  if (!singleton) {
    singleton = new WebSocketService(config);
  }

  return singleton;
}
