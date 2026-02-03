export interface ChatRequest {
  question?: string;
  messages?: Array<{
    role: "user" | "assistant" | "system";
    content: string;
  }>;
  k?: number;
}

export interface ChatResponse {
  response: string;
}
