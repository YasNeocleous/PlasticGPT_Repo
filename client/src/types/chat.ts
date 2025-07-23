export interface ChatMessage {
  id: string;
  sender: "user" | "bot";
  text: string;
  timestamp: string;
}


export interface ChatRequest {
  message: string;
}

export interface ChatResponse {
  reply: string;
}
