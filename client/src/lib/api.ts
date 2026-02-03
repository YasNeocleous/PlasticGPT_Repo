import type { ChatRequest, ChatResponse } from "@/types/chat";

const BACKEND_URL = import.meta.env.VITE_CHAT_API_URL ?? "/api/chat";

export const sendMessage = async (
  data: ChatRequest
): Promise<ChatResponse> => {
  const res = await fetch(BACKEND_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

  if (!res.ok) throw new Error("Failed to fetch chat response");
  return res.json();
};

// Example usage in a React component:
// import { sendMessage } from "@/lib/api";
// sendMessage({ question: "What is rhinoplasty?" })
//   .then(res => console.log(res.response))
//   .catch(err => console.error(err));
