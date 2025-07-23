import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { nanoid } from "nanoid";
import { sendMessage } from "@/lib/api";
import type { ChatMessage } from "@/types/chat";

export const useChat = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);

  const mutation = useMutation({
    mutationFn: sendMessage,
    onSuccess: (data) => {
      setMessages((prev) => [
        ...prev,
        { id: nanoid(), sender: "bot", text: data.reply, timestamp: new Date().toISOString() },
      ]);
    },
    onError: () => {
      setMessages((prev) => [
        ...prev,
        { id: nanoid(), sender: "bot", text: "Error contacting server.", timestamp: new Date().toISOString() },
      ]);
    },
  });

  const send = (text: string) => {
    setMessages((prev) => [
      ...prev,
      { id: nanoid(), sender: "user", text, timestamp: new Date().toISOString() },
    ]);
    mutation.mutate({ message: text });
  };

  return { messages, send, isLoading: mutation.isPending };
};
