import React, { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
// import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
// import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger, DropdownMenuSeparator, DropdownMenuLabel } from "@/components/ui/dropdown-menu";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
// import { Label } from "@/components/ui/label";
// import { Switch } from "@/components/ui/switch";
// import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
// import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
// import { Separator } from "@/components/ui/separator";
import { toast } from "sonner";
import { Send, StopCircle, Copy, Check, Sun, Moon } from "lucide-react";
import ReactMarkdown, { Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github-dark.css";

/**
 * PRODUCTION-GRADE, SINGLE-FILE CHAT UI (TypeScript + React)
 * ----------------------------------------------------------
 * Drop this component into your app. It provides:
 *  - Message list with avatars & markdown (including code highlighting)
 *  - Streaming support (SSE or fetch reader)
 *  - File attachments (sends as multipart/form-data or base64 JSON)
 *  - Stop/Retry, Clear chat, System prompt, Temperature control
 *  - Light/Dark toggle
 *  - Clean shadcn/ui + Tailwind styling
 * 
 * Wire up `BACKEND_URL` and (if needed) auth headers in `createClient()`
 * and adjust the payload to match your colleagues' API.
 */

// ==== CONFIG: point this to your backend ====
const BACKEND_URL = import.meta.env.VITE_CHAT_API_URL ?? "/api/chat"; // e.g. https://api.yourdomain.com/chat
const USE_SSE = true; // flip to false if your backend streams via fetch body instead of SSE

// Optional: add tokens/headers here
function createClient() {
  const headers: Record<string, string> = {
    "Accept": USE_SSE ? "text/event-stream" : "application/json",
  };
  // Example: if you need auth
  const token = localStorage.getItem("auth_token");
  if (token) headers["Authorization"] = `Bearer ${token}`;
  return { headers };
}

// ==== Types ====
export type Role = "system" | "user" | "assistant" | "tool";
export type ChatMessage = {
  id: string;
  role: Role;
  content: string;
  createdAt: number;
};

// Utility: simple id
const rid = () => Math.random().toString(36).slice(2);

// ==== Markdown components ====
function CodeBlock({ inline, className, children }: { inline?: boolean; className?: string; children: React.ReactNode }) {
  const ref = useRef<HTMLPreElement | null>(null);
  const [copied, setCopied] = useState(false);
  const lang = (className || "").replace("language-", "") || "";

  const text = useMemo(() => {
    if (!children) return "";
    return Array.isArray(children) ? children.join("") : String(children);
  }, [children]);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {
      toast.error("Failed to copy");
    }
  };

  if (inline) return <code className="px-1 py-0.5 rounded bg-muted text-sm">{children}</code>;
  return (
    <div className="relative group">
      <pre ref={ref} className={`overflow-auto rounded-lg p-4 bg-[#0d1117] text-white text-sm` + (lang ? ` language-${lang}` : "") }>
        <code className={className}>{children}</code>
      </pre>
      <Button variant="secondary" size="icon" onClick={copy} className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
        {copied ? <Check className="h-4 w-4"/> : <Copy className="h-4 w-4"/>}
      </Button>
    </div>
  );
}

// ==== Message Bubble ====
function MessageBubble({ m }: { m: ChatMessage }) {
  const isUser = m.role === "user";
  const initials = isUser ? "U" : m.role === "assistant" ? "A" : "S";
  return (
    <div className={`flex gap-3 ${isUser ? "justify-end" : "justify-start"}`}>
      {!isUser && (
        <Avatar className="h-8 w-8">
          <AvatarFallback>🤖</AvatarFallback>
        </Avatar>
      )}
      <div className={`max-w-[80%] rounded-2xl px-4 py-3 border ${isUser ? "bg-primary text-primary-foreground ml-auto" : "bg-card"}`}>
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[rehypeHighlight]}
          components={{ code: CodeBlock as Components['code'] }}
        >
          {m.content}
        </ReactMarkdown>
  {/* ...existing code... */}
      </div>
      {isUser && (
        <Avatar className="h-8 w-8">
          <AvatarFallback>{initials}</AvatarFallback>
        </Avatar>
      )}
    </div>
  );
}

// ==== Main Component ====
export default function Chatbot() {
  const [messages, setMessages] = useState<ChatMessage[]>([{
    id: rid(), role: "assistant", content: "Hi! Ask me anything.", createdAt: Date.now()
  }]);
  const [input, setInput] = useState("");
  // const [systemPrompt, setSystemPrompt] = useState("You are a helpful, concise assistant.");
  // const [temperature, setTemperature] = useState(0.7);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isDark, setIsDark] = useState(false);
  // ...existing code...
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", isDark);
  }, [isDark]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages.length, isStreaming]);

  // ...existing code...

  // const clearChat = () => {
  //   if (isStreaming) return;
  //   setMessages([{ id: rid(), role: "assistant", content: "Cleared. How can I help?", createdAt: Date.now() }]);
  // };

  async function sendMessage() {
    const content = input.trim();
    if (!content || isStreaming) return;

  const userMsg: ChatMessage = { id: rid(), role: "user", content, createdAt: Date.now() };
    const asstMsg: ChatMessage = { id: rid(), role: "assistant", content: "", createdAt: Date.now() };
    setMessages((prev) => [...prev, userMsg, asstMsg]);
    setInput("");
  // ...existing code...

    try {
      setIsStreaming(true);
      if (USE_SSE) {
        await streamViaSSE([...messages, userMsg], asstMsg.id);
      } else {
        await streamViaReader([...messages, userMsg], asstMsg.id);
      }
    } catch (e: unknown) {
      console.error(e);
      let errorMsg = "Request failed";
      if (typeof e === "object" && e !== null && "message" in e) {
        errorMsg = (e as { message?: string }).message || errorMsg;
      }
      toast.error(errorMsg);
      // write error into assistant bubble
      setMessages((prev) => prev.map((m) => m.id === asstMsg.id ? { ...m, content: `⚠️ ${errorMsg}` } : m));
    } finally {
      setIsStreaming(false);
    }
  }

  function buildPayload(history: ChatMessage[]) {
    // Transform your local messages into your backend schema
    // Adjust keys to match your API contract
    const payload: {
      messages: { role: Role; content: string }[];
    } = {
      messages: history.map((m) => ({ role: m.role, content: m.content })),
    };
    return payload;
  }

  async function streamViaSSE(history: ChatMessage[], asstId: string) {
    const payload = buildPayload(history);
    const { headers } = createClient();

  // ...existing code...

    const url = BACKEND_URL + (BACKEND_URL.includes("?") ? "&" : "?") + "stream=1";
    const res = await fetch(url, {
      method: "POST",
      headers: { ...headers, "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    // If the backend returned a single JSON object (non-streaming), handle
    // it as a fallback: parse JSON and append the full response text.
    const contentType = (res.headers.get("content-type") || "").toLowerCase();
    if (contentType.includes("application/json")) {
      const json = await res.json();
      const text = (json && (json.response || json.result || json.text)) || JSON.stringify(json);
      appendAssistant(asstId, String(text));
      return;
    }

    const reader = (res.body as ReadableStream<Uint8Array>).getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split(/\n/);
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") return;
        try {
          const { delta, error } = JSON.parse(data);
          if (error) throw new Error(error);
          if (delta) appendAssistant(asstId, delta);
        } catch {
          // Fallback: treat as plain text
          appendAssistant(asstId, data);
        }
      }
    }
  }

  async function streamViaReader(history: ChatMessage[], asstId: string) {
    const payload = buildPayload(history);
    const { headers } = createClient();

    // For non-SSE chunked text
    const res = await fetch(BACKEND_URL, {
      method: "POST",
      headers: { ...headers, "Content-Type": "application/json", "Accept": "text/plain" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const contentType = (res.headers.get("content-type") || "").toLowerCase();
    if (contentType.includes("application/json")) {
      const json = await res.json();
      const text = (json && (json.response || json.result || json.text)) || JSON.stringify(json);
      appendAssistant(asstId, String(text));
      return;
    }

    if (!res.body) throw new Error("Empty response body");

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      appendAssistant(asstId, decoder.decode(value));
    }
  }

  function appendAssistant(id: string, chunk: string) {
    setMessages((prev) => prev.map((m) => m.id === id ? { ...m, content: m.content + chunk } : m));
  }

  async function stopStreaming() {
    // This implementation uses fetch streams which can't be cancelled cleanly without AbortController.
    // For production, wrap fetch with AbortController and call controller.abort() here.
    toast("Stop requested. New messages will cancel existing streams if you implement AbortController.");
  }

  // ...existing code...

  return (
    <TooltipProvider>
      <div className="mx-auto max-w-5xl p-4 md:p-6 space-y-4">
        <header className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <img src="/logo.png" alt="Logo" className="h-8 w-auto mr-2" />
            <Badge variant={isStreaming ? "default" : "secondary"} className="ml-1">
              {isStreaming ? "Streaming" : "Idle"}
            </Badge>
          </div>
          <div className="flex items-center gap-2">
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="outline" size="icon" onClick={() => setIsDark((d) => !d)}>
                  {isDark ? <Sun className="h-4 w-4"/> : <Moon className="h-4 w-4"/>}
                </Button>
              </TooltipTrigger>
              <TooltipContent>Toggle theme</TooltipContent>
            </Tooltip>
          </div>
        </header>

        <Card className="border rounded-2xl overflow-hidden">
          <CardHeader className="pb-0">
            <CardTitle className="text-base">Conversation</CardTitle>
          </CardHeader>
          <CardContent className="pt-4">
            <div className="border rounded-xl p-2 bg-muted/30">
              <ScrollArea className="h-[55vh]" ref={scrollRef as React.RefObject<HTMLDivElement>}>
                <div className="p-3 space-y-4">
                  {messages.map((m) => (
                    <MessageBubble key={m.id} m={m} />
                  ))}
                </div>
              </ScrollArea>
              <div className="p-3 border-t mt-2">
                {/* ...existing code... */}
                <div className="flex items-end gap-2">
                  <div className="flex-1">
                    <Textarea
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      rows={3}
                      placeholder="Ask your question…"
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
                      }}
                    />
                    {/* ...existing code... */}
                  </div>
                  <div className="flex gap-2">
                    {/* Clear button removed */}
                    {isStreaming ? (
                      <Button variant="destructive" onClick={stopStreaming}>
                        <StopCircle className="h-4 w-4 mr-1"/>Stop
                      </Button>
                    ) : (
                      <Button onClick={sendMessage} disabled={!input.trim()}>
                        <Send className="h-4 w-4 mr-1"/>Send
                      </Button>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

  {/* Footer removed as requested */}
      </div>
    </TooltipProvider>
  );
}
