import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ChatWindow } from "@/components/ChatWindow";

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <div className="flex flex-col items-center mt-6">
        <img
          src="/logo.png"
          alt="Plastic Surgery GPT"
          className="h-24 mb-6 max-w-xs sm:max-w-sm rounded-xl shadow"
        />
        <ChatWindow />
      </div>
    </QueryClientProvider>
  );
}

export default App;
