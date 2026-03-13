"use client";

import { useState, useRef, useEffect } from "react";

// Define the structure of a chat message
type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: string[];
};

function getArxivSearchUrl(source: string) {
  const normalizedSource = source.replace(/^Processing:\s*/i, "").trim();
  const params = new URLSearchParams({
    query: normalizedSource,
    searchtype: "all",
    source: "header",
  });

  return `https://arxiv.org/search/?${params.toString()}`;
}

// Animated loading indicator component
function LoadingIndicator({ step }: { step: number }) {
  const steps = [
    "Searching knowledge base",
    "Retrieving relevant documents",
    "Analyzing context",
    "Generating response with Llama 3",
  ];

  return (
    <div className="self-start bg-blue-50 text-gray-800 p-5 rounded-2xl rounded-bl-none shadow-md border-2 border-blue-200">
      <div className="flex items-center gap-3">
        <div className="flex gap-1.5">
          <span
            className="w-3 h-3 bg-blue-600 rounded-full animate-bounce"
            style={{ animationDelay: "0ms", animationDuration: "1s" }}
          ></span>
          <span
            className="w-3 h-3 bg-blue-600 rounded-full animate-bounce"
            style={{ animationDelay: "150ms", animationDuration: "1s" }}
          ></span>
          <span
            className="w-3 h-3 bg-blue-600 rounded-full animate-bounce"
            style={{ animationDelay: "300ms", animationDuration: "1s" }}
          ></span>
        </div>
        <span className="font-semibold text-base">{steps[step]}</span>
      </div>
    </div>
  );
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState(0);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Generate a random session ID on load
  const [sessionId, setSessionId] = useState(() =>
    Math.random().toString(36).substring(2, 15),
  );

  // Auto-scroll to the bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Cycle through loading messages to show progress
  useEffect(() => {
    if (!loading) {
      setLoadingStep(0);
      return;
    }

    const steps = [
      "Searching knowledge base",
      "Retrieving relevant documents",
      "Analyzing context",
      "Generating response with Llama 3",
    ];

    const interval = setInterval(() => {
      setLoadingStep((prev) => (prev + 1) % steps.length);
    }, 2000); // Change message every 2 seconds

    return () => clearInterval(interval);
  }, [loading]);

  // Calculate remaining follow-ups
  const userMessageCount = messages.filter((m) => m.role === "user").length;
  // If 0 messages, 3 remaining. Otherwise, subtract (user messages - 1 initial question)
  const followUpsRemaining =
    userMessageCount === 0 ? 3 : 3 - (userMessageCount - 1);
  const isLimitReached = userMessageCount > 0 && followUpsRemaining <= 0;

  const handleAsk = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query || isLimitReached) return;

    const userQuery = query;
    setQuery(""); // Clear input box early for better UX

    // Add user question to the chat history UI immediately
    const updatedMessages: Message[] = [
      ...messages,
      { role: "user", content: userQuery },
    ];
    setMessages(updatedMessages);
    setLoading(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: userQuery,
          session_id: sessionId, // Send session ID for backend to track conversation
          // chat_history: messages.map((m) => ({
          //   role: m.role,
          //   content: m.content,
          // })), // Send previous history
        }),
      });
      if (!res.ok) {
        let errorMessage = `Backend returned HTTP ${res.status}`;
        try {
          const errorBody = await res.json();
          if (errorBody?.detail) {
            errorMessage = String(errorBody.detail);
          }
        } catch {
          // No JSON body is fine; keep default HTTP-based error.
        }
        throw new Error(errorMessage);
      }

      const data = await res.json();

      console.log("[BACKEND RESPONSE]", JSON.stringify(data, null, 2));
      console.log("[SOURCES RECEIVED]", data.sources);
      console.log("[RESPONSE TEXT]", data.response?.slice(0, 200));

      // Append AI response to chat history
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.response,
          sources: data.sources || [],
        },
      ]);
    } catch (error) {
      console.error("Error fetching data:", error);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            error instanceof Error
              ? `Error connecting to the backend: ${error.message}`
              : "Error connecting to the backend.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setMessages([]);
    setQuery("");
    // Generate a new Session ID so Redis treats it as a brand new chat!
    setSessionId(Math.random().toString(36).substring(2, 15));
  };

  return (
    <main className="flex min-h-screen flex-col items-center p-4 md:p-10 bg-gray-50 text-gray-900">
      <div className="w-full max-w-4xl flex flex-col gap-4 h-[90vh]">
        {/* Header & Badges */}
        <div className="flex justify-between items-center bg-white p-4 rounded-xl shadow-sm border">
          <h1 className="text-2xl font-bold">AI Research Copilot</h1>
          <div className="flex items-center gap-4">
            {userMessageCount > 0 && (
              <span
                className={`text-sm font-semibold px-3 py-1 rounded-full ${followUpsRemaining > 0 ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"}`}
              >
                Follow-ups left: {followUpsRemaining}
              </span>
            )}
            <button
              onClick={handleReset}
              className="text-sm text-gray-500 hover:text-gray-800 underline"
            >
              Reset Chat
            </button>
          </div>
        </div>

        {/* Chat Display Area */}
        <div className="flex-1 bg-white border rounded-xl p-4 shadow-sm overflow-y-auto flex flex-col gap-4">
          {messages.length === 0 ? (
            <div className="text-gray-400 text-center my-auto flex flex-col items-center justify-center h-full">
              <span className="text-4xl mb-2">🤖</span>
              Ask your first research question...
            </div>
          ) : (
            messages.map((msg, index) => (
              <div
                key={index}
                className={`flex flex-col max-w-[85%] ${msg.role === "user" ? "self-end items-end" : "self-start items-start"}`}
              >
                <div
                  className={`p-4 rounded-2xl ${msg.role === "user" ? "bg-blue-600 text-white rounded-br-none" : "bg-gray-100 text-gray-900 rounded-bl-none"}`}
                >
                  <p className="whitespace-pre-wrap leading-relaxed">
                    {msg.content}
                  </p>
                </div>

                {/* Citations (Only show on assistant messages if they exist) */}
                {msg.role === "assistant" &&
                  msg.sources &&
                  msg.sources.length > 0 && (
                    <div className="mt-1 pl-2">
                      <span className="text-xs font-semibold text-gray-500">
                        Sources:{" "}
                      </span>
                      <span className="text-xs text-gray-400">
                        {msg.sources.map((source, sourceIndex) => (
                          <span key={`${source}-${sourceIndex}`}>
                            <a
                              href={getArxivSearchUrl(source)}
                              target="_blank"
                              rel="noreferrer"
                              className="text-blue-600 underline hover:text-blue-700"
                            >
                              {source}
                            </a>
                            {sourceIndex < msg.sources!.length - 1 ? ", " : ""}
                          </span>
                        ))}
                      </span>
                    </div>
                  )}
              </div>
            ))
          )}
          {loading && <LoadingIndicator step={loadingStep} />}
          <div ref={messagesEndRef} /> {/* Auto-scroll target */}
        </div>

        {/* Input Area */}
        {isLimitReached ? (
          <div className="bg-orange-100 border border-orange-300 text-orange-800 p-4 rounded-xl text-center shadow-sm">
            <p className="mb-3 font-medium">
              You have reached the maximum follow-ups for this context.
            </p>
            <button
              onClick={handleReset}
              className="bg-orange-600 text-white px-6 py-2 rounded-lg font-semibold hover:bg-orange-700 transition-colors"
            >
              Start New Topic
            </button>
          </div>
        ) : (
          <form onSubmit={handleAsk} className="flex gap-2">
            <input
              type="text"
              className="flex-1 p-4 border rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder={
                messages.length === 0
                  ? "Ask a research question..."
                  : "Ask a follow-up question..."
              }
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading || !query}
              className="bg-blue-600 text-white px-8 py-4 rounded-xl font-semibold hover:bg-blue-700 disabled:opacity-50 transition-colors"
            >
              Send
            </button>
          </form>
        )}
      </div>
    </main>
  );
}
