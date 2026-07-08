"use client";

import Image from "next/image";
import { useState, useRef, useEffect } from "react";

type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: string[];
};

const SAMPLE_PROMPTS: string[] = [
  "What are recent ArXiv papers on retrieval-augmented generation?",
  "Summarize the latest findings about Llama 3 for reasoning tasks.",
  "How does pgvector compare to other vector DBs for research use-cases?",
  "Show me papers about efficient embedding techniques in 2024.",
  "Explain how agentic RAG can autonomously expand its knowledge base.",
];

function getArxivSearchUrl(source: string) {
  const normalizedSource = source.replace(/^Processing:\s*/i, "").trim();
  const params = new URLSearchParams({
    query: normalizedSource,
    searchtype: "all",
    source: "header",
  });

  return `https://arxiv.org/search/?${params.toString()}`;
}

function LoadingIndicator({ step }: { step: number }) {
  const steps = [
    "Searching knowledge base",
    "Retrieving relevant documents",
    "Analyzing context",
    "Generating response with Gemini",
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

type HomeClientProps = {
  lastUpdatedLabel: string;
};

export default function HomeClient({ lastUpdatedLabel }: HomeClientProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState(0);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const [promptIndex, setPromptIndex] = useState(0);
  const [cycling, setCycling] = useState(true);

  const [sessionId, setSessionId] = useState(() =>
    Math.random().toString(36).substring(2, 15),
  );

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (!cycling) return;
    if (query || messages.length > 0 || loading) {
      return;
    }

    const iv = setInterval(() => {
      setPromptIndex((p) => (p + 1) % SAMPLE_PROMPTS.length);
    }, 3500);

    return () => clearInterval(iv);
  }, [query, messages.length, loading, cycling]);

  useEffect(() => {
    if (!loading) {
      setLoadingStep(0);
      return;
    }

    const steps = [
      "Searching knowledge base",
      "Retrieving relevant documents",
      "Analyzing context",
      "Generating response with Gemini",
    ];

    const interval = setInterval(() => {
      setLoadingStep((prev) => (prev + 1) % steps.length);
    }, 2000);

    return () => clearInterval(interval);
  }, [loading]);

  const userMessageCount = messages.filter((m) => m.role === "user").length;
  const followUpsRemaining =
    userMessageCount === 0 ? 3 : 3 - (userMessageCount - 1);
  const isLimitReached = userMessageCount > 0 && followUpsRemaining <= 0;

  const handleAsk = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!query || isLimitReached || loading) return;

    const userQuery = query;
    setQuery("");
    setLoading(true);

    const updatedMessages: Message[] = [
      ...messages,
      { role: "user", content: userQuery },
    ];
    setMessages(updatedMessages);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: userQuery,
          session_id: sessionId,
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
          // Keep the HTTP-based fallback.
        }
        throw new Error(errorMessage);
      }

      const data = await res.json();

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
    setSessionId(Math.random().toString(36).substring(2, 15));
  };

  const socialLinkClassName =
    "inline-flex h-9 w-9 items-center justify-center rounded-full border border-gray-200 bg-white text-gray-700 shadow-sm transition-colors hover:border-gray-300 hover:bg-gray-50 hover:text-gray-900";

  return (
    <main className="flex min-h-screen flex-col items-center p-4 md:p-10 bg-gray-50 text-gray-900">
      <div className="w-full max-w-6xl mx-auto flex flex-col gap-4 h-[90vh]">
        <div className="flex justify-between items-center bg-white p-4 rounded-xl shadow-sm border">
          <h1
            style={{
              fontFamily: '"Times New Roman", Times, serif',
              fontWeight: 400,
            }}
            className="text-3xl tracking-tight text-[#332f7b]"
          >
            Syntropy: Agentic Research Assistant
          </h1>
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

        <div className="flex-1 bg-white border rounded-xl p-4 shadow-sm overflow-y-auto flex flex-col gap-4">
          {messages.length === 0 ? (
            <div className="text-gray-600 text-center my-auto flex flex-col items-center justify-center h-full gap-3 px-6">
              <Image
                src="/syntropy-logo.png"
                alt="Syntropy logo"
                width={64}
                height={64}
                priority
                className="h-16 w-16 object-contain"
              />
              <p className="max-w-2xl text-sm text-gray-500">
                An agentic RAG research assistant that retrieves relevant
                knowledge from a vector database and — when needed — performs
                live research (ArXiv ingestion) to expand its knowledge base.
              </p>
              <p className="max-w-2xl text-sm text-gray-500">
                Powered by Llama-3 inference, Supabase pgvector for persistence,
                and Jina embeddings for fast semantic search — ask a research
                question to get started or try one of the sample prompts below.
              </p>
              <div className="mt-2 text-sm text-gray-400 italic">
                Try:{" "}
                <span key={promptIndex} className="fade-up">
                  {SAMPLE_PROMPTS[promptIndex]}
                </span>
              </div>
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
          <div ref={messagesEndRef} />
        </div>

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
          <form
            onSubmit={handleAsk}
            className="flex flex-col md:flex-row gap-2"
          >
            <div className="relative flex-1">
              <input
                ref={inputRef}
                type="text"
                className="w-full p-4 border rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder={
                  messages.length === 0 ? "" : "Ask a follow-up question..."
                }
                value={query}
                onChange={(e) => {
                  setCycling(false);
                  setQuery(e.target.value);
                }}
                onFocus={() => {
                  if (cycling && !query && messages.length === 0) {
                    setQuery(SAMPLE_PROMPTS[promptIndex]);
                    setCycling(false);
                  }
                }}
                disabled={loading}
              />

              {cycling && !query && messages.length === 0 && (
                <button
                  type="button"
                  onClick={() => {
                    setQuery(SAMPLE_PROMPTS[promptIndex]);
                    setCycling(false);
                    setTimeout(() => inputRef.current?.focus(), 0);
                  }}
                  className="absolute left-4 top-1/2 -translate-y-1/2 text-sm text-gray-400 italic fade-up text-left"
                >
                  {SAMPLE_PROMPTS[promptIndex]}
                </button>
              )}
            </div>
            <button
              type="submit"
              disabled={loading || isLimitReached || !query.trim()}
              className="md:w-auto w-full bg-blue-600 text-white px-8 py-4 rounded-xl font-semibold hover:bg-blue-700 disabled:opacity-50 transition-colors"
            >
              {loading ? "Searching..." : "Send"}
            </button>
          </form>
        )}

        <footer className="mt-1 flex flex-col items-center gap-3 border-t border-gray-200 pt-4 text-center text-sm text-gray-500 md:flex-row md:justify-between md:text-left">
          <div className="flex flex-col items-center gap-1 md:items-start">
            <div className="flex flex-wrap items-center justify-center gap-x-2 gap-y-1 md:justify-start">
              <span className="font-semibold tracking-wide text-[#332f7b]">
                Syntropy
              </span>
              <span className="text-xs uppercase tracking-[0.1em] text-gray-400">
                Last updated {lastUpdatedLabel} IST
              </span>
            </div>
            <span>Made by Harshal Sharma</span>
          </div>

          <div className="flex items-center gap-3">
            <a
              href="https://github.com/Harshalsharma05"
              target="_blank"
              rel="noreferrer"
              aria-label="GitHub profile"
              className={socialLinkClassName}
            >
              <svg
                viewBox="0 0 24 24"
                fill="currentColor"
                aria-hidden="true"
                className="h-4 w-4"
              >
                <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.48 0-.237-.009-.868-.014-1.703-2.782.604-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.907-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.832.091-.647.35-1.086.636-1.336-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.689-.103-.254-.446-1.279.098-2.664 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0 1 12 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.026 2.747-1.026.546 1.385.203 2.41.1 2.664.64.701 1.028 1.596 1.028 2.689 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .265.18.577.688.478A10.019 10.019 0 0 0 22 12.017C22 6.484 17.523 2 12 2Z" />
              </svg>
            </a>
            <a
              href="https://www.linkedin.com/in/harshal-sharma-98851b2ab/"
              target="_blank"
              rel="noreferrer"
              aria-label="LinkedIn profile"
              className={socialLinkClassName}
            >
              <svg
                viewBox="0 0 24 24"
                fill="currentColor"
                aria-hidden="true"
                className="h-4 w-4"
              >
                <path d="M20.447 20.452h-3.554v-5.569c0-1.327-.024-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.94v5.666H9.352V9h3.414v1.561h.049c.476-.9 1.637-1.852 3.368-1.852 3.601 0 4.266 2.368 4.266 5.448v6.295ZM5.337 7.433a2.064 2.064 0 1 1 0-4.128 2.064 2.064 0 0 1 0 4.128ZM7.115 20.452H3.558V9h3.557v11.452Z" />
              </svg>
            </a>
          </div>
        </footer>
      </div>
    </main>
  );
}
