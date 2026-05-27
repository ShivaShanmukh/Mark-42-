"use client";

import { useState, useRef, useEffect } from "react";

const API_BASE = "/api";
const DEFAULT_REPO_URL = "https://github.com/ShivaShanmukh/Job-Agent";

interface Message {
  role: "user" | "assistant";
  content: string;
  citations?: string[];
  streaming?: boolean;
}

interface IndexedRepo {
  repo_name: string;
  github_url: string;
  chunk_count: number;
}

export default function Home() {
  const [repoUrl, setRepoUrl] = useState(DEFAULT_REPO_URL);
  const [indexStatus, setIndexStatus] = useState<"idle" | "loading" | "done" | "error">("idle");
  const [indexedRepo, setIndexedRepo] = useState<IndexedRepo | null>(null);
  const [indexMessage, setIndexMessage] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputQuery, setInputQuery] = useState("");
  const [isQuerying, setIsQuerying] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleIndex() {
    if (!repoUrl.trim()) return;
    setIndexStatus("loading");
    setIndexMessage("Cloning and indexing repository...");

    try {
      const res = await fetch(`${API_BASE}/index`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ github_url: repoUrl.trim() }),
      });
      const data = await res.json();

      if (!res.ok) throw new Error(data.detail || "Indexing failed");

      setIndexedRepo({
        repo_name: data.repo_name,
        github_url: repoUrl.trim(),
        chunk_count: data.chunk_count,
      });
      setIndexMessage(`Indexed ${data.chunk_count} chunks from ${data.repo_name}`);
      setIndexStatus("done");
      setMessages([]);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setIndexMessage(`Error: ${msg}`);
      setIndexStatus("error");
    }
  }

  async function handleQuery(e: React.FormEvent) {
    e.preventDefault();
    if (!inputQuery.trim() || !indexedRepo || isQuerying) return;

    const userMsg: Message = { role: "user", content: inputQuery.trim() };
    const assistantMsg: Message = { role: "assistant", content: "", streaming: true };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setInputQuery("");
    setIsQuerying(true);

    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: userMsg.content, repo_name: indexedRepo.repo_name }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Query failed");
      }

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let citations: string[] = [];
      // Buffer incomplete lines across network chunks
      let lineBuffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        lineBuffer += decoder.decode(value, { stream: true });
        const lines = lineBuffer.split("\n");
        // Keep the last (potentially incomplete) line in the buffer
        lineBuffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const payload = JSON.parse(line.slice(6));
            if (payload.type === "token") {
              setMessages((prev) => {
                const updated = [...prev];
                const last = updated[updated.length - 1];
                if (last.role === "assistant") {
                  updated[updated.length - 1] = { ...last, content: last.content + payload.content };
                }
                return updated;
              });
            } else if (payload.type === "citations") {
              citations = payload.files;
            } else if (payload.type === "error") {
              throw new Error(payload.content);
            }
          } catch (parseErr) {
            if (parseErr instanceof SyntaxError) continue; // skip genuinely malformed frames
            throw parseErr;
          }
        }
      }

      // Flush any remaining buffered line
      if (lineBuffer.startsWith("data: ")) {
        try {
          const payload = JSON.parse(lineBuffer.slice(6));
          if (payload.type === "citations") citations = payload.files;
        } catch {
          // ignore
        }
      }

      setMessages((prev) => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        if (last.role === "assistant") {
          updated[updated.length - 1] = { ...last, streaming: false, citations };
        }
        return updated;
      });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setMessages((prev) => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        if (last.role === "assistant") {
          updated[updated.length - 1] = { ...last, content: `Error: ${msg}`, streaming: false };
        }
        return updated;
      });
    } finally {
      setIsQuerying(false);
    }
  }

  return (
    <main className="min-h-screen bg-white text-gray-900 flex flex-col">
      {/* Header */}
      <header className="border-b border-gray-200 px-6 py-4 flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-emerald-500 flex items-center justify-center">
          <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
          </svg>
        </div>
        <div>
          <h1 className="font-semibold text-lg leading-tight">Codebase Intelligence</h1>
          <p className="text-xs text-gray-500">Ask questions about any GitHub repo in natural language</p>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <aside className="w-72 border-r border-gray-200 flex flex-col p-4 gap-4 bg-gray-50">
          <div>
            <label className="text-xs font-medium text-gray-600 uppercase tracking-wide mb-2 block">
              GitHub Repository
            </label>
            <input
              type="url"
              value={repoUrl}
              onChange={(e) => setRepoUrl(e.target.value)}
              placeholder="https://github.com/owner/repo"
              className="w-full text-sm border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-emerald-500 bg-white"
            />
          </div>

          <button
            onClick={handleIndex}
            disabled={indexStatus === "loading" || !repoUrl.trim()}
            className="w-full bg-emerald-500 hover:bg-emerald-600 disabled:bg-emerald-300 text-white font-medium py-2 px-4 rounded-lg transition-colors text-sm"
          >
            {indexStatus === "loading" ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
                </svg>
                Indexing...
              </span>
            ) : (
              "Index Repo"
            )}
          </button>

          {indexMessage && (
            <div
              className={`text-xs rounded-lg px-3 py-2 ${
                indexStatus === "done"
                  ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
                  : indexStatus === "error"
                  ? "bg-red-50 text-red-700 border border-red-200"
                  : "bg-gray-100 text-gray-600"
              }`}
            >
              {indexMessage}
            </div>
          )}

          {indexedRepo && (
            <div className="text-xs bg-white border border-gray-200 rounded-lg p-3 space-y-1">
              <p className="font-medium text-gray-800">{indexedRepo.repo_name}</p>
              <p className="text-gray-500">{indexedRepo.chunk_count.toLocaleString()} chunks indexed</p>
              <a href={indexedRepo.github_url} target="_blank" rel="noopener noreferrer" className="text-emerald-600 hover:underline truncate block">
                {indexedRepo.github_url}
              </a>
            </div>
          )}

          <div className="mt-auto text-xs text-gray-400 space-y-1">
            <p className="font-medium text-gray-500">Example queries</p>
            {[
              "How does the job agent handle retries?",
              "What APIs does this project integrate with?",
              "Explain the main entry point",
              "How is authentication handled?",
            ].map((q) => (
              <button
                key={q}
                onClick={() => setInputQuery(q)}
                className="block w-full text-left text-gray-500 hover:text-emerald-600 hover:bg-emerald-50 rounded px-2 py-1 transition-colors"
              >
                {q}
              </button>
            ))}
          </div>
        </aside>

        {/* Chat */}
        <div className="flex-1 flex flex-col">
          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center h-full text-center text-gray-400 gap-3">
                <svg className="w-12 h-12 text-gray-200" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
                <div>
                  <p className="font-medium text-gray-500">No conversation yet</p>
                  <p className="text-sm mt-1">Index a repo on the left, then ask a question below</p>
                </div>
              </div>
            )}

            {messages.map((msg, i) => (
              <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                <div
                  className={`max-w-2xl rounded-2xl px-4 py-3 ${
                    msg.role === "user"
                      ? "bg-emerald-500 text-white"
                      : "bg-gray-100 text-gray-900"
                  }`}
                >
                  <p className="text-sm whitespace-pre-wrap leading-relaxed">
                    {msg.content}
                    {msg.streaming && (
                      <span className="inline-block w-1.5 h-4 bg-emerald-500 ml-1 animate-pulse rounded-sm" />
                    )}
                  </p>

                  {msg.citations && msg.citations.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-gray-200">
                      <p className="text-xs font-medium text-gray-500 mb-1">Sources</p>
                      <div className="flex flex-wrap gap-1">
                        {msg.citations.map((file) => (
                          <span key={file} className="text-xs bg-white border border-gray-200 rounded px-2 py-0.5 text-gray-600 font-mono">
                            {file}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="border-t border-gray-200 p-4">
            <form onSubmit={handleQuery} className="flex gap-3">
              <input
                type="text"
                value={inputQuery}
                onChange={(e) => setInputQuery(e.target.value)}
                disabled={!indexedRepo || isQuerying}
                placeholder={indexedRepo ? `Ask about ${indexedRepo.repo_name}...` : "Index a repo first"}
                className="flex-1 border border-gray-300 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 disabled:bg-gray-50 disabled:text-gray-400"
              />
              <button
                type="submit"
                disabled={!indexedRepo || isQuerying || !inputQuery.trim()}
                className="bg-emerald-500 hover:bg-emerald-600 disabled:bg-emerald-200 text-white rounded-xl px-5 py-3 transition-colors"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                </svg>
              </button>
            </form>
          </div>
        </div>
      </div>
    </main>
  );
}
