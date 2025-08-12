// src/chat.jsx
import React, { useEffect, useRef, useState } from "react";

// ---- Your endpoints ----
const API_URL   = "http://172.16.105.166:1306/v1/chat/completions";
const STOP_URL  = "http://172.16.105.166:1306/v1/stop";
const CLEAR_URL = "http://172.16.105.166:1306/v1/clear";
const MODEL = "qwen-0.6b";
const STREAM = true;            // set false if your server doesn't stream
const RESPONSE_TIMEOUT_MS = 180000; // 3 minutes, like your original script

export default function Chat() {
  // ===== Sidebar & chats =====
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [chats, setChats] = useState([
    { id: "1", title: "Welcome", active: true },
    { id: "2", title: "Design review notes", active: false },
    { id: "3", title: "API debugging", active: false },
  ]);

  // ===== Conversation =====
  const [messages, setMessages] = useState([
    { sender: "bot", text: "Hi! How can I help today?" },
  ]);
  const [input, setInput] = useState("");
  const [isGenerating, setIsGenerating] = useState(false); // <-- button toggle state
  const endRef = useRef(null);

  // streaming control
  const abortRef = useRef(null);
  const timeoutRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ========= Helpers for LLM comms =========
  async function stopLLM() {
    // Call your stop endpoint and cancel the stream
    try {
      await fetch(STOP_URL, { method: "POST" });
    } catch (e) {
      console.error("Error calling stop endpoint:", e);
    }
    abortRef.current?.abort();
  }

  async function clearCache() {
    try {
      const res = await fetch(CLEAR_URL, { method: "POST" });
      if (!res.ok) console.error(`Clear-cache failed: HTTP ${res.status}`);
    } catch (e) {
      console.error("Error calling clear endpoint:", e);
    }
  }

  async function callLLM(prompt, signal, onChunk) {
    const payload = {
      model: MODEL,
      messages: [{ role: "user", content: prompt }],
      stream: STREAM,
    };

    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    if (!STREAM) {
      const data = await res.json();
      const text = data?.choices?.[0]?.message?.content ?? "(no content)";
      onChunk(text);
      return;
    }

    // Streaming (SSE-like)
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";
      for (let part of parts) {
        if (!part.startsWith("data:")) continue;
        const data = part.replace(/^data:\s*/, "").trim();
        if (data === "[DONE]") return;
        try {
          const json = JSON.parse(data);
          const chunk = json?.choices?.[0]?.delta?.content;
          if (chunk) onChunk(chunk);
        } catch {
          // ignore malformed chunk
        }
      }
    }
  }

  // ========= Send / Stop (button toggle) =========
  async function onSendOrStop(e) {
    e?.preventDefault?.();

    // If already generating -> STOP
    if (isGenerating) {
      await stopLLM();
      return;
    }

    const text = input.trim();
    if (!text) return;

    // Push user message and clear input
    setMessages((m) => [...m, { sender: "user", text }]);
    setInput("");

    // Insert empty bot placeholder to stream into
    setMessages((m) => [...m, { sender: "bot", text: "" }]);

    // start streaming
    setIsGenerating(true); // <-- button shows "Stop"
    const controller = new AbortController();
    abortRef.current = controller;

    // timeout guard
    clearTimeout(timeoutRef.current);
    timeoutRef.current = setTimeout(async () => {
      if (abortRef.current) {
        alert("⚠️ Response timed out.");
        await stopLLM();
      }
    }, RESPONSE_TIMEOUT_MS);

    try {
      await callLLM(text, controller.signal, (chunk) => {
        setMessages((m) => {
          const copy = [...m];
          const last = copy[copy.length - 1];
          if (last && last.sender === "bot") last.text += chunk;
          return copy;
        });
      });
    } catch (err) {
      setMessages((m) => {
        const copy = [...m];
        const last = copy[copy.length - 1];
        if (last && last.sender === "bot") {
          if (err?.name === "AbortError") {
            last.text += "\n⚠️ Generation stopped.";
          } else {
            last.text += "\n⚠️ Chat service unavailable.";
          }
        }
        return copy;
      });
      console.error(err);
    } finally {
      clearTimeout(timeoutRef.current);
      abortRef.current = null;
      setIsGenerating(false); // <-- button back to "Send"
    }
  }

  // ========= UI handlers =========
  const onKeyTextArea = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSendOrStop();
    }
  };

  // For your original "buttontoggle" (keyboard on the button)
  const buttontoggle = (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onSendOrStop();
    }
  };

  const onNewChat = async () => {
    // visually switch active chat
    const id = String(Date.now());
    setChats((prev) =>
      prev.map((c) => ({ ...c, active: false })).concat({ id, title: "New chat", active: true })
    );
    // clear UI messages
    setMessages([{ sender: "bot", text: "Started a new chat 👋" }]);
    // call your CLEAR endpoint
    await clearCache();
  };

  const onSelectChat = (id) => {
    setChats((prev) => prev.map((c) => ({ ...c, active: c.id === id })));
    // demo: pretend to load a convo
    setMessages([
      { sender: "bot", text: `Loaded conversation ${id}` },
      { sender: "user", text: "Cool, thanks!" },
    ]);
  };

  const onDeleteChat = (id) => {
    setChats((prev) => {
      const next = prev.filter((c) => c.id !== id);
      if (!next.some((c) => c.active) && next.length) {
        next[0].active = true;
        setMessages([{ sender: "bot", text: "Switched to first chat." }]);
      } else if (next.length === 0) {
        setMessages([{ sender: "bot", text: "No chats left. Start a new one!" }]);
      }
      return next;
    });
  };

  const Sidebar = () => (
    <>
      {/* Desktop sidebar */}
      <aside className="hidden sm:flex sm:flex-col w-72 shrink-0 border-r bg-white">
        <div className="flex items-center justify-between px-4 h-14 border-b">
          <div className="font-semibold">Chats</div>
          <button
            onClick={onNewChat}
            className="px-2 py-1 text-sm rounded-md bg-blue-600 text-white hover:bg-blue-700"
          >
            New
          </button>
        </div>
        <nav className="flex-1 overflow-y-auto p-2 space-y-1">
          {chats.map((c) => (
            <div
              key={c.id}
              className={`group flex items-center justify-between rounded-md px-3 py-2 cursor-pointer ${
                c.active ? "bg-blue-50 text-blue-700" : "hover:bg-gray-100"
              }`}
            >
              <button
                onClick={() => onSelectChat(c.id)}
                className="truncate text-left flex-1"
                title={c.title}
              >
                {c.title}
              </button>
              <button
                onClick={() => onDeleteChat(c.id)}
                className="opacity-60 hover:opacity-100 ml-2"
                aria-label="Delete chat"
                title="Delete"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M3 6h18" />
                  <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                  <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
                  <path d="M10 11v6M14 11v6" />
                </svg>
              </button>
            </div>
          ))}
        </nav>
      </aside>

      {/* Mobile trigger */}
      <button
        onClick={() => setSidebarOpen(true)}
        className="sm:hidden fixed top-4 left-4 z-30 rounded-md border bg-white/90 px-3 py-2 shadow"
        aria-label="Open menu"
      >
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M3 6h18M3 12h18M3 18h18" />
        </svg>
      </button>

      {/* Mobile drawer */}
      <div className={`sm:hidden fixed inset-0 z-40 ${sidebarOpen ? "" : "pointer-events-none"}`}>
        <div
          className={`absolute inset-0 bg-black/40 transition-opacity ${sidebarOpen ? "opacity-100" : "opacity-0"}`}
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
        <aside
          className={`absolute left-0 top-0 h-full w-72 bg-white border-r shadow-xl transition-transform ${
            sidebarOpen ? "translate-x-0" : "-translate-x-full"
          }`}
          role="dialog"
          aria-modal="true"
        >
          <div className="flex items-center justify-between px-4 h-14 border-b">
            <div className="font-semibold">Chats</div>
            <button
              onClick={() => setSidebarOpen(false)}
              className="rounded-md p-2 hover:bg-gray-100"
              aria-label="Close menu"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M6 6l12 12M6 18L18 6" />
              </svg>
            </button>
          </div>
          <div className="p-3">
            <button
              onClick={async () => {
                await onNewChat();
                setSidebarOpen(false);
              }}
              className="w-full mb-3 px-3 py-2 text-sm rounded-md bg-blue-600 text-white hover:bg-blue-700"
            >
              New Chat
            </button>
            <div className="space-y-1">
              {chats.map((c) => (
                <div
                  key={c.id}
                  className={`group flex items-center justify-between rounded-md px-3 py-2 ${
                    c.active ? "bg-blue-50 text-blue-700" : "hover:bg-gray-100"
                  }`}
                >
                  <button
                    onClick={() => {
                      onSelectChat(c.id);
                      setSidebarOpen(false);
                    }}
                    className="truncate text-left flex-1"
                    title={c.title}
                  >
                    {c.title}
                  </button>
                  <button
                    onClick={() => onDeleteChat(c.id)}
                    className="opacity-60 hover:opacity-100 ml-2"
                    aria-label="Delete chat"
                    title="Delete"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M3 6h18" />
                      <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
                      <path d="M10 11v6M14 11v6" />
                    </svg>
                  </button>
                </div>
              ))}
            </div>
          </div>
        </aside>
      </div>
    </>
  );

  // ========= Render =========
  const btnLabel = isGenerating ? "Stop" : "Send";

  return (
    <div className="flex min-h-screen bg-gray-100">
      <Sidebar />

      {/* Main chat panel */}
      <main className="flex-1 flex items-start justify-center p-4">
        <div className="flex flex-col h-full w-full max-w-2xl mx-auto bg-white shadow-lg rounded-lg">
          {/* Header */}
          <div className="px-4 py-2 bg-blue-600 text-white font-semibold rounded-t-lg">
            Chat with AI
          </div>

          {/* Conversation */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50">
            {messages.length === 0 && (
              <div className="text-center text-gray-500">Say hello 👋</div>
            )}
            {messages.map((m, i) => (
              <div
                key={i}
                className={`max-w-[75%] px-4 py-2 rounded-lg shadow-sm ${
                  m.sender === "user"
                    ? "bg-blue-500 text-white ml-auto"
                    : "bg-white text-gray-800 mr-auto"
                }`}
              >
                {m.text}
              </div>
            ))}
            <div ref={endRef} />
          </div>

          {/* Input */}
          <form
            onSubmit={onSendOrStop}
            className="flex items-center p-4 bg-white border-t border-gray-200 rounded-b-lg"
          >
            <textarea
              rows={1}
              className="flex-1 resize-none border border-gray-300 rounded-full p-3 focus:outline-none focus:ring-2 focus:ring-blue-400"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyTextArea}
              placeholder={isGenerating ? "Model is responding…" : "Type your message…"}
              disabled={false}
            />
            <button
              type="submit"
              onKeyDown={buttontoggle}
              className={`ml-4 px-4 py-2 rounded-full text-white transition ${
                isGenerating
                  ? "bg-red-600 hover:bg-red-700"
                  : "bg-blue-600 hover:bg-blue-700"
              }`}
            >
              {btnLabel}
            </button>
          </form>
        </div>
      </main>
    </div>
  );
}
