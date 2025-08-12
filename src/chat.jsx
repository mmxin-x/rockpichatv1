// src/chat.jsx
import React, { useEffect, useRef, useState } from "react";

// ---- Your endpoints ----
const API_URL   = "http://172.20.10.4:1306/v1/chat/completions";
const STOP_URL  = "http://172.20.10.4:1306/v1/stop";
const CLEAR_URL = "http://172.20.10.4:1306/v1/clear";
const MODEL = "qwen-0.6b";
const RESPONSE_TIMEOUT_MS = 180000; // 3 minutes

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
    { sender: "bot", text: "Hi! **How can I help** today?\n\n- Ask me anything\n-" },
  ]);
  const [input, setInput] = useState("");
  const [isGenerating, setIsGenerating] = useState(false); // <-- button toggle
  const endRef = useRef(null);

  // Streaming controls / guards
  const abortRef = useRef(null);
  const timeoutRef = useRef(null);
  const botIndexRef = useRef(-1);   // index of the bot placeholder to append to
  const sessionRef = useRef(0);     // unique per request
  const lastEventRef = useRef("");  // de-dupe identical SSE frames

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ========= Markdown formatting (safe, no external deps) =========
  function escapeHtml(s) {
    return s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function formatTextSegment(text) {
    // Line-wise transforms: lists, headings, blockquotes
    const lines = text.split("\n");
    let out = [];
    let inUl = false;
    let inOl = false;

    const closeLists = () => {
      if (inUl) { out.push("</ul>"); inUl = false; }
      if (inOl) { out.push("</ol>"); inOl = false; }
    };

    for (let line of lines) {
      // Unordered list
      const mUl = line.match(/^\s*[-*]\s+(.+)/);
      if (mUl) {
        if (!inUl) { closeLists(); out.push("<ul>"); inUl = true; }
        out.push(`<li>${mUl[1]}</li>`);
        continue;
      }
      // Ordered list
      const mOl = line.match(/^\s*\d+\.\s+(.+)/);
      if (mOl) {
        if (!inOl) { closeLists(); out.push("<ol>"); inOl = true; }
        out.push(`<li>${mOl[1]}</li>`);
        continue;
      }

      // Not a list item
      if (inUl || inOl) closeLists();

      // Headings
      let m;
      if ((m = line.match(/^######\s+(.*)$/))) { out.push(`<h6>${m[1]}</h6>`); continue; }
      if ((m = line.match(/^#####\s+(.*)$/)))  { out.push(`<h5>${m[1]}</h5>`); continue; }
      if ((m = line.match(/^####\s+(.*)$/)))   { out.push(`<h4>${m[1]}</h4>`); continue; }
      if ((m = line.match(/^###\s+(.*)$/)))    { out.push(`<h3>${m[1]}</h3>`); continue; }
      if ((m = line.match(/^##\s+(.*)$/)))     { out.push(`<h2>${m[1]}</h2>`); continue; }
      if ((m = line.match(/^#\s+(.*)$/)))      { out.push(`<h1>${m[1]}</h1>`); continue; }

      // Blockquote
      const bq = line.match(/^>\s?(.*)$/);
      if (bq) { out.push(`<blockquote>${bq[1]}</blockquote>`); continue; }

      // Plain line
      out.push(line);
    }
    closeLists();

    let html = out.join("\n");

    // Inline transforms (on text segments only)
    // Links: [text](http(s)://url)
    html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, `<a href="$2" target="_blank" rel="noopener noreferrer" class="underline">$1</a>`);
    // Inline code: `code`
    html = html.replace(/`([^`]+)`/g, `<code class="px-1 py-0.5 rounded bg-gray-100 font-mono">$1</code>`);
    // Bold then italic
    html = html.replace(/\*\*(.+?)\*\*/g, `<strong>$1</strong>`);
    html = html.replace(/(^|[^\*])\*(?!\s)(.+?)(?<!\s)\*(?!\*)/g, `$1<em>$2</em>`);

    // Paragraphs: wrap chunks that aren't obvious blocks
    const blocks = html.split(/\n{2,}/).map(chunk => {
      const trimmed = chunk.trim();
      if (!trimmed) return "";
      if (/^<(ul|ol|h\d|pre|blockquote)/.test(trimmed)) return trimmed;
      return `<p>${trimmed.replace(/\n/g, "<br />")}</p>`;
    });

    return blocks.join("\n");
  }

  function formatMarkdown(src) {
    // 1) Escape HTML to avoid XSS
    const escaped = escapeHtml(src);

    // 2) Tokenize fenced code blocks ```lang\ncode\n```
    const segments = [];
    const re = /```([a-zA-Z0-9_-]+)?\n([\s\S]*?)```/g;
    let lastIndex = 0;
    let m;
    while ((m = re.exec(escaped))) {
      if (m.index > lastIndex) {
        segments.push({ type: "text", value: escaped.slice(lastIndex, m.index) });
      }
      segments.push({ type: "code", lang: m[1] || "", code: m[2] });
      lastIndex = re.lastIndex;
    }
    if (lastIndex < escaped.length) {
      segments.push({ type: "text", value: escaped.slice(lastIndex) });
    }

    // 3) Format text segments; 4) Reinsert code blocks
    const html = segments
      .map(seg => {
        if (seg.type === "code") {
          return `<pre class="overflow-auto rounded bg-gray-900 text-gray-100 p-3"><code class="language-${seg.lang}">${seg.code}</code></pre>`;
        }
        return formatTextSegment(seg.value);
      })
      .join("");

    return html;
  }

  // ========= LLM comms =========
  async function stopLLM() {
    try { await fetch(STOP_URL, { method: "POST" }); } catch (e) { console.error("stop error:", e); }
    abortRef.current?.abort();
  }

  async function clearCache() {
    try {
      const res = await fetch(CLEAR_URL, { method: "POST" });
      if (!res.ok) console.error(`Clear-cache failed: HTTP ${res.status}`);
    } catch (e) { console.error("clear error:", e); }
  }

  async function callLLM(prompt, signal, onChunk) {
    const payload = {
      model: MODEL,
      messages: [{ role: "user", content: prompt }],
      stream: true,
    };

    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // Split on blank line (CRLF/LF safe)
      const parts = buffer.split(/\r?\n\r?\n/);
      buffer = parts.pop() || "";

      for (let part of parts) {
        if (!part.startsWith("data:")) continue;
        const data = part.replace(/^data:\s*/, "").trim();
        if (!data) continue;
        if (data === "[DONE]") return;

        // De-dupe identical raw frames (some proxies resend)
        if (data === lastEventRef.current) continue;
        lastEventRef.current = data;

        try {
          const json = JSON.parse(data);
          const delta = json?.choices?.[0]?.delta ?? {};
          if (typeof delta.content !== "string" || !delta.content) continue;
          onChunk(delta.content);
        } catch {
          // ignore malformed
        }
      }
    }
  }

  // ========= Send / Stop (button toggle) =========
  async function onSendOrStop(e) {
    e?.preventDefault?.();

    if (isGenerating) {
      await stopLLM();
      return;
    }

    const text = input.trim();
    if (!text) return;

    // If somehow a stream is alive, stop it first
    if (abortRef.current) await stopLLM();

    // new session + reset de-dupe
    sessionRef.current += 1;
    const mySession = sessionRef.current;
    lastEventRef.current = "";

    // push user + ONE bot placeholder in one update
    setMessages(prev => {
      const next = [...prev, { sender: "user", text }, { sender: "bot", text: "" }];
      botIndexRef.current = next.length - 1;
      return next;
    });
    setInput("");
    setIsGenerating(true);

    const controller = new AbortController();
    abortRef.current = controller;

    clearTimeout(timeoutRef.current);
    timeoutRef.current = setTimeout(async () => {
      if (mySession !== sessionRef.current) return;
      alert("⚠️ Response timed out.");
      await stopLLM();
    }, RESPONSE_TIMEOUT_MS);

    try {
      await callLLM(text, controller.signal, (chunk) => {
        if (mySession !== sessionRef.current) return; // stale stream
        setMessages(prev => {
          const copy = [...prev];
          const i = botIndexRef.current;
          if (copy[i]) copy[i] = { ...copy[i], text: copy[i].text + chunk };
          return copy;
        });
      });
    } catch (err) {
      setMessages(prev => {
        const copy = [...prev];
        const i = botIndexRef.current;
        if (copy[i]) {
          copy[i] = {
            ...copy[i],
            text:
              copy[i].text +
              (err?.name === "AbortError" ? "\n⚠️ Generation stopped." : "\n⚠️ Chat service unavailable."),
          };
        }
        return copy;
      });
      console.error(err);
    } finally {
      clearTimeout(timeoutRef.current);
      abortRef.current = null;
      setIsGenerating(false);
    }
  }

  // ========= UI handlers =========
  const onKeyTextArea = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSendOrStop();
    }
  };

  const buttontoggle = (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onSendOrStop();
    }
  };

  const onNewChat = async () => {
    const id = String(Date.now());
    setChats(prev => prev.map(c => ({ ...c, active: false })).concat({ id, title: "New chat", active: true }));
    setMessages([{ sender: "bot", text: "Started a new chat 👋" }]);
    await clearCache();
  };

  const onSelectChat = (id) => {
    setChats(prev => prev.map(c => ({ ...c, active: c.id === id })));
    setMessages([
      { sender: "bot", text: `Loaded conversation ${id}` },
      { sender: "user", text: "Cool, thanks!" },
    ]);
  };

  const onDeleteChat = (id) => {
    setChats(prev => {
      const next = prev.filter(c => c.id !== id);
      if (!next.some(c => c.active) && next.length) {
        next[0].active = true;
        setMessages([{ sender: "bot", text: "Switched to first chat." }]);
      } else if (next.length === 0) {
        setMessages([{ sender: "bot", text: "No chats left. Start a new one!" }]);
      }
      return next;
    });
  };

  // ===== Sidebar =====
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
          className={`absolute left-0 top-0 h/full w-72 bg-white border-r shadow-xl transition-transform ${
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
              className="w/full mb-3 px-3 py-2 text-sm rounded-md bg-blue-600 text-white hover:bg-blue-700"
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
                className={`max-w-[75%] px-4 py-2 rounded-lg shadow-sm prose prose-sm max-w-none ${
                  m.sender === "user"
                    ? "bg-blue-500 text-white ml-auto"
                    : "bg-white text-gray-800 mr-auto"
                }`}
              >
                {/* Render Markdown */}
                <div
                  dangerouslySetInnerHTML={{ __html: formatMarkdown(m.text) }}
                />
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
