import { useState, useRef, useEffect } from 'react';

// Make sure this points at your Flask endpoint:
const API_URL =
  import.meta.env.VITE_API_URL ||
  'http://192.168.1.157:1306/v1/chat/completions';

export default function Chat() {
  const [messages, setMessages] = useState([]);       // user/bot conversation
  const [thinkSteps, setThinkSteps] = useState([]);   // extracted <think> content
  const [input, setInput] = useState('');
  const [isSending, setIsSending] = useState(false);
  const endRef = useRef(null);

  const sendMessage = async () => {
    const content = input.trim();
    if (!content) return;

    // 1) show user message
    setMessages(m => [...m, { sender: 'user', text: content }]);
    setInput('');
    setIsSending(true);

    try {
      // 2) call chat endpoint
      const res = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'qwen-0.6b',
          messages: [{ role: 'user', content }],
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const fullReply = data.choices?.[0]?.message?.content || '';

      // 3) extract all <think>...</think> sections
      const thinkMatches = [...fullReply.matchAll(/<think>([\s\S]*?)<\/think>/gi)];
      const thinks = thinkMatches.map(m => m[1].trim());

      // 4) strip out the <think> tags from the visible reply
      const visibleReply = fullReply.replace(/<think>[\s\S]*?<\/think>/gi, '').trim();

      // 5) append to state
      if (thinks.length) {
        setThinkSteps(prev => [...prev, ...thinks]);
      }
      setMessages(m => [...m, { sender: 'bot', text: visibleReply }]);

    } catch (err) {
      console.error('chat error', err);
      setMessages(m => [
        ...m,
        { sender: 'bot', text: '⚠️ Chat service unavailable.' },
      ]);
    } finally {
      setIsSending(false);
    }
  };

  const onKey = e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isSending) sendMessage();
    }
  };

  // auto-scroll on new messages
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, thinkSteps]);

  return (
    <div className="flex flex-col h-full max-w-md mx-auto bg-white shadow-lg rounded-lg">

      {/* Header */}
      <div className="px-4 py-2 bg-blue-600 text-white font-semibold rounded-t-lg">
        Chat with AI
      </div>

      {/* Process box */}
      <div className="flex-1 overflow-y-auto p-4 space-y-2 bg-gray-100 border-b border-gray-200">
        <div className="text-gray-700 font-medium mb-2">Process</div>
        {thinkSteps.length === 0 && (
          <div className="text-gray-500 text-sm">No processing steps yet.</div>
        )}
        {thinkSteps.map((step, i) => (
          <div
            key={i}
            className="text-gray-800 bg-white px-3 py-2 rounded-lg shadow-sm"
          >
            {step}
          </div>
        ))}
      </div>

      {/* Conversation box */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50">
        {messages.length === 0 && (
          <div className="text-center text-gray-500">Say hello 👋</div>
        )}
        {messages.map((m, i) => (
          <div
            key={i}
            className={`max-w-[75%] px-4 py-2 rounded-lg shadow-sm ${
              m.sender === 'user'
                ? 'bg-blue-500 text-white ml-auto'
                : 'bg-white text-gray-800 mr-auto'
            }`}
          >
            {m.text}
          </div>
        ))}
        <div ref={endRef} />
      </div>

      {/* Input area */}
      <div className="flex items-center p-4 bg-white border-t border-gray-200 rounded-b-lg">
        <textarea
          rows={1}
          className="flex-1 resize-none border border-gray-300 rounded-full p-3 focus:outline-none focus:ring-2 focus:ring-blue-400"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={onKey}
          placeholder="Type your message…"
          disabled={isSending}
        />
        <button
          onClick={sendMessage}
          disabled={!input.trim() || isSending}
          className="ml-4 px-4 py-2 bg-blue-600 text-white rounded-full disabled:bg-gray-300 hover:bg-blue-700 transition"
        >
          {isSending ? 'Sending…' : 'Send'}
        </button>
      </div>
    </div>
  );
}
