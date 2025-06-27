// chat.js
;(function(){
  // LLM CONFIG 
  const CHAT_URL = 'http://192.168.1.157:1306/v1/chat/completions';

  // Session metadata 
  const sessionId = crypto.randomUUID?.() ?? Math.random().toString(36).slice(2);
  const sessionStart = new Date().toISOString();

  // window allows this to be a global variable
  window.chatHistory = [];    // Array<{ sender: 'user'|'bot', text: string }>
  window.processSteps = [];   // Array<string>


  // Helper getters
  window.getChatHistory = () => [...window.chatHistory];
  window.getProcessSteps = () => [...window.processSteps];
  window.getSessionInfo = () => ({ sessionId, sessionStart });

  // Create DOM
 //  A DOM API helps to modify the page dynamically by allowing it the change the elements in the nodes
  const root = document.getElementById('chat-root');
  const container = document.createElement('div');
  container.className = 'flex flex-col h-full max-w-md w-full bg-white shadow-lg rounded-lg';

  // Header
  const header = document.createElement('div');
  header.className = 'px-4 py-2 bg-blue-600 text-white font-semibold rounded-t-lg';
  header.textContent = 'Chat with AI';
  container.appendChild(header);

  // Process box
 //   const processBox = document.createElement('div');
 //   processBox.className = 'flex-1 overflow-y-auto p-4 space-y-2 bg-gray-100 border-b border-gray-200';
//   container.appendChild(processBox);

  // Conversation box
  const convoBox = document.createElement('div');
  convoBox.className = 'flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50';
  container.appendChild(convoBox);

  // Input area
  const inputArea = document.createElement('div');
  //You can change this to modify the ui 
  inputArea.className = 'flex items-center p-4 bg-white border-t border-gray-200 rounded-b-lg';
  const textarea = document.createElement('textarea');
  textarea.rows = 1;
  textarea.placeholder = 'Type your message…';
  //You can change this to modify the ui 
  textarea.className = 'flex-1 resize-none border border-gray-300 rounded-full p-3 focus:outline-none focus:ring-2 focus:ring-blue-400';
  const sendBtn = document.createElement('button');
  sendBtn.textContent = 'Send';
  //You can change this to modify the ui 
  sendBtn.className = 'ml-4 px-4 py-2 bg-blue-600 text-white rounded-full disabled:bg-gray-300 hover:bg-blue-700 transition';
  inputArea.append(textarea, sendBtn);
  container.appendChild(inputArea);

  // Attach to root
  root.appendChild(container);

  // Helper functions to substitue the react inbuilds
  function scrollToBottom() {
    convoBox.scrollTop = convoBox.scrollHeight;
    processBox.scrollTop = processBox.scrollHeight;
  }

//   function renderProcess() {
//     processBox.innerHTML = '<div class="text-gray-700 font-medium mb-2">Process</div>';
//     if (!window.processSteps.length) {
//       const el = document.createElement('div');
//       el.className = 'text-gray-500 text-sm';
//       el.textContent = 'No steps yet.';
//       processBox.appendChild(el);
//     } else {
//       window.processSteps.forEach(step => {
//         const el = document.createElement('div');
//         el.className = 'text-gray-800 bg-white px-3 py-2 rounded-lg shadow-sm';
//         el.textContent = step;
//         processBox.appendChild(el);
//       });
//     }
//   }


//1): convoBox.innerHTML = '' wipes out all existing child nodes.

//2): If chatHistory is empty, it shows a default hello prompt.

//3): Otherwise it loops over each { sender, text } entry in chatHistory, creates a styled bubble <div>, sets its classes based on who sent it (you vs. the bot), fills in the text, and appends it.

//4): Calls scrollToBottom() to keep the latest messages visible.

  function renderMessages() {
    convoBox.innerHTML = '';
    if (!window.chatHistory.length) {
      const el = document.createElement('div');
      el.className = 'text-center text-gray-500';
      el.textContent = 'Say hello 👋';
      convoBox.appendChild(el);
    } else {
      window.chatHistory.forEach(m => {
        const msgEl = document.createElement('div');
        msgEl.className = [
          'max-w-[75%] px-4 py-2 rounded-lg shadow-sm',
          m.sender === 'user'
            ? 'bg-blue-500 text-white ml-auto'
            : 'bg-white text-gray-800 mr-auto'
        ].join(' ');
        msgEl.textContent = m.text;
        convoBox.appendChild(msgEl);
      });
    }
    scrollToBottom();
  }

  // ———— Send logic ————
  async function sendMessage() {
    const content = textarea.value.trim();
    if (!content) return;

    // 1) Add user message
    window.chatHistory.push({ sender: 'user', text: content });
    renderMessages();
    textarea.value = '';
    sendBtn.disabled = true;

    try {
      // 2) Fetch from LLM
      const resp = await fetch(CHAT_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'qwen-0.6b', 
          messages: [{ role: 'user', content }],
        }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      const full = data.choices?.[0]?.message?.content || '';

      // 3) Extract <think>
      const thinks = [...full.matchAll(/<think>([\s\S]*?)<\/think>/gi)].map(m => m[1].trim());
      window.processSteps.push(...thinks);

      // 4) Add visible reply
      const visible = full.replace(/<think>[\s\S]*?<\/think>/gi, '').trim();
      window.chatHistory.push({ sender: 'bot', text: visible });

      renderProcess();
      renderMessages();

    } catch (err) {
      console.error('chat error', err);
      window.chatHistory.push({ sender: 'bot', text: '⚠️ Chat service unavailable.' });
      renderMessages();
    } finally {
      sendBtn.disabled = false;
    }
  }

  // Events
  sendBtn.addEventListener('click', sendMessage);
  textarea.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // Initial render
  // renderProcess();
  renderMessages();

})();
