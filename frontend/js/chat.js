const Chat = (() => {
  const STORAGE_KEY = "hyperscalees_chat_v1";

  const messagesEl = document.getElementById("chat-messages");
  const formEl = document.getElementById("chat-form");
  const inputEl = document.getElementById("chat-input");
  const sendBtn = document.getElementById("chat-send");
  const initialGreetingEl = document.getElementById("initial-greeting");
  const newChatBtn = document.getElementById("new-chat-btn");
  // Captured before any mutation, so "New Chat" can restore the exact
  // original markup regardless of how many messages came and went.
  const initialGreetingHTML = initialGreetingEl.outerHTML;

  // In-memory + persisted conversation: [{role, content, citations?, error?}]
  let messages = [];

  const md = window.markdownit({
    html: false,
    linkify: true,
    breaks: true,
    highlight(str, lang) {
      if (lang && window.hljs.getLanguage(lang)) {
        try {
          return `<pre class="hljs"><code>${window.hljs.highlight(str, { language: lang }).value}</code></pre>`;
        } catch (_) {
          /* fall through */
        }
      }
      return `<pre class="hljs"><code>${md.utils.escapeHtml(str)}</code></pre>`;
    },
  });

  function renderMath(el) {
    if (!window.renderMathInElement) return;
    window.renderMathInElement(el, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false },
      ],
      // Streamed/partial text can contain an unmatched "$" mid-token; render
      // it as visible error text instead of throwing and breaking the rest
      // of the message.
      throwOnError: false,
    });
  }

  function saveHistory() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
    } catch (_) {
      /* localStorage unavailable (private mode, quota) — history just won't persist */
    }
  }

  function addMessage(role, text) {
    const wrapper = document.createElement("div");
    wrapper.className = `chat-message ${role}`;
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    if (role === "user") {
      bubble.textContent = text;
    } else {
      bubble.innerHTML = md.render(text || "");
    }
    wrapper.appendChild(bubble);
    messagesEl.appendChild(wrapper);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return bubble;
  }

  function addTypingBubble() {
    const wrapper = document.createElement("div");
    wrapper.className = "chat-message assistant";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
    wrapper.appendChild(bubble);
    messagesEl.appendChild(wrapper);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return bubble;
  }

  function addProviderBadge(bubble, provider) {
    const badge = document.createElement("div");
    badge.className = "provider-badge";
    badge.textContent = provider === "groq" ? "via Groq" : "via Ollama (local)";
    bubble.appendChild(badge);
  }

  function addCitations(bubble, citations) {
    if (!citations || !citations.length) return;
    const box = document.createElement("div");
    box.className = "citations";
    for (const c of citations) {
      const chip = document.createElement("span");
      chip.className = "citation-chip";
      chip.textContent = c.path;
      chip.title = `Open ${c.path} in the file browser`;
      chip.addEventListener("click", () => RepoBrowser.openFile(c.path));
      box.appendChild(chip);
    }
    bubble.appendChild(box);
  }

  function renderStoredMessage(msg) {
    if (msg.role === "user") {
      addMessage("user", msg.content);
      return;
    }

    const wrapper = document.createElement("div");
    wrapper.className = "chat-message assistant";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    wrapper.appendChild(bubble);
    messagesEl.appendChild(wrapper);

    if (msg.provider) addProviderBadge(bubble, msg.provider);

    if (msg.error) {
      const banner = document.createElement("div");
      banner.className = "chat-error-banner";
      banner.textContent = msg.error;
      bubble.appendChild(banner);
      return;
    }

    const contentDiv = document.createElement("div");
    contentDiv.className = "msg-content";
    contentDiv.innerHTML = md.render(msg.content || "");
    bubble.appendChild(contentDiv);
    renderMath(contentDiv);
    addCitations(bubble, msg.citations);
  }

  function restoreHistory() {
    let stored;
    try {
      stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    } catch (_) {
      stored = [];
    }
    if (!Array.isArray(stored) || stored.length === 0) return;

    messages = stored;
    initialGreetingEl.remove();
    for (const msg of messages) renderStoredMessage(msg);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  async function streamNdjson(response, onEvent) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let newlineIndex;
      while ((newlineIndex = buffer.indexOf("\n")) !== -1) {
        const line = buffer.slice(0, newlineIndex).trim();
        buffer = buffer.slice(newlineIndex + 1);
        if (line) onEvent(JSON.parse(line));
      }
    }
    if (buffer.trim()) onEvent(JSON.parse(buffer.trim()));
  }

  async function sendMessage(message) {
    addMessage("user", message);
    const priorHistory = messages.map(({ role, content }) => ({ role, content }));
    messages.push({ role: "user", content: message });
    saveHistory();

    const assistantBubble = addTypingBubble();
    let answerText = "";
    let provider = null;
    let contentDiv = null;
    let errorMessage = null;
    const citations = [];

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, history: priorHistory }),
      });

      await streamNdjson(response, (event) => {
        if (event.type === "meta") {
          provider = event.provider;
          assistantBubble.innerHTML = "";
          addProviderBadge(assistantBubble, provider);
          contentDiv = document.createElement("div");
          contentDiv.className = "msg-content";
          assistantBubble.appendChild(contentDiv);
        } else if (event.type === "citation") {
          citations.push(event);
        } else if (event.type === "token") {
          answerText += event.text;
          contentDiv.innerHTML = md.render(answerText);
          messagesEl.scrollTop = messagesEl.scrollHeight;
        } else if (event.type === "error") {
          errorMessage = event.message;
          const banner = document.createElement("div");
          banner.className = "chat-error-banner";
          banner.textContent = errorMessage;
          assistantBubble.appendChild(banner);
        } else if (event.type === "done") {
          // Rendered once here, not per-token: mid-stream text can contain an
          // unmatched "$" (equation not yet fully arrived), which would
          // otherwise misrender/flicker on every intermediate token.
          if (contentDiv) renderMath(contentDiv);
          addCitations(assistantBubble, citations);
        }
      });
    } catch (err) {
      errorMessage = `Request failed: ${err}`;
      assistantBubble.innerHTML = `<div class="chat-error-banner">${errorMessage}</div>`;
    }

    messages.push({ role: "assistant", content: answerText, citations, error: errorMessage, provider });
    saveHistory();
  }

  function startNewChat() {
    messages = [];
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch (_) {
      /* localStorage unavailable — nothing to clear */
    }
    messagesEl.innerHTML = initialGreetingHTML;
  }

  function autoGrow() {
    inputEl.style.height = "auto";
    inputEl.style.height = `${Math.min(inputEl.scrollHeight, 160)}px`;
  }

  function init() {
    restoreHistory();

    newChatBtn.addEventListener("click", startNewChat);
    inputEl.addEventListener("input", autoGrow);

    formEl.addEventListener("submit", (e) => {
      e.preventDefault();
      const message = inputEl.value.trim();
      if (!message) return;
      inputEl.value = "";
      autoGrow();
      sendBtn.disabled = true;
      sendMessage(message).finally(() => {
        sendBtn.disabled = false;
        inputEl.focus();
      });
    });

    inputEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        formEl.requestSubmit();
      }
    });
  }

  return { init };
})();
