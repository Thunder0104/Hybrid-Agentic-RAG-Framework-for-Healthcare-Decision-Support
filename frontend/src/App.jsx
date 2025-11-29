import React, { useState, useEffect, useRef } from "react";
import "./App.css";

// Your custom chat conditions
const CHAT_CONDITIONS = {
  maxLength: 600,
  cooldownMs: 2000, // user must wait 2s between messages
  forbiddenWords: [], // add restricted phrases if you want
};

const STORAGE_KEY = "hds_conversations_v1";
const ACTIVE_KEY = "hds_active_conversation_id_v1";

const initialSystemMessage = {
  id: "sys-1",
  role: "assistant",
  content:
    "Welcome to the Healthcare Decision Support Assistant. I can help you interpret information and explore options, but I do not replace a licensed healthcare professional or provide a diagnosis.",
};

function createNewConversation(session_id = null) {
  const now = Date.now();

  // If backend gives a session_id then use it
  const id = session_id
    ? session_id
    : (typeof crypto !== "undefined" && crypto.randomUUID
        ? crypto.randomUUID()
        : `conv-${now}`);

  return {
    id: id,
    session_id: id,
    title: "New chat",
    createdAt: now,
    updatedAt: now,
    messages: [initialSystemMessage],
  };
}


function getInitialChatState() {
  if (typeof window === "undefined") {
    const conv = createNewConversation();
    return {
      conversations: [conv],
      activeConversationId: conv.id,
    };
  }

  try {
    const storedConvs = localStorage.getItem(STORAGE_KEY);
    const storedActiveId = localStorage.getItem(ACTIVE_KEY);

    let conversations;
    if (storedConvs) {
      conversations = JSON.parse(storedConvs);
    }

    if (!Array.isArray(conversations) || conversations.length === 0) {
      const conv = createNewConversation();
      return {
        conversations: [conv],
        activeConversationId: conv.id,
      };
    }

    let activeConversationId = storedActiveId;
    const exists = conversations.some((c) => c.id === activeConversationId);
    if (!activeConversationId || !exists) {
      activeConversationId = conversations[0].id;
    }

    return { conversations, activeConversationId };
  } catch {
    const conv = createNewConversation();
    return {
      conversations: [conv],
      activeConversationId: conv.id,
    };
  }
}

function validateMessage(content, lastSentAt) {
  content = content.trim();

  if (!content) return "Message cannot be empty.";

  if (content.length > CHAT_CONDITIONS.maxLength) {
    return `Message is too long. Max ${CHAT_CONDITIONS.maxLength} characters allowed.`;
  }

  const lowered = content.toLowerCase();
  for (const bad of CHAT_CONDITIONS.forbiddenWords) {
    if (lowered && bad && lowered.includes(bad.toLowerCase())) {
      return `Your message contains a restricted phrase.`;
    }
  }

  if (lastSentAt) {
    const now = Date.now();
    const diff = now - lastSentAt;
    if (diff < CHAT_CONDITIONS.cooldownMs) {
      const waitSec = Math.ceil(
        (CHAT_CONDITIONS.cooldownMs - diff) / 1000
      );
      return `Please wait ${waitSec} more second(s) before sending another question.`;
    }
  }

  return null;
}

export default function App() {
  const [chatState, setChatState] = useState(getInitialChatState);
  const [input, setInput] = useState("");
  const [error, setError] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [lastSentAt, setLastSentAt] = useState(null);

  const bottomRef = useRef(null);

  const { conversations, activeConversationId } = chatState;

  const activeConversation =
    conversations.find((c) => c.id === activeConversationId) ||
    conversations[0];

  const activeMessages = activeConversation
    ? activeConversation.messages
    : [initialSystemMessage];

  // Persist chat whenever conversations / active ID change
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
      localStorage.setItem(ACTIVE_KEY, activeConversationId);
    } catch (e) {
      console.error("Failed to save chat state to localStorage", e);
    }
  }, [conversations, activeConversationId]);

  // Auto-scroll on new messages / thinking state
  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [activeMessages, isThinking]);

  // Reset timing state when switching conversations
  useEffect(() => {
    setLastSentAt(null);
    setError("");
    setIsSending(false);
    setIsThinking(false);
  }, [activeConversationId]);

  const handleSend = async () => {
    setError("");
    const trimmed = input.trim();

    const validationError = validateMessage(trimmed, lastSentAt);
    if (validationError) {
      setError(validationError);
      return;
    }

    if (!activeConversation) {
      // If somehow there's no active conversation, create one
      const newConv = createNewConversation();
      setChatState({
        conversations: [newConv],
        activeConversationId: newConv.id,
      });
      return;
    }

    const userMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: trimmed,
    };

    // History we send to backend (current messages + this message)
    const historyForBackend = [...activeMessages, userMessage];

    // Add user message to active conversation immediately
    setChatState((prev) => {
      const { conversations, activeConversationId } = prev;
      const updatedConversations = conversations.map((c) => {
        if (c.id !== activeConversationId) return c;
        const newMessages = [...c.messages, userMessage];

        const newTitle =
          c.title === "New chat" || !c.title
            ? userMessage.content.slice(0, 40)
            : c.title;

        return {
          ...c,
          title: newTitle,
          messages: newMessages,
          updatedAt: Date.now(),
        };
      });

      return { ...prev, conversations: updatedConversations };
    });

    setInput("");
    setIsSending(true);
    setIsThinking(true);
    setLastSentAt(Date.now());

    try {
      // CALL YOUR AGENTIC + RAG BACKEND HERE
      const response = await fetch("/api/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          user_query: trimmed,
          history: historyForBackend, // full conversation history
          session_id: activeConversation.session_id
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`);
      }

      const data = await response.json();
      console.log(data)
      const assistantText =
        data.answer || data.message || "No answer field found in response.";

      const assistantMessage = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: assistantText,
      };

      // Append assistant message to active conversation
      setChatState((prev) => {
        const { conversations, activeConversationId } = prev;
        const updatedConversations = conversations.map((c) => {
          if (c.id !== activeConversationId) return c;
          return {
            ...c,
            messages: [...c.messages, assistantMessage],
            updatedAt: Date.now(),
          };
        });

        return { ...prev, conversations: updatedConversations };
      });
    } catch (e) {
      console.error(e);
      setError("Something went wrong while contacting the assistant.");
    } finally {
      setIsSending(false);
      setIsThinking(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!isSending) {
        handleSend();
      }
    }
  };

  const handleNewChat = async () => {
    // 1. Call backend to get a new session ID
    const response = await fetch("api/session/start", {
      method: "POST",
    });
    const data = await response.json();
    const sessionId = data.session_id;

    // 2. Create conversation with this sessionId
    const newConv = createNewConversation(sessionId);

    // 3. Add to chat state
    setChatState((prev) => ({
      conversations: [newConv, ...prev.conversations],
      activeConversationId: newConv.id,
    }));

    setInput("");
    setError("");
    setIsSending(false);
    setIsThinking(false);
    setLastSentAt(null);
  };

  const handleSelectConversation = (id) => {
    if (id === activeConversationId) return;
    setChatState((prev) => ({
      ...prev,
      activeConversationId: id,
    }));
  };

  return (
    <div className="chat-root">
      <header className="chat-header">
        <div className="chat-header-title">
          🩺 Healthcare Decision Support Assistant
        </div>
        <div className="chat-header-subtitle">
          Ask about symptoms, reports, or treatment options. This is for
          informational decision support only and does not replace a clinician
          or provide medical diagnosis.
        </div>
      </header>

      <div className="chat-body">
        {/* Sidebar with conversations */}
        <aside className="chat-sidebar">
          <button
            type="button"
            className="chat-new-chat-button"
            onClick={handleNewChat}
            disabled={isSending || isThinking}
          >
            + New Chat
          </button>

          <div className="chat-conversation-list">
            {conversations.map((conv) => {
              const isActive = conv.id === activeConversationId;
              const title = conv.title || "New chat";
              return (
                <button
                  key={conv.id}
                  type="button"
                  className={`chat-conversation-item ${
                    isActive ? "chat-conversation-item-active" : ""
                  }`}
                  onClick={() => handleSelectConversation(conv.id)}
                >
                  <div className="chat-conversation-title">{title}</div>
                  {conv.updatedAt && (
                    <div className="chat-conversation-time">
                      {new Date(conv.updatedAt).toLocaleString()}
                    </div>
                  )}
                </button>
              );
            })}
          </div>
        </aside>

        {/* Main chat area */}
        <main className="chat-main">
          <div className="chat-messages">
            {activeMessages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} />
            ))}

            {/* Thinking / typing indicator while backend runs */}
            {isThinking && <TypingIndicator />}

            <div ref={bottomRef} />
          </div>
        </main>
      </div>

      <footer className="chat-footer">
        {error && <div className="chat-error">{error}</div>}
        <div className="chat-input-container">
          <textarea
            className="chat-input"
            placeholder='Describe your question or context (e.g., "Help me understand this lab result…")'
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
          />
          <button
            className="chat-send-button"
            onClick={handleSend}
            disabled={isSending}
          >
            {isSending ? "Sending..." : "Ask"}
          </button>
        </div>
        <div className="chat-footer-note">
          This tool supports clinical reasoning but is not a substitute for
          professional medical advice, diagnosis, or treatment.
        </div>
      </footer>
    </div>
  );
}

function MessageBubble({ message }) {
  const isUser = message.role === "user";
  const icon = isUser ? "🧑‍⚕️" : "🩺";
  const label = isUser ? "Clinician / User" : "Decision Support Assistant";

  return (
    <div
      className={`message-row ${
        isUser ? "message-row-user" : "message-row-assistant"
      }`}
    >
      <div className="message-avatar">
        <span>{icon}</span>
      </div>
      <div className="message-content-wrapper">
        <div className="message-label">{label}</div>
        <div
          className={`message-bubble ${
            isUser ? "message-bubble-user" : "message-bubble-assistant"
          }`}
        >
          {message.content}
        </div>
      </div>
    </div>
  );
}

// 💭 Typing / thinking bubble (three animated dots)
function TypingIndicator() {
  return (
    <div className="message-row message-row-assistant">
      <div className="message-avatar">
        <span>🩺</span>
      </div>
      <div className="message-content-wrapper">
        <div className="message-label">Decision Support Assistant</div>
        <div className="message-bubble message-bubble-assistant typing-bubble">
          <span className="typing-dot" />
          <span className="typing-dot" />
          <span className="typing-dot" />
        </div>
      </div>
    </div>
  );
}
