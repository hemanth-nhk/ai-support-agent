/**
 * AI Support Agent - Full Dashboard Controller
 * Matches Store Assistant design: Left sidebar, Center chat, and Right settings panel.
 */

(function () {
  'use strict';

  // --- State ---
  let currentSessionId = '';
  let isLoading = false;

  // --- DOM Elements ---
  const chatScrollArea = document.querySelector('.chat-scroll-area');
  const emptyState = document.getElementById('emptyState');
  const messagesList = document.getElementById('messagesList');
  const chatForm = document.getElementById('chatForm');
  const messageInput = document.getElementById('messageInput');
  const sendBtn = document.getElementById('sendBtn');
  const newChatBtn = document.getElementById('newChatBtn');
  const sessionUuidText = document.getElementById('sessionUuidText');

  // Settings Panel Elements
  const sidebarRight = document.getElementById('sidebarRight');
  const toggleSettingsBtn = document.getElementById('toggleSettingsBtn');
  const apiKeyInput = document.getElementById('apiKeyInput');
  const saveApiKeyBtn = document.getElementById('saveApiKeyBtn');
  const toggleKeyVisibilityBtn = document.getElementById('toggleKeyVisibilityBtn');
  const apiKeyStatusBox = document.getElementById('apiKeyStatusBox');
  const apiKeyStatusText = document.getElementById('apiKeyStatusText');

  // Theme Elements
  const themeToggleQuickBtn = document.getElementById('themeToggleQuickBtn');
  const themeSegments = document.querySelectorAll('.theme-segment');

  // --- Helpers ---
  function formatTime(date) {
    const d = date || new Date();
    return d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  function scrollToBottom() {
    if (chatScrollArea) {
      chatScrollArea.scrollTop = chatScrollArea.scrollHeight;
    }
  }

  // --- Session Management ---
  function initSession() {
    currentSessionId = crypto.randomUUID ? crypto.randomUUID() : ('sess-' + Date.now());
    if (sessionUuidText) {
      sessionUuidText.textContent = `${currentSessionId.slice(0, 8)}...`;
      sessionUuidText.title = `Full Session ID: ${currentSessionId}`;
    }
  }

  function startNewChat() {
    initSession();
    messagesList.innerHTML = '';
    messagesList.style.display = 'none';
    if (emptyState) emptyState.style.display = 'block';
    messageInput.value = '';
    messageInput.focus();
  }

  // --- Theme Management ---
  function applyTheme(theme) {
    let resolvedTheme = theme;
    if (theme === 'system') {
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      resolvedTheme = prefersDark ? 'dark' : 'light';
    }
    document.documentElement.setAttribute('data-theme', resolvedTheme);
    localStorage.setItem('support_theme', theme);

    themeSegments.forEach((btn) => {
      if (btn.getAttribute('data-theme-val') === theme) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });
  }

  function initTheme() {
    const savedTheme = localStorage.getItem('support_theme') || 'dark';
    applyTheme(savedTheme);
  }

  function toggleQuickTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
    const nextTheme = currentTheme === 'dark' ? 'light' : 'dark';
    applyTheme(nextTheme);
  }

  // --- API Key Management ---
  function getApiKey() {
    return localStorage.getItem('support_api_key') || '';
  }

  function updateApiKeyStatusDisplay() {
    const key = getApiKey();
    if (key && key.trim().length > 0) {
      apiKeyStatusText.textContent = 'API key saved';
      apiKeyStatusBox.classList.remove('not-saved');
    } else {
      apiKeyStatusText.textContent = 'No key configured';
      apiKeyStatusBox.classList.add('not-saved');
    }
  }

  function initApiKey() {
    const key = getApiKey();
    if (key) {
      apiKeyInput.value = key;
    }
    updateApiKeyStatusDisplay();
  }

  function saveApiKey() {
    const val = apiKeyInput.value.trim();
    if (val) {
      localStorage.setItem('support_api_key', val);
    } else {
      localStorage.removeItem('support_api_key');
    }
    updateApiKeyStatusDisplay();
  }

  function toggleKeyVisibility() {
    const isPassword = apiKeyInput.type === 'password';
    apiKeyInput.type = isPassword ? 'text' : 'password';
  }

  // --- Message UI Rendering ---
  function ensureMessagesListVisible() {
    if (emptyState) emptyState.style.display = 'none';
    messagesList.style.display = 'flex';
  }

  function appendUserMessage(text) {
    ensureMessagesListVisible();

    const turn = document.createElement('div');
    turn.className = 'chat-turn user';

    // Avatar
    const avatar = document.createElement('div');
    avatar.className = 'turn-avatar user-circle';
    avatar.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
        <circle cx="12" cy="7" r="4"></circle>
      </svg>
    `;

    // Content
    const content = document.createElement('div');
    content.className = 'turn-content';

    const bubble = document.createElement('div');
    bubble.className = 'turn-bubble';
    bubble.textContent = text;
    content.appendChild(bubble);

    const time = document.createElement('div');
    time.className = 'turn-time';
    time.textContent = formatTime();
    content.appendChild(time);

    turn.appendChild(avatar);
    turn.appendChild(content);
    messagesList.appendChild(turn);
    scrollToBottom();
  }

  function appendTypingIndicator() {
    ensureMessagesListVisible();

    const turn = document.createElement('div');
    turn.className = 'chat-turn assistant';
    turn.id = 'activeTypingIndicator';

    const avatar = document.createElement('div');
    avatar.className = 'turn-avatar bot';
    avatar.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="3" y="11" width="18" height="10" rx="2"></rect>
        <circle cx="12" cy="5" r="2"></circle>
        <path d="M12 7v4"></path>
        <line x1="8" y1="16" x2="8.01" y2="16"></line>
        <line x1="16" y1="16" x2="16.01" y2="16"></line>
      </svg>
    `;

    const content = document.createElement('div');
    content.className = 'turn-content';

    const bubble = document.createElement('div');
    bubble.className = 'typing-bubble';
    bubble.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';

    content.appendChild(bubble);
    turn.appendChild(avatar);
    turn.appendChild(content);

    messagesList.appendChild(turn);
    scrollToBottom();
  }

  function removeTypingIndicator() {
    const indicator = document.getElementById('activeTypingIndicator');
    if (indicator) indicator.remove();
  }

  function getStatusPillClass(status) {
    const s = (status || '').toLowerCase();
    if (s === 'ok') return 'pill-status-ok';
    if (s === 'clarification' || s === 'needs_clarification') return 'pill-status-warn';
    return 'pill-status-error';
  }

  function appendAssistantMessage(data) {
    ensureMessagesListVisible();

    const turn = document.createElement('div');
    turn.className = 'chat-turn assistant';

    // Bot Avatar
    const avatar = document.createElement('div');
    avatar.className = 'turn-avatar bot';
    avatar.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="3" y="11" width="18" height="10" rx="2"></rect>
        <circle cx="12" cy="5" r="2"></circle>
        <path d="M12 7v4"></path>
        <line x1="8" y1="16" x2="8.01" y2="16"></line>
        <line x1="16" y1="16" x2="16.01" y2="16"></line>
      </svg>
    `;

    const content = document.createElement('div');
    content.className = 'turn-content';

    // Message Bubble
    const bubble = document.createElement('div');
    bubble.className = 'turn-bubble';
    bubble.textContent = data.response;
    content.appendChild(bubble);

    // Time
    const time = document.createElement('div');
    time.className = 'turn-time';
    time.textContent = formatTime();
    content.appendChild(time);

    // Telemetry Badges Row
    const telemetry = document.createElement('div');
    telemetry.className = 'telemetry-row';

    // Intent pill
    const intentPill = document.createElement('span');
    intentPill.className = 'telemetry-pill pill-intent';
    intentPill.textContent = `intent: ${data.intent || 'UNKNOWN'}`;
    telemetry.appendChild(intentPill);

    // Tool pill
    const toolPill = document.createElement('span');
    toolPill.className = 'telemetry-pill pill-tool';
    toolPill.textContent = `tool: ${data.tool_used || 'none'}`;
    telemetry.appendChild(toolPill);

    // Status pill
    const statusClass = getStatusPillClass(data.status);
    const statusPill = document.createElement('span');
    statusPill.className = `telemetry-pill ${statusClass}`;
    statusPill.textContent = `status: ${data.status || 'unknown'}`;
    telemetry.appendChild(statusPill);

    // Retrievals pill
    const count = typeof data.retrieval_count === 'number' ? data.retrieval_count : 0;
    const retPill = document.createElement('span');
    retPill.className = 'telemetry-pill pill-retrievals';
    retPill.textContent = `retrievals: ${count}`;
    telemetry.appendChild(retPill);

    content.appendChild(telemetry);

    turn.appendChild(avatar);
    turn.appendChild(content);

    messagesList.appendChild(turn);
    scrollToBottom();
  }

  function appendAuthErrorAlert() {
    ensureMessagesListVisible();

    const turn = document.createElement('div');
    turn.className = 'chat-turn assistant';

    const avatar = document.createElement('div');
    avatar.className = 'turn-avatar bot';
    avatar.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="3" y="11" width="18" height="10" rx="2"></rect>
        <circle cx="12" cy="5" r="2"></circle>
        <path d="M12 7v4"></path>
      </svg>
    `;

    const content = document.createElement('div');
    content.className = 'turn-content';

    const bubble = document.createElement('div');
    bubble.className = 'turn-bubble';
    bubble.innerHTML = `<strong>Authentication Required (401)</strong><br />This query requires a support API key. Please configure your key in the Settings panel on the right.`;
    content.appendChild(bubble);

    const telemetry = document.createElement('div');
    telemetry.className = 'telemetry-row';
    telemetry.innerHTML = `
      <span class="telemetry-pill pill-intent">intent: DATA_QUERY</span>
      <span class="telemetry-pill pill-status-error">status: 401 Unauthorized</span>
    `;
    content.appendChild(telemetry);

    turn.appendChild(avatar);
    turn.appendChild(content);
    messagesList.appendChild(turn);
    scrollToBottom();
  }

  function appendGeneralError(errorMessage) {
    ensureMessagesListVisible();

    const turn = document.createElement('div');
    turn.className = 'chat-turn assistant';

    const avatar = document.createElement('div');
    avatar.className = 'turn-avatar bot';
    avatar.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="3" y="11" width="18" height="10" rx="2"></rect>
      </svg>
    `;

    const content = document.createElement('div');
    content.className = 'turn-content';

    const bubble = document.createElement('div');
    bubble.className = 'turn-bubble';
    bubble.textContent = `Error: ${errorMessage}`;
    content.appendChild(bubble);

    const telemetry = document.createElement('div');
    telemetry.className = 'telemetry-row';
    telemetry.innerHTML = `<span class="telemetry-pill pill-status-error">status: failed</span>`;
    content.appendChild(telemetry);

    turn.appendChild(avatar);
    turn.appendChild(content);
    messagesList.appendChild(turn);
    scrollToBottom();
  }

  // --- Send Message ---
  async function sendMessage(text) {
    const message = text.trim();
    if (!message || isLoading) return;

    appendUserMessage(message);
    messageInput.value = '';
    messageInput.style.height = 'auto';
    isLoading = true;
    sendBtn.disabled = true;
    messageInput.disabled = true;
    appendTypingIndicator();

    try {
      const headers = {
        'Content-Type': 'application/json',
      };
      const apiKey = getApiKey();
      if (apiKey && apiKey.trim().length > 0) {
        headers['X-API-Key'] = apiKey.trim();
      }

      const payload = {
        message: message,
        session_id: currentSessionId,
      };

      const res = await fetch('/chat', {
        method: 'POST',
        headers: headers,
        body: JSON.stringify(payload),
      });

      removeTypingIndicator();

      if (res.status === 401) {
        appendAuthErrorAlert();
        return;
      }

      if (!res.ok) {
        let errorDetail = `HTTP ${res.status}`;
        try {
          const errData = await res.json();
          if (errData && errData.detail) {
            errorDetail = typeof errData.detail === 'string' ? errData.detail : JSON.stringify(errData.detail);
          }
        } catch (_) {}
        appendGeneralError(errorDetail);
        return;
      }

      const data = await res.json();
      appendAssistantMessage(data);
    } catch (err) {
      removeTypingIndicator();
      appendGeneralError(err.message || 'Network request failed. Is the server running?');
    } finally {
      isLoading = false;
      sendBtn.disabled = false;
      messageInput.disabled = false;
      messageInput.focus();
    }
  }

  // --- Event Listeners ---
  function initEventListeners() {
    // Form submit
    chatForm.addEventListener('submit', function (e) {
      e.preventDefault();
      sendMessage(messageInput.value);
    });

    // Enter to send in textarea
    messageInput.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage(messageInput.value);
      }
    });

    // New Chat
    newChatBtn.addEventListener('click', startNewChat);

    // Left Navigation items
    document.querySelectorAll('.nav-item').forEach((item) => {
      item.addEventListener('click', () => {
        document.querySelectorAll('.nav-item').forEach((n) => n.classList.remove('active'));
        item.classList.add('active');
        const prompt = item.getAttribute('data-prompt');
        if (prompt) {
          sendMessage(prompt);
        }
      });
    });

    // Recent Chats items
    document.querySelectorAll('.recent-chat-item').forEach((item) => {
      item.addEventListener('click', () => {
        const prompt = item.getAttribute('data-prompt');
        if (prompt) {
          sendMessage(prompt);
        }
      });
    });

    // Quick Prompt Cards inside hero
    document.querySelectorAll('.quick-card').forEach((card) => {
      card.addEventListener('click', () => {
        const prompt = card.getAttribute('data-prompt');
        if (prompt) {
          sendMessage(prompt);
        }
      });
    });

    // Theme Switches
    themeToggleQuickBtn.addEventListener('click', toggleQuickTheme);
    themeSegments.forEach((btn) => {
      btn.addEventListener('click', () => {
        const themeVal = btn.getAttribute('data-theme-val');
        if (themeVal) applyTheme(themeVal);
      });
    });

    // API Key Settings
    saveApiKeyBtn.addEventListener('click', saveApiKey);
    toggleKeyVisibilityBtn.addEventListener('click', toggleKeyVisibility);

    // Toggle right panel on smaller viewports
    if (toggleSettingsBtn && sidebarRight) {
      toggleSettingsBtn.addEventListener('click', () => {
        sidebarRight.classList.toggle('open');
      });
    }
  }

  // --- Init ---
  function init() {
    initTheme();
    initSession();
    initApiKey();
    initEventListeners();
    messageInput.focus();
  }

  document.addEventListener('DOMContentLoaded', init);
})();
