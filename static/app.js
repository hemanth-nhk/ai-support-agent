/**
 * AI Support Agent - Frontend Controller
 * Minimal, focused, single-session interface connected to FastAPI /chat.
 */

(function () {
  'use strict';

  // --- State ---
  let currentSessionId = '';
  let isLoading = false;

  // --- DOM Elements ---
  const chatMain = document.getElementById('chatMain');
  const emptyState = document.getElementById('emptyState');
  const messagesList = document.getElementById('messagesList');
  const chatForm = document.getElementById('chatForm');
  const messageInput = document.getElementById('messageInput');
  const sendBtn = document.getElementById('sendBtn');
  const newChatBtn = document.getElementById('newChatBtn');
  const sessionIdDisplay = document.getElementById('sessionIdDisplay');

  // Settings Modal Elements
  const settingsBtn = document.getElementById('settingsBtn');
  const settingsModal = document.getElementById('settingsModal');
  const closeSettingsBtn = document.getElementById('closeSettingsBtn');
  const apiKeyInput = document.getElementById('apiKeyInput');
  const saveApiKeyBtn = document.getElementById('saveApiKeyBtn');
  const clearApiKeyBtn = document.getElementById('clearApiKeyBtn');
  const apiKeyStatusBox = document.getElementById('apiKeyStatusBox');
  const apiKeyStatusText = document.getElementById('apiKeyStatusText');

  // Theme Toggle Elements
  const themeToggleBtn = document.getElementById('themeToggleBtn');

  // --- Session Management ---
  function initSession() {
    currentSessionId = crypto.randomUUID ? crypto.randomUUID() : ('sess-' + Date.now());
    if (sessionIdDisplay) {
      sessionIdDisplay.textContent = `Session: ${currentSessionId.slice(0, 8)}...`;
      sessionIdDisplay.title = `Full Session ID: ${currentSessionId}`;
    }
  }

  function startNewChat() {
    initSession();
    messagesList.innerHTML = '';
    messagesList.style.display = 'none';
    emptyState.style.display = 'block';
    messageInput.value = '';
    messageInput.focus();
  }

  // --- Theme Management ---
  function initTheme() {
    const savedTheme = localStorage.getItem('support_theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
  }

  function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
    const nextTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', nextTheme);
    localStorage.setItem('support_theme', nextTheme);
  }

  // --- API Key Management ---
  function getApiKey() {
    return localStorage.getItem('support_api_key') || '';
  }

  function updateApiKeyStatusDisplay() {
    const key = getApiKey();
    const dot = apiKeyStatusBox.querySelector('.key-status-dot');
    if (key && key.trim().length > 0) {
      apiKeyStatusText.textContent = `Configured (${'•'.repeat(Math.min(key.length, 12))})`;
      if (dot) dot.classList.add('active');
    } else {
      apiKeyStatusText.textContent = 'No key configured';
      if (dot) dot.classList.remove('active');
    }
  }

  function openSettings() {
    apiKeyInput.value = getApiKey();
    updateApiKeyStatusDisplay();
    settingsModal.style.display = 'flex';
    apiKeyInput.focus();
  }

  function closeSettings() {
    settingsModal.style.display = 'none';
  }

  function saveApiKey() {
    const val = apiKeyInput.value.trim();
    if (val) {
      localStorage.setItem('support_api_key', val);
    } else {
      localStorage.removeItem('support_api_key');
    }
    updateApiKeyStatusDisplay();
    closeSettings();
  }

  function clearApiKey() {
    localStorage.removeItem('support_api_key');
    apiKeyInput.value = '';
    updateApiKeyStatusDisplay();
  }

  // --- Message UI Rendering ---
  function scrollToBottom() {
    chatMain.scrollTop = chatMain.scrollHeight;
  }

  function ensureMessagesListVisible() {
    emptyState.style.display = 'none';
    messagesList.style.display = 'flex';
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  function appendUserMessage(text) {
    ensureMessagesListVisible();

    const item = document.createElement('div');
    item.className = 'message-row user';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.textContent = text;

    item.appendChild(bubble);
    messagesList.appendChild(item);
    scrollToBottom();
  }

  function appendTypingIndicator() {
    ensureMessagesListVisible();

    const indicator = document.createElement('div');
    indicator.className = 'message-row assistant';
    indicator.id = 'activeTypingIndicator';

    const bubble = document.createElement('div');
    bubble.className = 'typing-bubble';
    bubble.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';

    indicator.appendChild(bubble);
    messagesList.appendChild(indicator);
    scrollToBottom();
  }

  function removeTypingIndicator() {
    const indicator = document.getElementById('activeTypingIndicator');
    if (indicator) {
      indicator.remove();
    }
  }

  function appendAssistantMessage(data) {
    ensureMessagesListVisible();

    const item = document.createElement('div');
    item.className = 'message-row assistant';

    // Message Bubble
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.textContent = data.response;
    item.appendChild(bubble);

    // Subtle Developer Metadata Line (secondary, low contrast)
    const meta = document.createElement('div');
    meta.className = 'metadata-line';

    const statusVal = (data.status || 'unknown').toLowerCase();
    let statusClass = 'meta-val';
    if (statusVal === 'ok') statusClass = 'meta-val-ok';
    else if (statusVal === 'clarification' || statusVal === 'needs_clarification') statusClass = 'meta-val-warn';
    else if (statusVal === 'error') statusClass = 'meta-val-err';

    meta.innerHTML = `
      <span class="meta-item"><span class="meta-key">intent:</span> <span class="meta-val">${escapeHtml(data.intent || 'UNKNOWN')}</span></span>
      <span class="meta-item">&bull;</span>
      <span class="meta-item"><span class="meta-key">tool:</span> <span class="meta-val">${escapeHtml(data.tool_used || 'none')}</span></span>
      <span class="meta-item">&bull;</span>
      <span class="meta-item"><span class="meta-key">status:</span> <span class="${statusClass}">${escapeHtml(data.status || 'unknown')}</span></span>
      <span class="meta-item">&bull;</span>
      <span class="meta-item"><span class="meta-key">retrievals:</span> <span class="meta-val">${typeof data.retrieval_count === 'number' ? data.retrieval_count : 0}</span></span>
    `;

    item.appendChild(meta);
    messagesList.appendChild(item);
    scrollToBottom();
  }

  function appendAuthErrorAlert() {
    ensureMessagesListVisible();

    const item = document.createElement('div');
    item.className = 'message-row assistant';

    const alertBox = document.createElement('div');
    alertBox.className = 'alert-card alert-auth';
    alertBox.innerHTML = `
      <div>
        <strong>Authentication Required (401)</strong><br />
        This query requires an authorized API key.
      </div>
      <button class="btn btn-primary btn-sm" id="openSettingsFromAlert">Settings</button>
    `;
    item.appendChild(alertBox);

    const meta = document.createElement('div');
    meta.className = 'metadata-line';
    meta.innerHTML = `
      <span class="meta-item"><span class="meta-key">intent:</span> <span class="meta-val">DATA_QUERY</span></span>
      <span class="meta-item">&bull;</span>
      <span class="meta-item"><span class="meta-key">status:</span> <span class="meta-val-err">401 Unauthorized</span></span>
    `;
    item.appendChild(meta);

    messagesList.appendChild(item);

    const alertBtn = alertBox.querySelector('#openSettingsFromAlert');
    if (alertBtn) {
      alertBtn.addEventListener('click', openSettings);
    }

    scrollToBottom();
  }

  function appendGeneralError(errorMessage) {
    ensureMessagesListVisible();

    const item = document.createElement('div');
    item.className = 'message-row assistant';

    const alertBox = document.createElement('div');
    alertBox.className = 'alert-card alert-error';
    alertBox.textContent = `Error: ${errorMessage}`;
    item.appendChild(alertBox);

    const meta = document.createElement('div');
    meta.className = 'metadata-line';
    meta.innerHTML = `<span class="meta-item"><span class="meta-key">status:</span> <span class="meta-val-err">failed</span></span>`;
    item.appendChild(meta);

    messagesList.appendChild(item);
    scrollToBottom();
  }

  // --- Chat Submit Handler ---
  async function sendMessage(text) {
    const message = text.trim();
    if (!message || isLoading) return;

    appendUserMessage(message);
    messageInput.value = '';
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

    // New Chat
    newChatBtn.addEventListener('click', startNewChat);

    // Quick Prompts
    document.querySelectorAll('.prompt-chip').forEach((chip) => {
      chip.addEventListener('click', () => {
        const prompt = chip.getAttribute('data-prompt');
        if (prompt) {
          sendMessage(prompt);
        }
      });
    });

    // Theme Toggle
    themeToggleBtn.addEventListener('click', toggleTheme);

    // Settings Modal
    settingsBtn.addEventListener('click', openSettings);
    closeSettingsBtn.addEventListener('click', closeSettings);
    saveApiKeyBtn.addEventListener('click', saveApiKey);
    clearApiKeyBtn.addEventListener('click', clearApiKey);

    settingsModal.addEventListener('click', (e) => {
      if (e.target === settingsModal) {
        closeSettings();
      }
    });

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && settingsModal.style.display === 'flex') {
        closeSettings();
      }
    });
  }

  // --- Initialization ---
  function init() {
    initTheme();
    initSession();
    initEventListeners();
    messageInput.focus();
  }

  document.addEventListener('DOMContentLoaded', init);
})();
