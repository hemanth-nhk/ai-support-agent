/**
 * AI Support Agent - Full Dashboard Controller
 * Phase 1: Real LocalStorage Threads, Thread Switching, Dynamic Recent/Archived Chats,
 * Clear Chat (Fresh Session), Archive/Restore, Delete Confirmation, and Demo Customer Switcher.
 */

(function () {
  'use strict';

  // --- Storage Keys ---
  const STORAGE_KEY_THREADS = 'support_agent_threads';
  const STORAGE_KEY_ACTIVE = 'support_agent_active_thread_id';
  const STORAGE_KEY_CUSTOMER = 'support_agent_demo_customer';
  const STORAGE_KEY_THEME = 'support_theme';
  const STORAGE_KEY_API_KEY = 'support_api_key';

  // --- Demo Customers Dataset (Testing Context Only) ---
  const DEMO_CUSTOMERS = {
    '1': { id: 1, name: 'Alice Johnson', initial: 'A', orders: '1001 (Laptop), 1002 (Mouse)' },
    '2': { id: 2, name: 'Bob Smith', initial: 'B', orders: '1003 (Keyboard)' },
    '3': { id: 3, name: 'Charlie Brown', initial: 'C', orders: '1004 (Monitor)' },
  };

  // --- State ---
  let threads = [];
  let activeThreadId = '';
  let currentSessionId = '';
  let activeDemoCustomer = '1';
  let isLoading = false;
  let pendingDeleteThreadId = null;

  // --- DOM Elements ---
  const chatScrollArea = document.querySelector('.chat-scroll-area');
  const emptyState = document.getElementById('emptyState');
  const floatingPromptsBar = document.getElementById('floatingPromptsBar');
  const messagesList = document.getElementById('messagesList');
  const chatForm = document.getElementById('chatForm');
  const messageInput = document.getElementById('messageInput');
  const sendBtn = document.getElementById('sendBtn');
  const newChatBtn = document.getElementById('newChatBtn');
  const sessionUuidText = document.getElementById('sessionUuidText');

  // Toolbar & Header Controls
  const activeThreadTitle = document.getElementById('activeThreadTitle');
  const clearChatBtn = document.getElementById('clearChatBtn');
  const archiveChatBtn = document.getElementById('archiveChatBtn');
  const deleteChatBtn = document.getElementById('deleteChatBtn');

  // Dynamic Sidebar Elements
  const recentChatsList = document.getElementById('recentChatsList');
  const archivedSection = document.getElementById('archivedSection');
  const archivedToggleBtn = document.getElementById('archivedToggleBtn');
  const archivedCountText = document.getElementById('archivedCountText');
  const archivedChatsList = document.getElementById('archivedChatsList');

  // Profile Dropdown Elements
  const userProfileBtn = document.getElementById('userProfileBtn');
  const profileDropdown = document.getElementById('profileDropdown');
  const profileBadgeInitial = document.getElementById('profileBadgeInitial');
  const profileBadgeName = document.getElementById('profileBadgeName');
  const resetDemoCustomerBtn = document.getElementById('resetDemoCustomerBtn');

  // Attachment Elements (PDF, DOCX, TXT)
  const attachDocBtn = document.getElementById('attachDocBtn') || document.getElementById('attachmentPlaceholderBtn');
  const documentFileInput = document.getElementById('documentFileInput');
  const attachmentChipBar = document.getElementById('attachmentChipBar');
  const attachmentChip = document.getElementById('attachmentChip');
  const attachmentFilename = document.getElementById('attachmentFilename');
  const attachmentMeta = document.getElementById('attachmentMeta');
  const removeAttachmentBtn = document.getElementById('removeAttachmentBtn');

  // Delete Confirmation Modal Elements
  const deleteConfirmModal = document.getElementById('deleteConfirmModal');
  const cancelDeleteBtn = document.getElementById('cancelDeleteBtn');
  const confirmDeleteBtn = document.getElementById('confirmDeleteBtn');

  // Toast Container
  const toastContainer = document.getElementById('toastContainer');

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

  // ==========================================================================
  // Toast Utility
  // ==========================================================================
  function showToast(message, duration = 3000) {
    if (!toastContainer) return;
    const toast = document.createElement('div');
    toast.className = 'toast-message';
    toast.textContent = message;
    toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transition = 'opacity 0.25s ease';
      setTimeout(() => toast.remove(), 250);
    }, duration);
  }

  // ==========================================================================
  // Thread & LocalStorage Management
  // ==========================================================================
  function loadThreadsFromStorage() {
    try {
      const data = localStorage.getItem(STORAGE_KEY_THREADS);
      threads = data ? JSON.parse(data) : [];
    } catch (_) {
      threads = [];
    }
  }

  function saveThreadsToStorage() {
    try {
      localStorage.setItem(STORAGE_KEY_THREADS, JSON.stringify(threads));
    } catch (err) {
      console.error('Failed to save threads', err);
    }
  }

  function getActiveThread() {
    return threads.find((t) => t.id === activeThreadId) || null;
  }

  function createNewThread(initialTitle = 'New Conversation') {
    const newId = crypto.randomUUID ? crypto.randomUUID() : ('sess-' + Date.now());
    const newThread = {
      id: newId,
      title: initialTitle,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      isArchived: false,
      messages: [],
      attachment: null,
    };
    threads.unshift(newThread);
    saveThreadsToStorage();
    switchToThread(newId);
    return newThread;
  }

  function switchToThread(threadId) {
    activeThreadId = threadId;
    currentSessionId = threadId;
    localStorage.setItem(STORAGE_KEY_ACTIVE, threadId);

    const thread = getActiveThread();
    if (!thread) {
      createNewThread();
      return;
    }

    if (activeThreadTitle) {
      activeThreadTitle.textContent = thread.title || 'New Conversation';
    }

    if (sessionUuidText) {
      sessionUuidText.textContent = `${thread.id.slice(0, 8)}...`;
      sessionUuidText.title = `Full Session ID: ${thread.id}`;
    }

    renderThreadMessages(thread);
    renderSidebarThreads();
    updateAttachmentUI(thread.attachment || null);
    messageInput.focus();
  }

  function renderThreadMessages(thread) {
    messagesList.innerHTML = '';
    if (!thread || !thread.messages || thread.messages.length === 0) {
      messagesList.style.display = 'none';
      if (emptyState) emptyState.style.display = 'block';
      if (floatingPromptsBar) floatingPromptsBar.style.display = 'flex';
    } else {
      if (emptyState) emptyState.style.display = 'none';
      if (floatingPromptsBar) floatingPromptsBar.style.display = 'none';
      messagesList.style.display = 'flex';
      thread.messages.forEach((msg) => {
        if (msg.sender === 'user') {
          renderUserMessageElement(msg.text, msg.timestamp);
        } else {
          renderAssistantMessageElement(msg.data, msg.timestamp);
        }
      });
      scrollToBottom();
    }
  }

  // ==========================================================================
  // Dynamic Sidebar Renderers (Recent & Archived)
  // ==========================================================================
  function renderSidebarThreads() {
    if (!recentChatsList) return;
    recentChatsList.innerHTML = '';

    const activeList = threads.filter((t) => !t.isArchived).sort((a, b) => b.updatedAt - a.updatedAt);
    const archivedList = threads.filter((t) => t.isArchived).sort((a, b) => b.updatedAt - a.updatedAt);

    if (archivedCountText) {
      archivedCountText.textContent = archivedList.length;
    }

    // Render Active Recent Chats
    if (activeList.length === 0) {
      const emptyNote = document.createElement('div');
      emptyNote.style.padding = '8px 10px';
      emptyNote.style.fontSize = '0.74rem';
      emptyNote.style.color = 'var(--text-muted)';
      emptyNote.textContent = 'No recent conversations';
      recentChatsList.appendChild(emptyNote);
    } else {
      activeList.forEach((t) => {
        const item = document.createElement('div');
        item.className = `recent-chat-item ${t.id === activeThreadId ? 'active' : ''}`;

        const content = document.createElement('div');
        content.className = 'recent-chat-content';
        content.innerHTML = `
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
          </svg>
          <span class="recent-chat-title" title="${escapeHtml(t.title)}">${escapeHtml(t.title)}</span>
        `;
        content.addEventListener('click', () => switchToThread(t.id));

        // Action buttons (Archive, Delete)
        const actions = document.createElement('div');
        actions.className = 'item-action-btns';

        const archiveBtn = document.createElement('button');
        archiveBtn.className = 'item-micro-btn';
        archiveBtn.title = 'Archive conversation';
        archiveBtn.innerHTML = `
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="21 8 21 21 3 21 3 8"></polyline>
            <rect x="1" y="3" width="22" height="5"></rect>
            <line x1="10" y1="12" x2="14" y2="12"></line>
          </svg>
        `;
        archiveBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          archiveThread(t.id);
        });

        const delBtn = document.createElement('button');
        delBtn.className = 'item-micro-btn danger';
        delBtn.title = 'Delete conversation';
        delBtn.innerHTML = `
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
          </svg>
        `;
        delBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          promptDeleteThread(t.id);
        });

        actions.appendChild(archiveBtn);
        actions.appendChild(delBtn);

        item.appendChild(content);
        item.appendChild(actions);
        recentChatsList.appendChild(item);
      });
    }

    // Render Archived Chats
    if (archivedChatsList) {
      archivedChatsList.innerHTML = '';
      if (archivedList.length === 0) {
        const noArch = document.createElement('div');
        noArch.style.padding = '6px 8px';
        noArch.style.fontSize = '0.72rem';
        noArch.style.color = 'var(--text-muted)';
        noArch.textContent = 'No archived conversations';
        archivedChatsList.appendChild(noArch);
      } else {
        archivedList.forEach((t) => {
          const item = document.createElement('div');
          item.className = 'recent-chat-item';

          const content = document.createElement('div');
          content.className = 'recent-chat-content';
          content.innerHTML = `
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="21 8 21 21 3 21 3 8"></polyline>
              <rect x="1" y="3" width="22" height="5"></rect>
            </svg>
            <span class="recent-chat-title" title="${escapeHtml(t.title)}">${escapeHtml(t.title)}</span>
          `;
          content.addEventListener('click', () => switchToThread(t.id));

          const actions = document.createElement('div');
          actions.className = 'item-action-btns';

          const restoreBtn = document.createElement('button');
          restoreBtn.className = 'item-micro-btn';
          restoreBtn.title = 'Restore to active';
          restoreBtn.innerHTML = `
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="1 4 1 10 7 10"></polyline>
              <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"></path>
            </svg>
          `;
          restoreBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            restoreThread(t.id);
          });

          const delBtn = document.createElement('button');
          delBtn.className = 'item-micro-btn danger';
          delBtn.title = 'Delete permanently';
          delBtn.innerHTML = `
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          `;
          delBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            promptDeleteThread(t.id);
          });

          actions.appendChild(restoreBtn);
          actions.appendChild(delBtn);

          item.appendChild(content);
          item.appendChild(actions);
          archivedChatsList.appendChild(item);
        });
      }
    }
  }

  // ==========================================================================
  // Attachment Management (PDF, DOCX, TXT)
  // ==========================================================================
  function updateAttachmentUI(attachment, isUploading = false, uploadingName = '') {
    if (!attachmentChipBar) return;

    if (isUploading) {
      attachmentChipBar.style.display = 'flex';
      if (attachmentChip) attachmentChip.className = 'attachment-chip uploading';
      if (attachmentFilename) attachmentFilename.textContent = uploadingName;
      if (attachmentMeta) attachmentMeta.textContent = 'Uploading...';
      if (removeAttachmentBtn) removeAttachmentBtn.disabled = true;
      if (attachDocBtn) attachDocBtn.disabled = true;
      return;
    }

    if (attachDocBtn) attachDocBtn.disabled = false;

    if (attachment && attachment.document_id) {
      attachmentChipBar.style.display = 'flex';
      if (attachmentChip) attachmentChip.className = 'attachment-chip';
      if (attachmentFilename) {
        attachmentFilename.textContent = attachment.filename || 'Document';
        attachmentFilename.title = attachment.filename || 'Document';
      }
      if (attachmentMeta) {
        const count = attachment.chunk_count || 1;
        attachmentMeta.textContent = `${count} ${count === 1 ? 'chunk' : 'chunks'}`;
      }
      if (removeAttachmentBtn) removeAttachmentBtn.disabled = false;
    } else {
      attachmentChipBar.style.display = 'none';
      if (attachmentFilename) attachmentFilename.textContent = '';
      if (attachmentMeta) attachmentMeta.textContent = '';
    }
  }

  async function handleFileUpload(file) {
    if (!file) return;

    const allowedExts = ['.pdf', '.docx', '.txt'];
    const lowerName = file.name.toLowerCase();
    const isAllowed = allowedExts.some((ext) => lowerName.endsWith(ext));
    if (!isAllowed) {
      showToast('Unsupported file type. Please upload a PDF, DOCX, or TXT file.');
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      showToast('File exceeds the maximum allowed size of 10MB.');
      return;
    }

    if (file.size === 0) {
      showToast('Cannot upload an empty document.');
      return;
    }

    let thread = getActiveThread();
    if (!thread) {
      thread = createNewThread();
    }

    updateAttachmentUI(null, true, file.name);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('session_id', currentSessionId);

      const res = await fetch('/documents/upload', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        const msg = errData.detail || `Upload failed (${res.status})`;
        showToast(msg);
        updateAttachmentUI(thread ? thread.attachment : null);
        return;
      }

      const data = await res.json();
      thread.attachment = {
        document_id: data.document_id,
        filename: data.filename,
        chunk_count: data.chunk_count,
      };
      thread.updatedAt = Date.now();
      saveThreadsToStorage();
      updateAttachmentUI(thread.attachment);
      showToast(`Attached ${data.filename} (${data.chunk_count} chunks)`);
    } catch (err) {
      console.error('Document upload error:', err);
      showToast('Network error uploading document.');
      updateAttachmentUI(thread ? thread.attachment : null);
    }
  }

  async function handleRemoveAttachment() {
    const thread = getActiveThread();
    if (!thread || !thread.attachment) return;

    const docId = thread.attachment.document_id;
    if (attachmentMeta) attachmentMeta.textContent = 'Removing...';
    if (removeAttachmentBtn) removeAttachmentBtn.disabled = true;

    try {
      const res = await fetch(`/documents/${encodeURIComponent(docId)}?session_id=${encodeURIComponent(currentSessionId)}`, {
        method: 'DELETE',
      });

      if (res.ok || res.status === 404) {
        thread.attachment = null;
        thread.updatedAt = Date.now();
        saveThreadsToStorage();
        updateAttachmentUI(null);
        showToast('Document attachment removed.');
      } else {
        const errData = await res.json().catch(() => ({}));
        showToast(errData.detail || 'Failed to remove document.');
        updateAttachmentUI(thread.attachment);
      }
    } catch (err) {
      console.error('Delete attachment error:', err);
      showToast('Network error removing document.');
      updateAttachmentUI(thread.attachment);
    }
  }

  // ==========================================================================
  // Chat Actions: Clear, Archive, Delete
  // ==========================================================================
  function clearCurrentChat() {
    // Generates a genuinely FRESH session to clear backend in-memory state
    const newSessionId = crypto.randomUUID ? crypto.randomUUID() : ('sess-' + Date.now());
    let thread = getActiveThread();

    if (thread) {
      thread.id = newSessionId;
      thread.messages = [];
      thread.attachment = null;
      thread.title = 'New Conversation';
      thread.updatedAt = Date.now();
      saveThreadsToStorage();
    } else {
      createNewThread();
    }

    switchToThread(newSessionId);
    updateAttachmentUI(null);
    showToast('Chat cleared — fresh session started');
  }

  function archiveThread(threadId) {
    const thread = threads.find((t) => t.id === threadId);
    if (!thread) return;

    thread.isArchived = true;
    thread.updatedAt = Date.now();
    saveThreadsToStorage();

    showToast('Conversation archived');

    if (activeThreadId === threadId) {
      const nextActive = threads.find((t) => !t.isArchived);
      if (nextActive) {
        switchToThread(nextActive.id);
      } else {
        createNewThread();
      }
    } else {
      renderSidebarThreads();
    }
  }

  function restoreThread(threadId) {
    const thread = threads.find((t) => t.id === threadId);
    if (!thread) return;

    thread.isArchived = false;
    thread.updatedAt = Date.now();
    saveThreadsToStorage();

    showToast('Conversation restored to active');
    renderSidebarThreads();
  }

  function promptDeleteThread(threadId) {
    pendingDeleteThreadId = threadId;
    if (deleteConfirmModal) {
      deleteConfirmModal.style.display = 'flex';
    }
  }

  function confirmDelete() {
    if (!pendingDeleteThreadId) return;

    threads = threads.filter((t) => t.id !== pendingDeleteThreadId);
    saveThreadsToStorage();

    showToast('Conversation permanently deleted');

    if (activeThreadId === pendingDeleteThreadId) {
      const next = threads.find((t) => !t.isArchived);
      if (next) {
        switchToThread(next.id);
      } else {
        createNewThread();
      }
    } else {
      renderSidebarThreads();
    }

    closeDeleteModal();
  }

  function closeDeleteModal() {
    pendingDeleteThreadId = null;
    if (deleteConfirmModal) {
      deleteConfirmModal.style.display = 'none';
    }
  }

  // ==========================================================================
  // Demo Customer Context (Testing Context Only)
  // ==========================================================================
  function initDemoCustomer() {
    activeDemoCustomer = localStorage.getItem(STORAGE_KEY_CUSTOMER) || '1';
    applyDemoCustomer(activeDemoCustomer, false);
  }

  function applyDemoCustomer(customerId, notify = true) {
    const cust = DEMO_CUSTOMERS[customerId] || DEMO_CUSTOMERS['1'];
    activeDemoCustomer = String(cust.id);
    localStorage.setItem(STORAGE_KEY_CUSTOMER, activeDemoCustomer);

    if (profileBadgeInitial) profileBadgeInitial.textContent = cust.initial;
    if (profileBadgeName) profileBadgeName.textContent = cust.name.split(' ')[0];

    const radio = document.querySelector(`input[name="demoCustomer"][value="${activeDemoCustomer}"]`);
    if (radio) radio.checked = true;

    if (notify) {
      showToast(`Switched Demo Customer: ${cust.name} (#${cust.id})`);
    }
  }

  function toggleProfileDropdown() {
    if (!profileDropdown) return;
    const isVisible = profileDropdown.style.display === 'flex';
    profileDropdown.style.display = isVisible ? 'none' : 'flex';
  }

  function closeProfileDropdown() {
    if (profileDropdown) profileDropdown.style.display = 'none';
  }

  // ==========================================================================
  // Theme Management
  // ==========================================================================
  function applyTheme(theme) {
    let resolvedTheme = theme;
    if (theme === 'system') {
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      resolvedTheme = prefersDark ? 'dark' : 'light';
    }
    document.documentElement.setAttribute('data-theme', resolvedTheme);
    localStorage.setItem(STORAGE_KEY_THEME, theme);
  }

  function initTheme() {
    const savedTheme = localStorage.getItem(STORAGE_KEY_THEME) || 'dark';
    applyTheme(savedTheme);
  }

  function toggleQuickTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
    const nextTheme = currentTheme === 'dark' ? 'light' : 'dark';
    applyTheme(nextTheme);
  }

  // ==========================================================================
  // API Key Settings
  // ==========================================================================
  function getApiKey() {
    return localStorage.getItem(STORAGE_KEY_API_KEY) || '';
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
    if (key) apiKeyInput.value = key;
    updateApiKeyStatusDisplay();
  }

  function saveApiKey() {
    const val = apiKeyInput.value.trim();
    if (val) {
      localStorage.setItem(STORAGE_KEY_API_KEY, val);
    } else {
      localStorage.removeItem(STORAGE_KEY_API_KEY);
    }
    updateApiKeyStatusDisplay();
    showToast(val ? 'Support API key saved' : 'Support API key cleared');
  }

  function toggleKeyVisibility() {
    const isPassword = apiKeyInput.type === 'password';
    apiKeyInput.type = isPassword ? 'text' : 'password';
  }

  // ==========================================================================
  // UI Message Rendering Helpers
  // ==========================================================================
  function formatTime(date) {
    const d = date ? new Date(date) : new Date();
    return d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  function scrollToBottom() {
    if (chatScrollArea) chatScrollArea.scrollTop = chatScrollArea.scrollHeight;
  }

  function ensureMessagesListVisible() {
    if (emptyState) emptyState.style.display = 'none';
    if (floatingPromptsBar) floatingPromptsBar.style.display = 'none';
    messagesList.style.display = 'flex';
  }

  function renderUserMessageElement(text, timestamp) {
    ensureMessagesListVisible();

    const turn = document.createElement('div');
    turn.className = 'chat-turn user';

    const avatar = document.createElement('div');
    avatar.className = 'turn-avatar user-circle';
    avatar.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
        <circle cx="12" cy="7" r="4"></circle>
      </svg>
    `;

    const content = document.createElement('div');
    content.className = 'turn-content';

    const bubble = document.createElement('div');
    bubble.className = 'turn-bubble';
    bubble.textContent = text;
    content.appendChild(bubble);

    const time = document.createElement('div');
    time.className = 'turn-time';
    time.textContent = timestamp || formatTime();
    content.appendChild(time);

    turn.appendChild(avatar);
    turn.appendChild(content);
    messagesList.appendChild(turn);
  }

  function getStatusPillClass(status) {
    const s = (status || '').toLowerCase();
    if (s === 'ok') return 'pill-status-ok';
    if (s === 'clarification' || s === 'needs_clarification') return 'pill-status-warn';
    return 'pill-status-error';
  }

  function renderAssistantMessageElement(data, timestamp) {
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
        <line x1="8" y1="16" x2="8.01" y2="16"></line>
        <line x1="16" y1="16" x2="16.01" y2="16"></line>
      </svg>
    `;

    const content = document.createElement('div');
    content.className = 'turn-content';

    const bubble = document.createElement('div');
    bubble.className = 'turn-bubble';
    bubble.textContent = data.response;
    content.appendChild(bubble);

    const time = document.createElement('div');
    time.className = 'turn-time';
    time.textContent = timestamp || formatTime();
    content.appendChild(time);

    // Telemetry Badges Row
    const telemetry = document.createElement('div');
    telemetry.className = 'telemetry-row';

    const intentPill = document.createElement('span');
    intentPill.className = 'telemetry-pill pill-intent';
    intentPill.textContent = `Intent: ${data.intent || 'UNKNOWN'}`;
    telemetry.appendChild(intentPill);
    telemetry.appendChild(document.createTextNode('\n'));

    const toolPill = document.createElement('span');
    toolPill.className = 'telemetry-pill pill-tool';
    toolPill.textContent = `Tool: ${data.tool_used || 'none'}`;
    telemetry.appendChild(toolPill);
    telemetry.appendChild(document.createTextNode('\n'));

    const statusPill = document.createElement('span');
    statusPill.className = `telemetry-pill ${getStatusPillClass(data.status)}`;
    statusPill.textContent = `Status: ${data.status || 'unknown'}`;
    telemetry.appendChild(statusPill);
    telemetry.appendChild(document.createTextNode('\n'));

    const count = typeof data.retrieval_count === 'number' ? data.retrieval_count : 0;
    const retPill = document.createElement('span');
    retPill.className = 'telemetry-pill pill-retrievals';
    retPill.textContent = `Retrievals: ${count}`;
    telemetry.appendChild(retPill);

    content.appendChild(telemetry);

    turn.appendChild(avatar);
    turn.appendChild(content);
    messagesList.appendChild(turn);
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
    const el = document.getElementById('activeTypingIndicator');
    if (el) el.remove();
  }

  // ==========================================================================
  // Send Message Logic
  // ==========================================================================
  async function sendMessage(text) {
    const message = text.trim();
    if (!message || isLoading) return;

    let thread = getActiveThread();
    if (!thread) {
      thread = createNewThread();
    }

    // Auto-title thread from first message
    if (thread.messages.length === 0 || thread.title === 'New Conversation') {
      thread.title = message.length > 28 ? message.slice(0, 28) + '...' : message;
      if (activeThreadTitle) activeThreadTitle.textContent = thread.title;
    }

    const userTime = formatTime();
    renderUserMessageElement(message, userTime);

    thread.messages.push({
      sender: 'user',
      text: message,
      timestamp: userTime,
    });
    thread.updatedAt = Date.now();
    saveThreadsToStorage();
    renderSidebarThreads();

    messageInput.value = '';
    isLoading = true;
    sendBtn.disabled = true;
    messageInput.disabled = true;
    appendTypingIndicator();

    try {
      const headers = { 'Content-Type': 'application/json' };
      const apiKey = getApiKey();
      if (apiKey && apiKey.trim().length > 0) {
        headers['X-API-Key'] = apiKey.trim();
      }

      const payload = {
        message: message,
        session_id: currentSessionId,
      };

      if (thread && thread.attachment && thread.attachment.document_id) {
        payload.document_id = thread.attachment.document_id;
      }

      const res = await fetch('/chat', {
        method: 'POST',
        headers: headers,
        body: JSON.stringify(payload),
      });

      removeTypingIndicator();

      let assistantData;
      if (res.status === 401) {
        assistantData = {
          response: 'Authentication Required (401): This structured data query requires an authorized API key. Please configure your key in Settings on the right.',
          intent: 'DATA_QUERY',
          tool_used: 'none',
          status: 'error',
          retrieval_count: 0,
        };
      } else if (!res.ok) {
        let detail = `HTTP ${res.status}`;
        try {
          const err = await res.json();
          if (err && err.detail) detail = typeof err.detail === 'string' ? err.detail : JSON.stringify(err.detail);
        } catch (_) {}
        assistantData = {
          response: `Request failed: ${detail}`,
          intent: 'ERROR',
          tool_used: 'none',
          status: 'error',
          retrieval_count: 0,
        };
      } else {
        assistantData = await res.json();
      }

      const assistantTime = formatTime();
      renderAssistantMessageElement(assistantData, assistantTime);

      thread.messages.push({
        sender: 'assistant',
        data: assistantData,
        timestamp: assistantTime,
      });
      thread.updatedAt = Date.now();
      saveThreadsToStorage();
      renderSidebarThreads();
      scrollToBottom();
    } catch (err) {
      removeTypingIndicator();
      const errData = {
        response: `Network error: ${err.message || 'Unable to connect to support server'}`,
        intent: 'NETWORK_ERROR',
        tool_used: 'none',
        status: 'error',
        retrieval_count: 0,
      };
      const assistantTime = formatTime();
      renderAssistantMessageElement(errData, assistantTime);
      thread.messages.push({
        sender: 'assistant',
        data: errData,
        timestamp: assistantTime,
      });
      saveThreadsToStorage();
    } finally {
      isLoading = false;
      sendBtn.disabled = false;
      messageInput.disabled = false;
      messageInput.focus();
    }
  }

  // ==========================================================================
  // Event Listeners Setup
  // ==========================================================================
  function initEventListeners() {
    // Form submit
    chatForm.addEventListener('submit', (e) => {
      e.preventDefault();
      sendMessage(messageInput.value);
    });

    // Enter to submit
    messageInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage(messageInput.value);
      }
    });

    // New Chat
    newChatBtn.addEventListener('click', () => {
      createNewThread();
      showToast('Started new conversation');
    });

    // Chat Top Controls: Clear, Archive, Delete
    clearChatBtn.addEventListener('click', clearCurrentChat);
    archiveChatBtn.addEventListener('click', () => {
      if (activeThreadId) archiveThread(activeThreadId);
    });
    deleteChatBtn.addEventListener('click', () => {
      if (activeThreadId) promptDeleteThread(activeThreadId);
    });

    // Delete Modal Actions
    cancelDeleteBtn.addEventListener('click', closeDeleteModal);
    confirmDeleteBtn.addEventListener('click', confirmDelete);
    deleteConfirmModal.addEventListener('click', (e) => {
      if (e.target === deleteConfirmModal) closeDeleteModal();
    });

    // Archived Section Toggle
    archivedToggleBtn.addEventListener('click', () => {
      const isVisible = archivedChatsList.style.display === 'flex';
      archivedChatsList.style.display = isVisible ? 'none' : 'flex';
      archivedToggleBtn.classList.toggle('open', !isVisible);
    });

    // Quick Prompts inside Welcome card
    document.querySelectorAll('.quick-card').forEach((card) => {
      card.addEventListener('click', () => {
        const p = card.getAttribute('data-prompt');
        if (p) sendMessage(p);
      });
    });

    // Left Navigation items (Support categories populate composer input rather than auto-sending)
    const categoryTemplates = {
      'order-tracking': 'Where is my order ',
      'returns': 'What is your return and refund policy?',
      'products': 'Find orders for ',
      'faq': 'What is your shipping policy?',
    };

    document.querySelectorAll('.nav-item').forEach((item) => {
      item.addEventListener('click', () => {
        document.querySelectorAll('.nav-item').forEach((n) => n.classList.remove('active'));
        item.classList.add('active');

        const action = item.getAttribute('data-action');
        if (action && categoryTemplates[action]) {
          messageInput.value = categoryTemplates[action];
          messageInput.focus();
          messageInput.setSelectionRange(messageInput.value.length, messageInput.value.length);
        }
      });
    });

    // Floating prompt chips at the start of new chat
    document.querySelectorAll('.floating-prompt-chip').forEach((chip) => {
      chip.addEventListener('click', () => {
        const p = chip.getAttribute('data-prompt');
        if (p) sendMessage(p);
      });
    });

    // Attachment upload & remove controls
    if (attachDocBtn && documentFileInput) {
      attachDocBtn.addEventListener('click', () => {
        documentFileInput.click();
      });
      documentFileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
          handleFileUpload(e.target.files[0]);
          documentFileInput.value = '';
        }
      });
    }
    if (removeAttachmentBtn) {
      removeAttachmentBtn.addEventListener('click', handleRemoveAttachment);
    }

    // Profile Dropdown Toggle & Selection
    userProfileBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      toggleProfileDropdown();
    });

    document.querySelectorAll('input[name="demoCustomer"]').forEach((radio) => {
      radio.addEventListener('change', (e) => {
        applyDemoCustomer(e.target.value, true);
        closeProfileDropdown();
      });
    });

    if (resetDemoCustomerBtn) {
      resetDemoCustomerBtn.addEventListener('click', () => {
        applyDemoCustomer('1', true);
        closeProfileDropdown();
      });
    }

    document.addEventListener('click', (e) => {
      if (profileDropdown && !profileDropdown.contains(e.target) && e.target !== userProfileBtn) {
        closeProfileDropdown();
      }
    });

    // Theme toggle (header only)
    themeToggleQuickBtn.addEventListener('click', toggleQuickTheme);

    // Settings actions
    saveApiKeyBtn.addEventListener('click', saveApiKey);
    toggleKeyVisibilityBtn.addEventListener('click', toggleKeyVisibility);

    if (toggleSettingsBtn && sidebarRight) {
      toggleSettingsBtn.addEventListener('click', () => {
        sidebarRight.classList.toggle('open');
      });
    }

    // Escape key closes modals
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        closeDeleteModal();
        closeProfileDropdown();
      }
    });
  }

  // ==========================================================================
  // App Bootstrapping
  // ==========================================================================
  function init() {
    initTheme();
    initApiKey();
    initDemoCustomer();
    loadThreadsFromStorage();

    const savedActiveId = localStorage.getItem(STORAGE_KEY_ACTIVE);
    if (savedActiveId && threads.some((t) => t.id === savedActiveId)) {
      switchToThread(savedActiveId);
    } else if (threads.length > 0) {
      const firstActive = threads.find((t) => !t.isArchived) || threads[0];
      switchToThread(firstActive.id);
    } else {
      createNewThread();
    }

    initEventListeners();
  }

  document.addEventListener('DOMContentLoaded', init);
})();
