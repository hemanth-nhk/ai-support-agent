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
  const STORAGE_KEY_WORKSPACE_MODE = 'support_workspace_mode';
  const STORAGE_KEY_ACTIVE_ANALYSIS = 'support_active_analysis_id';
  const STORAGE_KEY_SESSION_TOKEN = 'support_session_token';
  const STORAGE_KEY_SIDEBAR_WIDTH = 'support_sidebar_width';

  // --- UI-2: Top-Level Workspace Modes ---
  const WORKSPACE_MODE_SUPPORT = 'support';
  const WORKSPACE_MODE_DATASET = 'dataset';
  let currentWorkspaceMode = WORKSPACE_MODE_SUPPORT;

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
  let currentSessionToken = localStorage.getItem(STORAGE_KEY_SESSION_TOKEN) || '';
  let activeDemoCustomer = '1';

  function getAuthHeaders(baseHeaders = {}) {
    const headers = { ...baseHeaders };
    if (currentSessionToken) {
      headers['X-Session-Token'] = currentSessionToken;
    }
    return headers;
  }
  let isLoading = false;
  let pendingDeleteThreadId = null;
  let pendingDeleteAnalysisId = null;
  let pendingDeleteDatasetId = null;
  let pendingDeleteDatasetSessionId = null;
  let activeModalTriggerEl = null;

  function restoreModalFocus() {
    if (activeModalTriggerEl && typeof activeModalTriggerEl.focus === 'function' && document.body.contains(activeModalTriggerEl)) {
      try {
        activeModalTriggerEl.focus();
      } catch (_) {}
    }
    activeModalTriggerEl = null;
  }

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

  // UI-2: Mode Selector & Sidebar Panels
  const tabSupportModeBtn = document.getElementById('tabSupportModeBtn');
  const tabDatasetModeBtn = document.getElementById('tabDatasetModeBtn');
  const supportSidebarPanel = document.getElementById('supportSidebarPanel');
  const datasetSidebarPanel = document.getElementById('datasetSidebarPanel');

  // UI-2: Sidebar Dataset Context Elements
  const sidebarDatasetUploadBtn = document.getElementById('sidebarDatasetUploadBtn');
  const sidebarSwitchDatasetBtn = document.getElementById('sidebarSwitchDatasetBtn');
  const sidebarDatasetEmptyState = document.getElementById('sidebarDatasetEmptyState');
  const sidebarDatasetActiveCard = document.getElementById('sidebarDatasetActiveCard');
  const sidebarDatasetFilename = document.getElementById('sidebarDatasetFilename');
  const sidebarDatasetFormatTag = document.getElementById('sidebarDatasetFormatTag');
  const sidebarRowsVal = document.getElementById('sidebarRowsVal');
  const sidebarColsVal = document.getElementById('sidebarColsVal');
  const sidebarMissingVal = document.getElementById('sidebarMissingVal');
  const sidebarDupsVal = document.getElementById('sidebarDupsVal');
  const sidebarColsCount = document.getElementById('sidebarColsCount');
  const sidebarColumnsList = document.getElementById('sidebarColumnsList');
  const datasetSessionUuidText = document.getElementById('datasetSessionUuidText');

  // UI-2: Center Schema Table Toggle
  const toggleSchemaTableBtn = document.getElementById('toggleSchemaTableBtn');
  const toggleSchemaTableBtnText = document.getElementById('toggleSchemaTableBtnText');
  const workspaceSchemaSection = document.getElementById('workspaceSchemaSection');

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

  // Dataset Attachment Elements (CSV, XLSX, JSON)
  const attachDatasetBtn = document.getElementById('attachDatasetBtn');
  const datasetFileInput = document.getElementById('datasetFileInput');
  const datasetChipBar = document.getElementById('datasetChipBar');
  const datasetChip = document.getElementById('datasetChip');
  const datasetFilename = document.getElementById('datasetFilename');
  const datasetMeta = document.getElementById('datasetMeta');
  const removeDatasetBtn = document.getElementById('removeDatasetBtn');

  // Dataset Workspace Elements
  const chatMain = document.getElementById('chatMain');
  const datasetWorkspaceView = document.getElementById('datasetWorkspaceView');
  const navDatasetsBtn = document.getElementById('navDatasetsBtn');
  const workspaceActiveDatasetLabel = document.getElementById('workspaceActiveDatasetLabel');
  const workspaceUploadFileInput = document.getElementById('workspaceUploadFileInput');
  const workspaceUploadBtn = document.getElementById('workspaceUploadBtn');
  const workspaceUploadTriggerBtn = document.getElementById('workspaceUploadTriggerBtn');
  const workspaceSwitchBtn = document.getElementById('workspaceSwitchBtn');
  const workspaceRemoveBtn = document.getElementById('workspaceRemoveBtn');
  const workspaceDeleteDatasetBtn = document.getElementById('workspaceDeleteDatasetBtn');
  const sidebarRemoveDatasetBtn = document.getElementById('sidebarRemoveDatasetBtn');
  const switchDatasetModal = document.getElementById('switchDatasetModal');
  const switchDatasetList = document.getElementById('switchDatasetList');
  const switchUploadNewBtn = document.getElementById('switchUploadNewBtn');
  const cancelSwitchDatasetBtn = document.getElementById('cancelSwitchDatasetBtn');
  const closeSwitchDatasetModalBtn = document.getElementById('closeSwitchDatasetModalBtn');
  const workspaceEmptyState = document.getElementById('workspaceEmptyState');
  const workspaceContentCard = document.getElementById('workspaceContentCard');
  const profileFormatTag = document.getElementById('profileFormatTag');
  const profileRowsVal = document.getElementById('profileRowsVal');
  const profileColsVal = document.getElementById('profileColsVal');
  const profileMissingVal = document.getElementById('profileMissingVal');
  const profileDupsVal = document.getElementById('profileDupsVal');
  const workspaceColumnsTbody = document.getElementById('workspaceColumnsTbody');
  const workspaceQueryInput = document.getElementById('workspaceQueryInput');
  const workspaceAnalyzeBtn = document.getElementById('workspaceAnalyzeBtn');
  const analyzeBtnText = document.getElementById('analyzeBtnText');
  const analyzeBtnSpinner = document.getElementById('analyzeBtnSpinner');
  const workspaceResultsCard = document.getElementById('workspaceResultsCard');
  const workspaceFindingText = document.getElementById('workspaceFindingText');
  const workspaceResultTableBox = document.getElementById('workspaceResultTableBox');
  const workspaceTableCountTag = document.getElementById('workspaceTableCountTag');
  const workspaceResultThead = document.getElementById('workspaceResultThead');
  const workspaceResultTbody = document.getElementById('workspaceResultTbody');
  const workspaceChartCard = document.getElementById('workspaceChartCard');
  const workspaceChartTitle = document.getElementById('workspaceChartTitle');
  const workspaceChartImg = document.getElementById('workspaceChartImg');
  const workspaceChartExpandBtn = document.getElementById('workspaceChartExpandBtn');
  const workspaceChartDownloadLink = document.getElementById('workspaceChartDownloadLink');

  // Chart Modal Elements
  const chartExpandModal = document.getElementById('chartExpandModal');
  const modalChartTitle = document.getElementById('modalChartTitle');
  const modalChartImg = document.getElementById('modalChartImg');
  const modalChartDownloadLink = document.getElementById('modalChartDownloadLink');
  const closeChartModalBtn = document.getElementById('closeChartModalBtn');
  const closeChartModalBottomBtn = document.getElementById('closeChartModalBottomBtn');

  // UI-1: Full Result Table Modal Elements
  const tableExpandModal = document.getElementById('tableExpandModal');
  const modalTableTitle = document.getElementById('modalTableTitle');
  const modalTableSubtitle = document.getElementById('modalTableSubtitle');
  const modalTableMeta = document.getElementById('modalTableMeta');
  const modalTableRowCount = document.getElementById('modalTableRowCount');
  const modalTableExportBtn = document.getElementById('modalTableExportBtn');
  const closeTableModalBtn = document.getElementById('closeTableModalBtn');
  const closeTableModalBottomBtn = document.getElementById('closeTableModalBottomBtn');
  const modalTableSearchInput = document.getElementById('modalTableSearchInput');
  const modalTableFilteredStatus = document.getElementById('modalTableFilteredStatus');
  const modalResultThead = document.getElementById('modalResultThead');
  const modalResultTbody = document.getElementById('modalResultTbody');

  // UI-1: Dataset Intelligence Conversational Architecture Elements
  const datasetIntelligenceCard = document.getElementById('datasetIntelligenceCard');
  const datasetChatScrollArea = document.getElementById('datasetChatScrollArea');
  const datasetChatEmptyState = document.getElementById('datasetChatEmptyState');
  const datasetChatTurns = document.getElementById('datasetChatTurns');
  const clearDatasetChatBtn = document.getElementById('clearDatasetChatBtn');
  const deleteDatasetChatBtn = document.getElementById('deleteDatasetChatBtn');
  const datasetComposerChips = document.getElementById('datasetComposerChips');
  const datasetChatForm = document.getElementById('datasetChatForm');
  const datasetChatInput = document.getElementById('datasetChatInput');
  const datasetSendBtn = document.getElementById('datasetSendBtn');
  const datasetSendBtnText = document.getElementById('datasetSendBtnText');
  const datasetSendSpinner = document.getElementById('datasetSendSpinner');

  // UI-3: Dataset Conversation History Elements
  const sidebarNewAnalysisBtn = document.getElementById('sidebarNewAnalysisBtn');
  const workspaceNewAnalysisBtn = document.getElementById('workspaceNewAnalysisBtn');
  const recentAnalysesList = document.getElementById('recentAnalysesList');
  const recentAnalysesContainer = document.getElementById('recentAnalysesContainer');

  let activeDataset = null;
  let activeDatasetTurns = [];
  let isDatasetLoading = false;
  let activeModalTableData = { columns: [], rows: [], title: '' };
  let activeAnalysisId = '';
  let analysesList = [];


  // Delete Confirmation Modal Elements
  const deleteConfirmModal = document.getElementById('deleteConfirmModal');
  const cancelDeleteBtn = document.getElementById('cancelDeleteBtn');
  const confirmDeleteBtn = document.getElementById('confirmDeleteBtn');

  // Toast Container
  const toastContainer = document.getElementById('toastContainer');

  // Settings Panel Elements
  const sidebarRight = document.getElementById('sidebarRight');
  const toggleSettingsBtn = document.getElementById('toggleSettingsBtn');
  const closeSettingsBtn = document.getElementById('closeSettingsBtn');
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
    updateDatasetAttachmentUI(thread.dataset || null);
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
        content.setAttribute('role', 'button');
        content.setAttribute('tabindex', '0');
        content.setAttribute('aria-label', `Open conversation: ${t.title}`);
        content.innerHTML = `
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
          </svg>
          <span class="recent-chat-title" title="${escapeHtml(t.title)}">${escapeHtml(t.title)}</span>
        `;
        content.addEventListener('click', () => switchToThread(t.id));
        content.addEventListener('keydown', (e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            switchToThread(t.id);
          }
        });

        // Action buttons (Archive, Delete)
        const actions = document.createElement('div');
        actions.className = 'item-action-btns';

        const archiveBtn = document.createElement('button');
        archiveBtn.className = 'item-micro-btn';
        archiveBtn.title = 'Archive conversation';
        archiveBtn.setAttribute('aria-label', `Archive conversation: ${t.title}`);
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
        delBtn.setAttribute('aria-label', `Delete conversation: ${t.title}`);
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
          content.setAttribute('role', 'button');
          content.setAttribute('tabindex', '0');
          content.setAttribute('aria-label', `Restore or open conversation: ${t.title}`);
          content.innerHTML = `
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="21 8 21 21 3 21 3 8"></polyline>
              <rect x="1" y="3" width="22" height="5"></rect>
            </svg>
            <span class="recent-chat-title" title="${escapeHtml(t.title)}">${escapeHtml(t.title)}</span>
          `;
          content.addEventListener('click', () => switchToThread(t.id));
          content.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              switchToThread(t.id);
            }
          });

          const actions = document.createElement('div');
          actions.className = 'item-action-btns';

          const restoreBtn = document.createElement('button');
          restoreBtn.className = 'item-micro-btn';
          restoreBtn.title = 'Restore to active';
          restoreBtn.setAttribute('aria-label', `Restore conversation: ${t.title}`);
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
          delBtn.setAttribute('aria-label', `Delete permanently: ${t.title}`);
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
      const uploadSessionId = (thread && thread.id) ? thread.id : currentSessionId;
      const formData = new FormData();
      formData.append('file', file);
      formData.append('session_id', uploadSessionId);

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
      if (data.session_token) {
        currentSessionToken = data.session_token;
        try {
          localStorage.setItem(STORAGE_KEY_SESSION_TOKEN, data.session_token);
        } catch (_) {}
      }
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
    } finally {
      if (attachDocBtn) attachDocBtn.disabled = false;
    }
  }

  async function handleRemoveAttachment() {
    const thread = getActiveThread();
    if (!thread || !thread.attachment) return;

    const docId = thread.attachment.document_id;
    if (attachmentMeta) attachmentMeta.textContent = 'Removing...';
    if (removeAttachmentBtn) removeAttachmentBtn.disabled = true;

    try {
      const targetSessionId = (thread && thread.id) ? thread.id : currentSessionId;
      const res = await fetch(`/documents/${encodeURIComponent(docId)}?session_id=${encodeURIComponent(targetSessionId)}`, {
        method: 'DELETE',
        headers: getAuthHeaders(),
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
  // Dataset Management & Workspace (CSV, XLSX, JSON) - Phase 5
  // ==========================================================================
  function updateDatasetAttachmentUI(dataset, isUploading = false, uploadingName = '') {
    if (!datasetChipBar) return;

    if (isUploading) {
      datasetChipBar.style.display = 'flex';
      if (datasetChip) datasetChip.className = 'attachment-chip dataset-chip uploading';
      if (datasetFilename) datasetFilename.textContent = uploadingName;
      if (datasetMeta) datasetMeta.textContent = 'Profiling dataset...';
      if (removeDatasetBtn) removeDatasetBtn.disabled = true;
      if (attachDatasetBtn) attachDatasetBtn.disabled = true;
      return;
    }

    if (attachDatasetBtn) attachDatasetBtn.disabled = false;

    if (dataset && dataset.dataset_id) {
      datasetChipBar.style.display = 'flex';
      if (datasetChip) datasetChip.className = 'attachment-chip dataset-chip';
      if (datasetFilename) {
        datasetFilename.textContent = dataset.filename || 'Dataset';
        datasetFilename.title = dataset.filename || 'Dataset';
      }
      if (datasetMeta) {
        const rows = dataset.row_count || 0;
        const cols = dataset.column_count || 0;
        datasetMeta.textContent = `${rows.toLocaleString()} rows • ${cols} cols`;
      }
      if (removeDatasetBtn) removeDatasetBtn.disabled = false;
    } else {
      datasetChipBar.style.display = 'none';
      if (datasetFilename) datasetFilename.textContent = '';
      if (datasetMeta) datasetMeta.textContent = '';
    }
  }

  async function handleDatasetUpload(file, isWorkspace = false) {
    if (!file) return;

    const allowedExts = ['.csv', '.xlsx', '.json'];
    const lowerName = file.name.toLowerCase();
    const isAllowed = allowedExts.some((ext) => lowerName.endsWith(ext));
    if (!isAllowed) {
      showToast('Unsupported dataset type. Please upload CSV, XLSX, or JSON.');
      return;
    }

    if (file.size > 50 * 1024 * 1024) {
      showToast('File exceeds maximum allowed dataset size of 50MB.');
      return;
    }

    if (file.size === 0) {
      showToast('Cannot upload an empty dataset file.');
      return;
    }

    let thread = getActiveThread();
    if (!thread && !isWorkspace) {
      thread = createNewThread();
    }

    let targetSessionId = currentSessionId;
    if (!targetSessionId) {
      if (isWorkspace && activeAnalysisId) {
        targetSessionId = activeAnalysisId;
      } else if (!isWorkspace && thread) {
        targetSessionId = thread.id;
      } else if (typeof crypto !== 'undefined' && crypto.randomUUID) {
        targetSessionId = crypto.randomUUID();
      } else {
        targetSessionId = 'sess_' + Date.now();
      }
      currentSessionId = targetSessionId;
    }

    if (isWorkspace) {
      if (workspaceActiveDatasetLabel) workspaceActiveDatasetLabel.textContent = `Profiling ${file.name}...`;
      if (workspaceUploadBtn) workspaceUploadBtn.disabled = true;
    } else {
      updateDatasetAttachmentUI(null, true, file.name);
    }

    try {
      const formData = new FormData();
      formData.append('file', file);
      if (targetSessionId) {
        formData.append('session_id', targetSessionId);
      }

      const res = await fetch('/datasets/upload', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        const msg = errData.detail || `Dataset upload failed (${res.status})`;
        showToast(msg);
        if (isWorkspace) {
          if (activeDataset) renderWorkspaceDataset(activeDataset);
          else showWorkspaceEmpty();
        } else {
          updateDatasetAttachmentUI(thread ? thread.dataset : null);
        }
        return;
      }

      const data = await res.json();
      if (data.session_token) {
        currentSessionToken = data.session_token;
        try {
          localStorage.setItem(STORAGE_KEY_SESSION_TOKEN, data.session_token);
        } catch (_) {}
      }
      if (!data.session_id && targetSessionId) {
        data.session_id = targetSessionId;
      }
      activeDataset = data;

      if (isWorkspace) {
        await createNewAnalysis(data);
      } else {
        thread.dataset = {
          dataset_id: data.dataset_id,
          filename: data.filename,
          row_count: data.row_count,
          column_count: data.column_count,
        };
        thread.updatedAt = Date.now();
        saveThreadsToStorage();

        updateDatasetAttachmentUI(thread.dataset);
        renderWorkspaceDataset(data);
      }
      showToast(`Uploaded ${data.filename} (${data.row_count.toLocaleString()} rows, ${data.column_count} columns)`);
    } catch (err) {
      console.error('Dataset upload error:', err);
      showToast('Network error uploading dataset.');
      if (isWorkspace) {
        if (activeDataset) renderWorkspaceDataset(activeDataset);
        else showWorkspaceEmpty();
      } else {
        updateDatasetAttachmentUI(thread ? thread.dataset : null);
      }
    } finally {
      if (workspaceUploadBtn) workspaceUploadBtn.disabled = false;
      if (sidebarDatasetUploadBtn) sidebarDatasetUploadBtn.disabled = false;
      if (workspaceUploadTriggerBtn) workspaceUploadTriggerBtn.disabled = false;
      if (attachDatasetBtn) attachDatasetBtn.disabled = false;
    }
  }

  function handleRemoveDataset() {
    const thread = getActiveThread();
    if (thread) {
      thread.dataset = null;
      thread.updatedAt = Date.now();
      saveThreadsToStorage();
    }
    updateDatasetAttachmentUI(null);
    showToast('Dataset detached from conversation.');
  }

  function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function formatCellValue(val) {
    if (val === null || val === undefined) return '';
    if (typeof val === 'number') {
      if (Number.isInteger(val)) return val.toLocaleString();
      return val.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    }
    return String(val);
  }

  function exportTableToCsv(columns, rows, filename = 'analysis_result.csv') {
    if (!columns || !rows || rows.length === 0) {
      showToast('No table data to export');
      return;
    }
    const lines = [];
    lines.push(columns.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(','));
    rows.forEach((row) => {
      const line = row
        .map((cell) => {
          if (cell === null || cell === undefined) return '""';
          if (typeof cell === 'number') return String(cell); // raw unformatted numeric value
          return `"${String(cell).replace(/"/g, '""')}"`;
        })
        .join(',');
      lines.push(line);
    });
    const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', filename.endsWith('.csv') ? filename : filename + '.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    showToast('Exported raw CSV');
  }

  function isNumericColumn(colIdx, allRows) {
    if (!allRows || allRows.length === 0) return false;
    let hasNonNull = false;
    for (let r = 0; r < allRows.length; r++) {
      const val = allRows[r][colIdx];
      if (val !== null && val !== undefined && val !== '') {
        hasNonNull = true;
        if (typeof val === 'number') continue;
        const num = Number(val);
        if (isNaN(num)) return false;
      }
    }
    return hasNonNull;
  }

  function openTableModal(columns, rows, title = 'Query Result Data') {
    if (!tableExpandModal) return;
    activeModalTriggerEl = document.activeElement;
    activeModalTableData = { columns, rows, title };

    if (modalTableTitle) modalTableTitle.textContent = 'View Results';
    if (modalTableSubtitle) modalTableSubtitle.textContent = title;
    if (modalTableMeta) {
      const rowCountText = `${rows.length} ${rows.length === 1 ? 'row' : 'rows'}`;
      const colCountText = `${columns.length} ${columns.length === 1 ? 'column' : 'columns'}`;
      modalTableMeta.textContent = `${rowCountText} · ${colCountText}`;
    }
    if (modalTableRowCount) modalTableRowCount.textContent = `${rows.length} rows`;
    if (modalTableSearchInput) modalTableSearchInput.value = '';
    if (modalTableFilteredStatus) {
      modalTableFilteredStatus.textContent = '';
      modalTableFilteredStatus.style.display = 'none';
    }

    renderModalTableRows(columns, rows, rows);
    tableExpandModal.style.display = 'flex';
    if (modalTableSearchInput) modalTableSearchInput.focus();
  }

  function renderModalTableRows(columns, rows, allRows = rows) {
    const numericFlags = columns.map((_, idx) => isNumericColumn(idx, allRows));

    if (modalResultThead) {
      modalResultThead.innerHTML = `<tr>${columns
        .map((c, idx) => `<th class="${numericFlags[idx] ? 'num-col' : ''}">${escapeHtml(formatCleanDisplayLabel(String(c)))}</th>`)
        .join('')}</tr>`;
    }
    if (modalResultTbody) {
      if (!rows || rows.length === 0) {
        modalResultTbody.innerHTML = `<tr><td colspan="${columns.length}" class="table-empty-notice">No matching rows found</td></tr>`;
      } else {
        modalResultTbody.innerHTML = rows
          .map(
            (r) =>
              `<tr>${r
                .map(
                  (v, idx) =>
                    `<td class="${numericFlags[idx] ? 'num-col' : ''}">${escapeHtml(formatCellValue(v))}</td>`
                )
                .join('')}</tr>`
          )
          .join('');
      }
    }
  }

  function filterModalTable() {
    if (!activeModalTableData || !activeModalTableData.rows) return;
    const term = (modalTableSearchInput ? modalTableSearchInput.value : '').toLowerCase().trim();
    if (!term) {
      renderModalTableRows(activeModalTableData.columns, activeModalTableData.rows, activeModalTableData.rows);
      if (modalTableFilteredStatus) {
        modalTableFilteredStatus.textContent = '';
        modalTableFilteredStatus.style.display = 'none';
      }
      return;
    }
    const filtered = activeModalTableData.rows.filter((row) =>
      row.some((cell) => cell !== null && cell !== undefined && String(cell).toLowerCase().includes(term))
    );
    renderModalTableRows(activeModalTableData.columns, filtered, activeModalTableData.rows);
    if (modalTableFilteredStatus) {
      modalTableFilteredStatus.textContent = `${filtered.length} of ${activeModalTableData.rows.length} rows`;
      modalTableFilteredStatus.style.display = 'inline-block';
    }
  }

  function closeTableModal() {
    if (!tableExpandModal) return;
    tableExpandModal.style.display = 'none';
    restoreModalFocus();
  }

  function formatCleanDisplayLabel(name) {
    if (!name) return 'Value';
    let raw = String(name).trim();
    let prefix = '';
    const lower = raw.toLowerCase();

    if (lower.startsWith('sum_')) {
      prefix = 'Total';
      raw = raw.substring(4).trim();
    } else if (lower.startsWith('avg_')) {
      prefix = 'Average';
      raw = raw.substring(4).trim();
    } else if (lower.startsWith('min_')) {
      prefix = 'Minimum';
      raw = raw.substring(4).trim();
    } else if (lower.startsWith('max_')) {
      prefix = 'Maximum';
      raw = raw.substring(4).trim();
    } else if (lower.startsWith('count_distinct_')) {
      prefix = 'Distinct';
      raw = raw.substring(15).trim();
    } else if (lower.startsWith('count_')) {
      prefix = 'Count of';
      raw = raw.substring(6).trim();
    }

    if (prefix) {
      const words = raw.replace(/_/g, ' ').split(/\s+/);
      if (words.length > 0 && words[0].toLowerCase() === prefix.toLowerCase()) {
        raw = words.join(' ');
      } else {
        raw = `${prefix} ${words.join(' ')}`;
      }
    } else {
      raw = raw.replace(/_/g, ' ');
    }

    return raw.split(/\s+/).map((p) => (p.toLowerCase() === p ? p.charAt(0).toUpperCase() + p.slice(1) : p)).join(' ');
  }

  function getBaseMetricColumnName(colName) {
    if (!colName) return 'Value';
    let s = String(colName).trim();
    s = s.replace(/^(?:sum_|avg_|min_|max_|count_distinct_|count_)/i, '');
    s = s.replace(/^(?:total|average|minimum|maximum|distinct|count\s+of)\s+/i, '');
    return s.replace(/_/g, ' ').trim();
  }

  function deriveAstBadgeText(query, data) {
    if (!data) return 'DATASET_ANALYSIS';

    // 1. Authoritative AST derivation from plan if provided by backend
    const plan = data.active_analysis_plan || data.analysis_plan;
    if (plan && typeof plan === 'object') {
      if (plan.pivot) {
        const agg = (plan.pivot.aggregation || 'SUM').toUpperCase();
        const val = getBaseMetricColumnName(plan.pivot.metric || 'Value');
        const row = plan.pivot.row_dimension || 'Dimension';
        const col = plan.pivot.column_dimension || 'Category';
        return `${agg}(${val}) • Rows: ${row} • Cols: ${col}`;
      }

      let badge = '';
      if (Array.isArray(plan.aggregations) && plan.aggregations.length > 0) {
        const firstAgg = plan.aggregations[0];
        const fn = (firstAgg.function || 'sum').toUpperCase();
        const colName = getBaseMetricColumnName(firstAgg.column || 'Value');
        badge = `${fn}(${colName})`;
      } else if (Array.isArray(plan.rolling_windows) && plan.rolling_windows.length > 0) {
        const rw = plan.rolling_windows[0];
        badge = `Rolling ${rw.function.toUpperCase()}(${rw.column})`;
      } else if (Array.isArray(plan.cumulative_metrics) && plan.cumulative_metrics.length > 0) {
        const cm = plan.cumulative_metrics[0];
        badge = `Cumulative ${cm.function.toUpperCase()}(${cm.column || ''})`;
      } else {
        badge = 'DATASET_ANALYSIS';
      }

      if (Array.isArray(plan.group_by) && plan.group_by.length > 0) {
        const dim = plan.group_by[0].column;
        if (dim) badge += ` • by ${dim.replace(/_/g, ' ')}`;
      }

      if (Array.isArray(plan.order_by) && plan.order_by.length > 0) {
        const ob = plan.order_by[0];
        badge += ob.descending ? ' • DESC' : ' • ASC';
      }

      if (plan.limit && plan.limit < 50) {
        badge += ` • Limit ${plan.limit}`;
      } else if (data.table && Array.isArray(data.table.rows) && (data.table.rows.length <= 10 && (plan.limit <= 50 || !plan.limit))) {
        badge += ` • Limit ${data.table.rows.length}`;
      }

      return badge;
    }

    if (data.pivot) {
      const agg = (data.pivot.aggregation || 'SUM').toUpperCase();
      const val = getBaseMetricColumnName(data.pivot.metric || data.pivot.value_column || 'Value');
      const row = data.pivot.row_dimension || 'Dimension';
      const col = data.pivot.column_dimension || 'Category';
      return `${agg}(${val}) • Rows: ${row} • Cols: ${col}`;
    }

    if (data.table && Array.isArray(data.table.columns) && data.table.columns.length > 0) {
      let contextQuery = (query || '').trim();
      const isMutationOnly = /^(?:bar\s*chart|line\s*chart|scatter|histogram|heatmap|chart|table|plot\s*this|visualize\s*this|show\s*this)\b/i.test(contextQuery);
      if (isMutationOnly && typeof activeDatasetTurns !== 'undefined' && Array.isArray(activeDatasetTurns)) {
        for (let i = activeDatasetTurns.length - 1; i >= 0; i--) {
          const t = activeDatasetTurns[i];
          if (t && t.userQuery && !/^(?:bar\s*chart|line\s*chart|chart|table|plot\s*this|visualize)/i.test(t.userQuery)) {
            contextQuery = t.userQuery;
            break;
          }
        }
      }

      const q = contextQuery.toLowerCase();
      const resp = (data.response || '').toLowerCase();
      const cols = data.table.columns;

      let metricCol = '';
      let dimCol = '';

      for (const col of cols) {
        const lower = col.toLowerCase();
        if (
          lower.startsWith('sum_') ||
          lower.startsWith('avg_') ||
          lower.startsWith('min_') ||
          lower.startsWith('max_') ||
          lower.startsWith('count_') ||
          lower.startsWith('total') ||
          lower.startsWith('average') ||
          lower.startsWith('minimum') ||
          lower.startsWith('maximum') ||
          lower.includes('sales') ||
          lower.includes('amount') ||
          lower.includes('price') ||
          lower.includes('quantity') ||
          lower.includes('revenue') ||
          lower.includes('profit')
        ) {
          if (!metricCol) metricCol = col;
        } else {
          if (!dimCol) dimCol = col;
        }
      }
      if (!dimCol && cols.length > 0) dimCol = cols[0];
      if (!metricCol && cols.length > 1) metricCol = cols[1];

      let agg = 'SUM';
      const rawMetric = metricCol || (cols.length > 1 ? cols[1] : cols[0]);
      const lowerRawMetric = rawMetric.toLowerCase();

      if (lowerRawMetric.startsWith('avg_') || lowerRawMetric.startsWith('average') || q.includes('average') || q.includes('avg')) {
        agg = 'AVG';
      } else if (lowerRawMetric.startsWith('count_') || lowerRawMetric.startsWith('count') || q.includes('count')) {
        agg = 'COUNT';
      } else if (lowerRawMetric.startsWith('max_') || lowerRawMetric.startsWith('maximum') || q.includes('maximum') || q.includes('max')) {
        agg = 'MAX';
      } else if (lowerRawMetric.startsWith('min_') || lowerRawMetric.startsWith('minimum') || q.includes('minimum') || q.includes('min')) {
        agg = 'MIN';
      }

      const cleanMetric = getBaseMetricColumnName(metricCol || 'Value');
      const cleanDim = dimCol ? dimCol.replace(/_/g, ' ') : '';

      let badge = cleanMetric ? `${agg}(${cleanMetric})` : agg;
      if (cleanDim) badge += ` • by ${cleanDim}`;

      const isAsc = q.includes('asc') || q.includes('ascending') || q.includes('lowest') || q.includes('least') || q.includes('bottom') || q.includes('worst');
      const isDesc = !isAsc && (q.includes('desc') || q.includes('descending') || q.includes('top') || q.includes('highest') || q.includes('most') || q.includes('best') || resp.includes('highest'));

      if (isAsc) badge += ' • ASC';
      else if (isDesc) badge += ' • DESC';

      const limitMatch = q.match(/\b(?:top|limit|first|bottom)\s*(\d+)\b/i);
      if (limitMatch && limitMatch[1]) {
        badge += ` • Limit ${limitMatch[1]}`;
      } else {
        const respLimitMatch = resp.match(/across\s+(\d+)\b/i);
        if (respLimitMatch && respLimitMatch[1]) {
          badge += ` • Limit ${respLimitMatch[1]}`;
        } else if (data.table.rows && (data.table.rows.length === 5 || (data.table.rows.length <= 10 && (isDesc || isAsc)))) {
          badge += ` • Limit ${data.table.rows.length}`;
        }
      }

      return badge;
    }

    return `${data.intent || 'DATASET_ANALYSIS'} • ${data.tool_used || 'query_dataset'}`;
  }

  function formatIsoTime(isoStr) {
    if (!isoStr) return formatTime();
    try {
      const d = new Date(isoStr);
      if (isNaN(d.getTime())) return formatTime();
      return d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
    } catch (_) {
      return formatTime();
    }
  }

  let recentAnalysesAbortController = null;
  let recentAnalysesRequestSeq = 0;
  const deletedAnalysisIds = new Set();

  async function loadRecentAnalyses() {
    if (!recentAnalysesList) return;

    if (recentAnalysesAbortController) {
      recentAnalysesAbortController.abort();
    }
    recentAnalysesAbortController = new AbortController();
    const reqSeq = ++recentAnalysesRequestSeq;

    try {
      const res = await fetch('/conversations?workspace_mode=dataset', {
        signal: recentAnalysesAbortController.signal,
      });
      if (!res.ok) return;
      const data = await res.json();
      if (reqSeq !== recentAnalysesRequestSeq) {
        // Discard stale out-of-order response
        return;
      }
      // Guarantee any conversation deleted during this session is never restored by stale responses
      analysesList = Array.isArray(data) ? data.filter((c) => !deletedAnalysisIds.has(c.id)) : [];
      renderRecentAnalysesList();
    } catch (e) {
      if (e.name !== 'AbortError') {
        console.warn('Failed to fetch recent analyses', e);
      }
    }
  }

  function renderRecentAnalysesList() {
    if (!recentAnalysesList) return;
    recentAnalysesList.innerHTML = '';
    if (!analysesList || analysesList.length === 0) {
      const emptyNote = document.createElement('div');
      emptyNote.style.padding = '8px 10px';
      emptyNote.style.fontSize = '0.74rem';
      emptyNote.style.color = 'var(--text-muted)';
      emptyNote.textContent = 'No previous analyses';
      recentAnalysesList.appendChild(emptyNote);
      return;
    }

    analysesList.forEach((conv) => {
      const item = document.createElement('div');
      item.className = `recent-analysis-item ${conv.id === activeAnalysisId ? 'active' : ''}`;

      const content = document.createElement('div');
      content.className = 'recent-chat-content';
      content.setAttribute('role', 'button');
      content.setAttribute('tabindex', '0');
      content.setAttribute('aria-label', `Open analysis: ${conv.title || 'Untitled'}`);

      const timeStr = conv.updated_at ? formatIsoTime(conv.updated_at) : '';
      const datasetTag = conv.dataset_id ? conv.dataset_id.slice(0, 6) : '';

      content.innerHTML = `
        <div class="recent-chat-text" style="flex: 1; overflow: hidden;">
          <span class="recent-chat-title" title="${escapeHtml(conv.title || 'Analysis')}">${escapeHtml(conv.title || 'Analysis')}</span>
          <span class="recent-chat-time">${escapeHtml(timeStr)}</span>
        </div>
        ${datasetTag ? `<span class="recent-analysis-badge" title="Bound Dataset: ${conv.dataset_id}">ds:${escapeHtml(datasetTag)}</span>` : ''}
      `;

      content.addEventListener('click', () => {
        switchAnalyticalConversation(conv.id);
      });
      content.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          switchAnalyticalConversation(conv.id);
        }
      });
      item.appendChild(content);

      const actions = document.createElement('div');
      actions.className = 'item-action-btns';

      const delBtn = document.createElement('button');
      delBtn.className = 'item-micro-btn danger';
      delBtn.title = 'Delete analysis';
      delBtn.setAttribute('aria-label', `Delete analysis: ${conv.title || 'Untitled'}`);
      delBtn.innerHTML = `
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="18" y1="6" x2="6" y2="18"></line>
          <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
      `;
      delBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        e.preventDefault();
        promptDeleteAnalysis(conv.id, conv.title);
      });
      actions.appendChild(delBtn);
      item.appendChild(actions);

      recentAnalysesList.appendChild(item);
    });
  }

  async function createNewAnalysis(targetDataset = null) {
    const ds = targetDataset || activeDataset;
    const dsId = ds ? ds.dataset_id : null;
    const title = ds ? `Analysis on ${ds.filename || 'Dataset'}` : 'New Analysis';

    try {
      const res = await fetch('/conversations', {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          title: title,
          workspace_mode: 'dataset',
          dataset_id: dsId,
          session_id: (ds && ds.session_id) || (activeDataset && activeDataset.session_id) || currentSessionId,
        }),
      });

      if (!res.ok) {
        showToast('Failed to create new analysis session');
        return;
      }

      const authHdr = res.headers.get('X-Session-Token');
      if (authHdr) {
        currentSessionToken = authHdr;
        try {
          localStorage.setItem(STORAGE_KEY_SESSION_TOKEN, authHdr);
        } catch (_) {}
      }

      const conv = await res.json();
      activeAnalysisId = conv.id;
      currentSessionId = conv.id;
      try {
        localStorage.setItem(STORAGE_KEY_ACTIVE_ANALYSIS, conv.id);
      } catch (_) {}

      if (datasetSessionUuidText) {
        datasetSessionUuidText.textContent = 'Active • Multi-turn';
        datasetSessionUuidText.title = `Analytical Session ID: ${conv.id}`;
      }

      activeDatasetTurns = [];
      renderDatasetConversation([]);

      if (ds) {
        renderWorkspaceDataset(ds);
      } else {
        showWorkspaceEmpty();
      }

      await loadRecentAnalyses();

      if (datasetChatInput) {
        datasetChatInput.value = '';
        datasetChatInput.focus();
      }
      showToast('Started new analysis');
    } catch (e) {
      console.error('Failed to create new analysis', e);
      showToast('Error starting new analysis');
    }
  }

  async function switchAnalyticalConversation(convId) {
    if (!convId) return;
    try {
      const res = await fetch(`/conversations/${convId}`, {
        headers: getAuthHeaders(),
      });
      if (!res.ok) {
        if (res.status === 404) {
          deletedAnalysisIds.add(convId);
          analysesList = analysesList.filter((c) => c.id !== convId);
          renderRecentAnalysesList();
          try {
            if (localStorage.getItem(STORAGE_KEY_ACTIVE_ANALYSIS) === convId) {
              localStorage.removeItem(STORAGE_KEY_ACTIVE_ANALYSIS);
            }
          } catch (_) {}
          if (activeAnalysisId === convId) {
            activeAnalysisId = '';
            currentSessionId = '';
            activeDatasetTurns = [];
            renderDatasetConversation([]);
            if (datasetSessionUuidText) {
              datasetSessionUuidText.textContent = 'Ready for Analysis';
              datasetSessionUuidText.title = 'No active analysis';
            }
          }
          if (analysesList.length > 0) {
            await switchAnalyticalConversation(analysesList[0].id);
          } else {
            if (datasetChatEmptyState) datasetChatEmptyState.style.display = 'flex';
            if (datasetChatTurns) datasetChatTurns.style.display = 'none';
          }
        }
        showToast('Could not load conversation');
        return;
      }
      const conv = await res.json();
      activeAnalysisId = conv.id;
      currentSessionId = conv.id;
      try {
        localStorage.setItem(STORAGE_KEY_ACTIVE_ANALYSIS, conv.id);
      } catch (_) {}

      if (datasetSessionUuidText) {
        datasetSessionUuidText.textContent = 'Active • Multi-turn';
        datasetSessionUuidText.title = `Analytical Session ID: ${conv.id}`;
      }

      // Restore dataset binding
      if (conv.dataset_id) {
        try {
          const dsRes = await fetch(`/datasets/${conv.dataset_id}?session_id=${conv.id}`, {
            headers: getAuthHeaders(),
          });
          if (dsRes.ok) {
            const dsData = await dsRes.json();
            renderWorkspaceDataset(dsData);
          } else {
            activeDataset = null;
            showWorkspaceEmpty();
            if (workspaceActiveDatasetLabel) {
              workspaceActiveDatasetLabel.textContent = 'No dataset selected (previous dataset unavailable)';
            }
          }
        } catch (_) {
          activeDataset = null;
          showWorkspaceEmpty();
        }
      } else {
        showWorkspaceEmpty();
      }

      // Restore messages into activeDatasetTurns
      activeDatasetTurns = [];
      if (conv.messages && Array.isArray(conv.messages)) {
        conv.messages.forEach((m) => {
          if (m.sender === 'user') {
            activeDatasetTurns.push({
              id: m.id,
              sender: 'user',
              query: m.user_message,
              timestamp: formatIsoTime(m.created_at),
            });
          } else if (m.sender === 'assistant') {
            activeDatasetTurns.push({
              id: m.id,
              sender: 'assistant',
              data: m.metadata || { response: m.assistant_response },
              userQuery: m.user_message || '',
              timestamp: formatIsoTime(m.created_at),
            });
          }
        });
      }

      renderDatasetConversation(activeDatasetTurns);
      renderRecentAnalysesList();
      if (datasetChatInput) datasetChatInput.focus();
    } catch (e) {
      console.error('Error switching analytical conversation', e);
      showToast('Failed to switch conversation');
    }
  }

  function promptDeleteAnalysis(convId, convTitle = 'Analysis') {
    pendingDeleteAnalysisId = convId;
    pendingDeleteThreadId = null;
    pendingDeleteDatasetId = null;
    pendingDeleteDatasetSessionId = null;
    activeModalTriggerEl = document.activeElement;
    if (deleteConfirmModal) {
      const titleEl = document.getElementById('deleteModalTitle');
      const msgEl = document.getElementById('deleteModalMessage');
      if (titleEl) titleEl.textContent = 'Delete Analysis';
      if (msgEl) msgEl.textContent = `Are you sure you want to permanently delete "${convTitle || 'this analysis'}"? This action cannot be undone.`;
      deleteConfirmModal.style.display = 'flex';
      if (cancelDeleteBtn) cancelDeleteBtn.focus();
    }
  }

  async function executeDeleteAnalysis(convId) {
    if (!convId) return;

    // Immediately abort any pending in-flight history request and advance request sequence
    if (recentAnalysesAbortController) {
      recentAnalysesAbortController.abort();
      recentAnalysesAbortController = null;
    }
    recentAnalysesRequestSeq++;

    let res;
    try {
      res = await fetch(`/conversations/${convId}`, { method: 'DELETE' });
    } catch (netErr) {
      console.error('Network error deleting analysis:', netErr);
      showToast('Network error: failed to delete analysis');
      return;
    }

    if (!res.ok && res.status !== 404) {
      let errMsg = 'Failed to delete analysis';
      try {
        const errJson = await res.json();
        if (errJson && errJson.detail) errMsg = errJson.detail;
      } catch (_) {}
      showToast(errMsg);
      return;
    }

    // Persistently mark as deleted in session filter
    deletedAnalysisIds.add(convId);
    analysesList = analysesList.filter((c) => c.id !== convId);
    renderRecentAnalysesList();

    try {
      if (localStorage.getItem(STORAGE_KEY_ACTIVE_ANALYSIS) === convId) {
        localStorage.removeItem(STORAGE_KEY_ACTIVE_ANALYSIS);
      }
    } catch (_) {}

    const wasActive = activeAnalysisId === convId;
    if (wasActive) {
      activeAnalysisId = '';
      currentSessionId = '';
      activeDatasetTurns = [];
      renderDatasetConversation([]);
      if (datasetSessionUuidText) {
        datasetSessionUuidText.textContent = 'Ready for Analysis';
        datasetSessionUuidText.title = 'No active analysis';
      }

      if (analysesList.length > 0) {
        await switchAnalyticalConversation(analysesList[0].id);
      } else {
        if (datasetChatEmptyState) datasetChatEmptyState.style.display = 'flex';
        if (datasetChatTurns) {
          datasetChatTurns.innerHTML = '';
          datasetChatTurns.style.display = 'none';
        }
        if (datasetChatInput) {
          datasetChatInput.value = '';
        }
      }
    }

    // Re-sync with backend to guarantee full persistence consistency
    await loadRecentAnalyses();
    showToast('Analysis permanently deleted');
  }

  function isDatasetChatNearBottom(threshold = 120) {
    if (!datasetChatScrollArea) return true;
    const scrollDistance = datasetChatScrollArea.scrollHeight - datasetChatScrollArea.scrollTop - datasetChatScrollArea.clientHeight;
    return scrollDistance <= threshold;
  }

  function scrollDatasetChatToBottom(force = false) {
    if (!datasetChatScrollArea) return;
    if (force || isDatasetChatNearBottom()) {
      datasetChatScrollArea.scrollTop = datasetChatScrollArea.scrollHeight;
    }
  }

  function appendDatasetTypingIndicator() {
    if (!datasetChatTurns) return;
    const existing = document.getElementById('activeDatasetTypingIndicator');
    if (existing) existing.remove();

    const wasNearBottom = isDatasetChatNearBottom();

    const turn = document.createElement('div');
    turn.className = 'dataset-turn dataset-turn-assistant';
    turn.id = 'activeDatasetTypingIndicator';

    const card = document.createElement('div');
    card.className = 'analytical-card';
    card.style.padding = '12px 16px';
    card.innerHTML = `
      <div style="display: flex; align-items: center; gap: 10px; color: var(--text-muted); font-size: 0.82rem;">
        <span class="spinner-small" style="display: inline-block;"></span>
        <span>Compiling deterministic AnalysisPlan &amp; executing DuckDB query...</span>
      </div>
    `;
    turn.appendChild(card);
    datasetChatTurns.appendChild(turn);
    if (wasNearBottom) {
      scrollDatasetChatToBottom(true);
    }
  }

  function removeDatasetTypingIndicator() {
    const el = document.getElementById('activeDatasetTypingIndicator');
    if (el) el.remove();
  }

  function appendDatasetUserTurnElement(turn) {
    if (!datasetChatTurns) return;
    if (datasetChatEmptyState) datasetChatEmptyState.style.display = 'none';
    datasetChatTurns.style.display = 'flex';

    const userTurnEl = document.createElement('div');
    userTurnEl.className = 'dataset-turn dataset-turn-user';
    if (turn.id) userTurnEl.id = turn.id;
    userTurnEl.innerHTML = `
      <div class="dataset-user-bubble">${escapeHtml(turn.query || '')}</div>
      <div class="dataset-turn-time">${escapeHtml(turn.timestamp || formatTime())}</div>
    `;
    datasetChatTurns.appendChild(userTurnEl);
    scrollDatasetChatToBottom(true);
  }

  function appendDatasetAssistantTurnElement(turn) {
    if (!datasetChatTurns) return;
    removeDatasetTypingIndicator();
    if (datasetChatEmptyState) datasetChatEmptyState.style.display = 'none';
    datasetChatTurns.style.display = 'flex';

    const wasNearBottom = isDatasetChatNearBottom();
    const asstTurnEl = renderDatasetAssistantTurnElement(turn);
    if (turn.id) asstTurnEl.id = turn.id;
    datasetChatTurns.appendChild(asstTurnEl);

    if (wasNearBottom) {
      scrollDatasetChatToBottom(true);
    }
  }

  function renderDatasetConversation(turns) {
    if (!datasetChatScrollArea) return;

    if (!turns || turns.length === 0) {
      if (datasetChatEmptyState) datasetChatEmptyState.style.display = 'flex';
      if (datasetChatTurns) {
        datasetChatTurns.style.display = 'none';
        datasetChatTurns.innerHTML = '';
      }
      return;
    }

    if (datasetChatEmptyState) datasetChatEmptyState.style.display = 'none';
    if (datasetChatTurns) {
      datasetChatTurns.style.display = 'flex';
      datasetChatTurns.innerHTML = '';

      turns.forEach((turn) => {
        if (turn.sender === 'user') {
          const userTurnEl = document.createElement('div');
          userTurnEl.className = 'dataset-turn dataset-turn-user';
          if (turn.id) userTurnEl.id = turn.id;
          userTurnEl.innerHTML = `
            <div class="dataset-user-bubble">${escapeHtml(turn.query || '')}</div>
            <div class="dataset-turn-time">${escapeHtml(turn.timestamp || formatTime())}</div>
          `;
          datasetChatTurns.appendChild(userTurnEl);
        } else if (turn.sender === 'assistant' && turn.data) {
          const asstTurnEl = renderDatasetAssistantTurnElement(turn);
          if (turn.id) asstTurnEl.id = turn.id;
          datasetChatTurns.appendChild(asstTurnEl);
        }
      });
    }

    scrollDatasetChatToBottom(true);
  }

  function renderDatasetAssistantTurnElement(turn) {
    const data = turn.data;
    const turnEl = document.createElement('div');
    turnEl.className = 'dataset-turn dataset-turn-assistant';

    // If status is error, show error card with retry button
    if (data.status === 'error' || data.intent === 'ERROR' || data.intent === 'NETWORK_ERROR') {
      const errCard = document.createElement('div');
      errCard.className = 'analytical-card analytical-error-card';
      const errMsg = data.response || 'An error occurred during dataset analysis.';
      errCard.innerHTML = `
        <div class="analytical-error-body">
          <div class="analytical-error-msg">⚠️ ${escapeHtml(errMsg)}</div>
          <button type="button" class="analytical-retry-btn" title="Retry this query">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="1 4 1 10 7 10"></polyline>
              <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"></path>
            </svg>
            <span>Retry</span>
          </button>
        </div>
      `;
      const retryBtn = errCard.querySelector('.analytical-retry-btn');
      if (retryBtn) {
        retryBtn.addEventListener('click', () => {
          if (turn.userQuery) runDatasetChat(turn.userQuery);
        });
      }
      turnEl.appendChild(errCard);
      return turnEl;
    }

    const card = document.createElement('div');
    card.className = 'analytical-card';

    // 1. Grounded Finding
    const findingBox = document.createElement('div');
    findingBox.className = 'analytical-finding';
    findingBox.innerHTML = `
      <span class="analytical-finding-icon">💡</span>
      <div class="analytical-finding-text">${escapeHtml(data.response || '')}</div>
    `;
    card.appendChild(findingBox);

    // 2. Inline Chart Card (when chart available)
    if (data.chart && data.chart.url) {
      const chartCard = document.createElement('div');
      chartCard.className = 'workspace-chart-card';
      chartCard.innerHTML = `
        <div class="chart-header-row">
          <div class="chart-title-wrap">
            <span class="chart-title-icon">📊</span>
            <span class="chart-title-text">${escapeHtml(data.chart.title || 'Generated Chart')}</span>
          </div>
          <div class="chart-actions">
            <button type="button" class="chart-action-btn expand-chart-btn" title="Fullscreen Chart">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="15 3 21 3 21 9"></polyline>
                <polyline points="9 21 3 21 3 15"></polyline>
                <line x1="21" y1="3" x2="14" y2="10"></line>
                <line x1="3" y1="21" x2="10" y2="14"></line>
              </svg>
              <span>Expand</span>
            </button>
            <a href="${data.chart.url}" download="${escapeHtml(data.chart.filename || 'chart.png')}" class="chart-action-btn" title="Download PNG">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="7 10 12 15 17 10"></polyline>
                <line x1="12" y1="15" x2="12" y2="3"></line>
              </svg>
              <span>Download</span>
            </a>
          </div>
        </div>
        <div class="chart-image-wrap">
          <img src="${data.chart.url}" alt="${escapeHtml(data.chart.title || 'Chart')}" class="chart-img-fluid" />
        </div>
      `;
      const expandBtn = chartCard.querySelector('.expand-chart-btn');
      if (expandBtn) {
        expandBtn.addEventListener('click', () => openChartModal(data.chart.url, data.chart.title || 'Chart View'));
      }
      card.appendChild(chartCard);
    }

    // 3. Compact Table Preview (when table available)
    if (data.table && Array.isArray(data.table.columns) && Array.isArray(data.table.rows) && data.table.rows.length > 0) {
      const totalRows = data.table.row_count || data.table.rows.length;
      const previewRows = data.table.rows.slice(0, 5);

      const tableBox = document.createElement('div');
      tableBox.className = 'table-preview-box';

      const displayCols = data.table.display_columns || data.table.columns;
      const tblHead = `<tr>${displayCols.map((c) => `<th>${escapeHtml(formatCleanDisplayLabel(String(c)))}</th>`).join('')}</tr>`;
      const tblBody = previewRows
        .map((r) => `<tr>${r.map((v) => `<td>${escapeHtml(formatCellValue(v))}</td>`).join('')}</tr>`)
        .join('');

      tableBox.innerHTML = `
        <div class="table-preview-header">
          <div class="table-preview-title">
            <span>📋</span>
            <span>Table Preview (Showing ${previewRows.length} of ${totalRows} rows)</span>
          </div>
          <span class="table-preview-count">${displayCols.length} columns</span>
        </div>
        <div class="table-scroll-wrap">
          <table class="analysis-result-table">
            <thead>${tblHead}</thead>
            <tbody>${tblBody}</tbody>
          </table>
        </div>
        <div class="table-preview-footer">
          <button type="button" class="btn-view-all">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
              <circle cx="12" cy="12" r="3"></circle>
            </svg>
            <span>View all ${totalRows} rows</span>
          </button>
          <button type="button" class="btn-export-csv" title="Export raw numerical values to CSV">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="7 10 12 15 17 10"></polyline>
              <line x1="12" y1="15" x2="12" y2="3"></line>
            </svg>
            <span>Export CSV</span>
          </button>
        </div>
      `;

      const viewAllBtn = tableBox.querySelector('.btn-view-all');
      if (viewAllBtn) {
        viewAllBtn.addEventListener('click', () => {
          openTableModal(displayCols.map(formatCleanDisplayLabel), data.table.rows, turn.userQuery || 'Query Result Data');
        });
      }

      const exportBtn = tableBox.querySelector('.btn-export-csv');
      if (exportBtn) {
        exportBtn.addEventListener('click', () => {
          exportTableToCsv(displayCols.map(formatCleanDisplayLabel), data.table.rows, 'analysis_result.csv');
        });
      }

      card.appendChild(tableBox);
    }

    // 4. AST / Query Badge & Timestamp
    const badgeText = deriveAstBadgeText(turn.userQuery, data);
    const badgeRow = document.createElement('div');
    badgeRow.className = 'ast-badge-container';
    badgeRow.innerHTML = `
      <span class="ast-badge" title="Deterministic DuckDB AST Analysis">
        <span class="ast-badge-icon">⚡</span>
        <span>${escapeHtml(badgeText)}</span>
      </span>
      <span class="dataset-turn-time">${escapeHtml(turn.timestamp || formatTime())}</span>
    `;
    card.appendChild(badgeRow);

    turnEl.appendChild(card);
    return turnEl;
  }

  async function runDatasetChat(queryText) {
    if (isDatasetLoading) return;

    const query = (queryText || (datasetChatInput ? datasetChatInput.value : '')).trim();
    if (!query) {
      showToast('Please enter an analytical question.');
      return;
    }
    if (!activeDataset) {
      showToast('No active dataset uploaded. Please upload a dataset first.');
      return;
    }

    // In-flight guard: Lock synchronously before any async calls to avoid duplicate dispatches
    isDatasetLoading = true;
    if (datasetChatInput) datasetChatInput.value = '';
    if (datasetSendBtn) datasetSendBtn.disabled = true;
    if (datasetSendSpinner) datasetSendSpinner.style.display = 'inline-block';
    if (datasetSendBtnText) datasetSendBtnText.style.display = 'none';
    if (datasetChatInput) datasetChatInput.disabled = true;

    try {
      if (!activeAnalysisId) {
        await createNewAnalysis();
      }
      if (!activeAnalysisId) {
        throw new Error('Unable to establish analytical session');
      }

      // Immediately append user turn without resetting existing DOM
      const userTime = formatTime();
      const userTurn = {
        id: 'dt-u-' + Date.now(),
        sender: 'user',
        query: query,
        timestamp: userTime,
      };
      activeDatasetTurns.push(userTurn);
      appendDatasetUserTurnElement(userTurn);

      appendDatasetTypingIndicator();

      const headers = getAuthHeaders({ 'Content-Type': 'application/json' });
      const apiKey = getApiKey();
      if (apiKey && apiKey.trim().length > 0) {
        headers['X-API-Key'] = apiKey.trim();
      }

      const payload = {
        message: query,
        session_id: activeAnalysisId,
        dataset_id: activeDataset.dataset_id,
        workspace_mode: WORKSPACE_MODE_DATASET,
      };

      const res = await fetch('/chat', {
        method: 'POST',
        headers: headers,
        body: JSON.stringify(payload),
      });

      const authHdr = res.headers.get('X-Session-Token');
      if (authHdr) {
        currentSessionToken = authHdr;
        try {
          localStorage.setItem(STORAGE_KEY_SESSION_TOKEN, authHdr);
        } catch (_) {}
      }

      removeDatasetTypingIndicator();

      let data;
      if (res.status === 401) {
        data = {
          response: 'Authentication Required (401): This structured data query requires an authorized API key.',
          intent: 'DATA_QUERY',
          tool_used: 'none',
          status: 'error',
        };
      } else if (!res.ok) {
        let detail = `HTTP ${res.status}`;
        try {
          const err = await res.json();
          if (err && err.detail) detail = typeof err.detail === 'string' ? err.detail : JSON.stringify(err.detail);
        } catch (_) {}
        data = {
          response: `Request failed: ${detail}`,
          intent: 'ERROR',
          tool_used: 'none',
          status: 'error',
        };
      } else {
        data = await res.json();
      }

      const assistantTime = formatTime();
      const assistantTurn = {
        id: 'dt-a-' + Date.now(),
        sender: 'assistant',
        data: data,
        userQuery: query,
        timestamp: assistantTime,
      };
      activeDatasetTurns.push(assistantTurn);
      appendDatasetAssistantTurnElement(assistantTurn);

      // Refresh recent analyses to update title & timestamp
      if (data.status !== 'error') {
        loadRecentAnalyses();
      }
    } catch (err) {
      removeDatasetTypingIndicator();
      const errTurn = {
        id: 'dt-err-' + Date.now(),
        sender: 'assistant',
        data: {
          response: `Network error: ${err.message || 'Unable to connect to analytics server'}`,
          intent: 'NETWORK_ERROR',
          tool_used: 'none',
          status: 'error',
        },
        userQuery: query,
        timestamp: formatTime(),
      };
      activeDatasetTurns.push(errTurn);
      appendDatasetAssistantTurnElement(errTurn);
    } finally {
      isDatasetLoading = false;
      if (datasetSendBtn) datasetSendBtn.disabled = false;
      if (datasetSendSpinner) datasetSendSpinner.style.display = 'none';
      if (datasetSendBtnText) datasetSendBtnText.style.display = 'inline-block';
      if (datasetChatInput) {
        datasetChatInput.disabled = false;
        datasetChatInput.focus();
      }
    }
  }

  async function clearDatasetChat() {
    if (!activeAnalysisId) return;
    try {
      await fetch(`/conversations/${activeAnalysisId}/clear`, { method: 'POST' });
    } catch (_) {}
    activeDatasetTurns = [];
    renderDatasetConversation([]);
    await loadRecentAnalyses();
    showToast('Dataset conversation cleared');
  }

  function runWorkspaceAnalysis(queryText) {
    return runDatasetChat(queryText);
  }

  function showWorkspaceEmpty() {
    activeDataset = null;
    activeDatasetTurns = [];
    if (workspaceEmptyState) workspaceEmptyState.style.display = 'flex';
    if (workspaceContentCard) workspaceContentCard.style.display = 'none';
    if (workspaceSwitchBtn) workspaceSwitchBtn.style.display = 'none';
    if (workspaceRemoveBtn) workspaceRemoveBtn.style.display = 'none';
    if (workspaceDeleteDatasetBtn) workspaceDeleteDatasetBtn.style.display = 'none';
    if (toggleSchemaTableBtn) toggleSchemaTableBtn.style.display = 'none';
    if (workspaceSchemaSection) workspaceSchemaSection.style.display = 'none';
    if (workspaceActiveDatasetLabel) workspaceActiveDatasetLabel.textContent = 'No dataset selected';
    const emptyUploadSpan = workspaceUploadBtn ? workspaceUploadBtn.querySelector('span') : null;
    if (emptyUploadSpan) emptyUploadSpan.textContent = 'Upload Dataset';

    // UI-2: Update Sidebar Dataset Context
    if (sidebarDatasetActiveCard) sidebarDatasetActiveCard.style.display = 'none';
    if (sidebarDatasetEmptyState) sidebarDatasetEmptyState.style.display = 'flex';

    renderDatasetConversation([]);
  }

  function renderWorkspaceDataset(data) {
    if (!data) return;
    activeDataset = data;

    if (workspaceActiveDatasetLabel) {
      workspaceActiveDatasetLabel.textContent = `${data.filename} (${data.dataset_id.slice(0, 8)}...)`;
    }
    const activeUploadSpan = workspaceUploadBtn ? workspaceUploadBtn.querySelector('span') : null;
    if (activeUploadSpan) activeUploadSpan.textContent = 'Upload New';
    if (workspaceEmptyState) workspaceEmptyState.style.display = 'none';
    if (workspaceContentCard) workspaceContentCard.style.display = 'flex';
    if (workspaceSwitchBtn) workspaceSwitchBtn.style.display = 'inline-flex';
    if (workspaceRemoveBtn) workspaceRemoveBtn.style.display = 'inline-flex';
    if (workspaceDeleteDatasetBtn) workspaceDeleteDatasetBtn.style.display = 'inline-flex';
    if (toggleSchemaTableBtn) toggleSchemaTableBtn.style.display = 'inline-flex';

    if (profileFormatTag) profileFormatTag.textContent = (data.format || 'csv').toUpperCase();
    if (profileRowsVal) profileRowsVal.textContent = (data.row_count || 0).toLocaleString();
    if (profileColsVal) profileColsVal.textContent = (data.column_count || 0).toLocaleString();

    const prof = data.profile || {};
    if (profileMissingVal) profileMissingVal.textContent = (prof.missing_cells || 0).toLocaleString();
    if (profileDupsVal) profileDupsVal.textContent = (prof.duplicate_rows || 0).toLocaleString();

    if (workspaceColumnsTbody && prof.columns) {
      workspaceColumnsTbody.innerHTML = '';
      prof.columns.forEach((col) => {
        const tr = document.createElement('tr');
        const missingPct = ((col.missing_percentage || 0) * 100).toFixed(1);
        tr.innerHTML = `
          <td><strong>${escapeHtml(col.name)}</strong></td>
          <td><span class="profile-format-tag">${escapeHtml(col.dtype)}</span></td>
          <td><code>${escapeHtml(col.dtype)}</code></td>
          <td>${col.missing_count || 0} (${missingPct}%)</td>
          <td>${(col.unique_count || 0).toLocaleString()}</td>
        `;
        workspaceColumnsTbody.appendChild(tr);
      });
    }

    // UI-2: Populate Sidebar Dataset Context Card & Columns List
    if (sidebarDatasetActiveCard) sidebarDatasetActiveCard.style.display = 'flex';
    if (sidebarDatasetEmptyState) sidebarDatasetEmptyState.style.display = 'none';
    if (sidebarDatasetFilename) {
      sidebarDatasetFilename.textContent = data.filename || 'dataset.csv';
      sidebarDatasetFilename.title = data.filename || 'dataset.csv';
    }
    if (sidebarDatasetFormatTag) {
      sidebarDatasetFormatTag.textContent = (data.format || 'csv').toUpperCase();
    }
    if (sidebarRowsVal) sidebarRowsVal.textContent = (data.row_count || 0).toLocaleString();
    if (sidebarColsVal) sidebarColsVal.textContent = (data.column_count || 0).toLocaleString();
    if (sidebarMissingVal) sidebarMissingVal.textContent = (prof.missing_cells || 0).toLocaleString();
    if (sidebarDupsVal) sidebarDupsVal.textContent = (prof.duplicate_rows || 0).toLocaleString();

    if (sidebarColsCount) {
      sidebarColsCount.textContent = prof.columns ? prof.columns.length : (data.column_count || 0);
    }

    if (sidebarColumnsList && prof.columns) {
      sidebarColumnsList.innerHTML = '';
      prof.columns.forEach((col) => {
        const item = document.createElement('button');
        item.type = 'button';
        item.className = 'sidebar-column-item';
        item.setAttribute('title', `Ask about ${col.name}`);
        item.setAttribute('aria-label', `Ask about column ${col.name}`);
        item.innerHTML = `
          <span class="s-col-name">${escapeHtml(col.name)}</span>
          <span class="s-col-dtype">${escapeHtml(col.dtype)}</span>
        `;
        item.addEventListener('click', () => {
          if (datasetChatInput) {
            datasetChatInput.value = `Show summary and distribution of ${col.name}`;
            datasetChatInput.focus();
            showToast(`Added query template for ${col.name}`);
          }
        });
        sidebarColumnsList.appendChild(item);
      });
    }

    renderDatasetConversation(activeDatasetTurns);
  }

  // UI-2: Unified Workspace Mode Switcher
  function switchWorkspaceMode(mode) {
    currentWorkspaceMode = mode === WORKSPACE_MODE_DATASET ? WORKSPACE_MODE_DATASET : WORKSPACE_MODE_SUPPORT;
    try {
      localStorage.setItem(STORAGE_KEY_WORKSPACE_MODE, currentWorkspaceMode);
    } catch (_) {}

    const isSupport = currentWorkspaceMode === WORKSPACE_MODE_SUPPORT;

    // 1. Header segmented tab buttons
    if (tabSupportModeBtn) {
      tabSupportModeBtn.classList.toggle('active', isSupport);
      tabSupportModeBtn.setAttribute('aria-selected', isSupport ? 'true' : 'false');
    }
    if (tabDatasetModeBtn) {
      tabDatasetModeBtn.classList.toggle('active', !isSupport);
      tabDatasetModeBtn.setAttribute('aria-selected', !isSupport ? 'true' : 'false');
    }

    // 2. Left sidebar panels
    if (supportSidebarPanel) supportSidebarPanel.style.display = isSupport ? 'flex' : 'none';
    if (datasetSidebarPanel) datasetSidebarPanel.style.display = isSupport ? 'none' : 'flex';

    // 3. Center workspace views
    if (chatMain) chatMain.style.display = isSupport ? 'flex' : 'none';
    if (datasetWorkspaceView) datasetWorkspaceView.style.display = isSupport ? 'none' : 'flex';

    // 4. Update Header New Chat / New Analysis button
    if (newChatBtn) {
      const labelSpan = newChatBtn.querySelector('span');
      if (labelSpan) {
        labelSpan.textContent = isSupport ? 'New Chat' : 'New Analysis';
      }
      newChatBtn.title = isSupport ? 'Start new support conversation' : 'Start fresh analytical query session';
    }

    // 5. If switching to dataset view, ensure dataset state or empty state is rendered
    if (!isSupport) {
      loadRecentAnalyses().then(() => {
        const savedAnalysisId = localStorage.getItem(STORAGE_KEY_ACTIVE_ANALYSIS);
        const validSaved = savedAnalysisId && analysesList.some((c) => c.id === savedAnalysisId);
        if (validSaved) {
          if (!activeAnalysisId || activeAnalysisId !== savedAnalysisId) {
            switchAnalyticalConversation(savedAnalysisId);
          }
        } else {
          try {
            localStorage.removeItem(STORAGE_KEY_ACTIVE_ANALYSIS);
          } catch (_) {}
          if (!activeAnalysisId && analysesList.length > 0) {
            switchAnalyticalConversation(analysesList[0].id);
          } else if (activeDataset) {
            renderWorkspaceDataset(activeDataset);
          } else {
            showWorkspaceEmpty();
          }
        }
      });
    } else {
      const activeThread = getActiveThread();
      if (activeThread && activeThread.id) {
        currentSessionId = activeThread.id;
      }
    }
  }

  function switchToWorkspaceView() {
    switchWorkspaceMode(WORKSPACE_MODE_DATASET);
  }

  function switchToChatView() {
    switchWorkspaceMode(WORKSPACE_MODE_SUPPORT);
  }

  function openChartModal(url, title = 'Chart View') {
    if (!chartExpandModal) return;
    activeModalTriggerEl = document.activeElement;
    if (modalChartTitle) modalChartTitle.textContent = title;
    if (modalChartImg) modalChartImg.src = url;
    if (modalChartDownloadLink) {
      modalChartDownloadLink.href = url;
      modalChartDownloadLink.download = 'chart.png';
    }
    chartExpandModal.style.display = 'flex';
    if (closeChartModalBtn) closeChartModalBtn.focus();
  }

  function closeChartModal() {
    if (!chartExpandModal) return;
    chartExpandModal.style.display = 'none';
    if (modalChartImg) modalChartImg.src = '';
    restoreModalFocus();
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
      thread.dataset = null;
      thread.title = 'New Conversation';
      thread.updatedAt = Date.now();
      saveThreadsToStorage();
    } else {
      createNewThread();
    }

    switchToThread(newSessionId);
    updateAttachmentUI(null);
    updateDatasetAttachmentUI(null);
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
    pendingDeleteAnalysisId = null;
    pendingDeleteDatasetId = null;
    pendingDeleteDatasetSessionId = null;
    activeModalTriggerEl = document.activeElement;
    if (deleteConfirmModal) {
      const titleEl = document.getElementById('deleteModalTitle');
      const msgEl = document.getElementById('deleteModalMessage');
      if (titleEl) titleEl.textContent = 'Delete Conversation';
      if (msgEl) msgEl.textContent = 'Are you sure you want to permanently delete this conversation? This action cannot be undone.';
      deleteConfirmModal.style.display = 'flex';
      if (cancelDeleteBtn) cancelDeleteBtn.focus();
    }
  }

  function promptDeleteCurrentDataset() {
    if (!activeDataset) {
      showToast('No active dataset to delete');
      return;
    }
    const targetSessionId = activeDataset.session_id || (activeAnalysisId || currentSessionId);
    promptDeleteDataset(activeDataset.dataset_id, activeDataset.filename, targetSessionId);
  }

  function promptDeleteDataset(datasetId, datasetName, sessionId) {
    pendingDeleteDatasetId = datasetId;
    pendingDeleteDatasetSessionId = sessionId || (activeDataset && activeDataset.dataset_id === datasetId ? activeDataset.session_id : null) || (activeAnalysisId || currentSessionId);
    pendingDeleteThreadId = null;
    pendingDeleteAnalysisId = null;
    activeModalTriggerEl = document.activeElement;
    if (deleteConfirmModal) {
      const titleEl = document.getElementById('deleteModalTitle');
      const msgEl = document.getElementById('deleteModalMessage');
      if (titleEl) titleEl.textContent = 'Delete Dataset Permanently';
      if (msgEl) msgEl.textContent = `Are you sure you want to permanently delete dataset "${datasetName || 'this dataset'}"? This will delete all stored dataset files and cannot be undone.`;
      deleteConfirmModal.style.display = 'flex';
      if (cancelDeleteBtn) cancelDeleteBtn.focus();
    }
  }

  async function executeDeleteDataset(datasetId, sessionId) {
    try {
      let resolvedSessionId = sessionId;
      if (!resolvedSessionId) {
        if (activeDataset && activeDataset.dataset_id === datasetId && activeDataset.session_id) {
          resolvedSessionId = activeDataset.session_id;
        } else if (activeAnalysisId) {
          resolvedSessionId = activeAnalysisId;
        } else {
          resolvedSessionId = currentSessionId;
        }
      }
      const queryParam = resolvedSessionId ? `?session_id=${encodeURIComponent(resolvedSessionId)}` : '';
      const res = await fetch(`/datasets/${encodeURIComponent(datasetId)}${queryParam}`, { method: 'DELETE' });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        showToast(err.detail || 'Failed to delete dataset');
        return;
      }
      showToast('Dataset permanently deleted');

      if (activeDataset && activeDataset.dataset_id === datasetId) {
        activeDataset = null;
        showWorkspaceEmpty();
      }

      if (activeAnalysisId) {
        const current = analysesList.find((c) => c.id === activeAnalysisId);
        if (current && current.dataset_id === datasetId) {
          current.dataset_id = null;
        }
      }

      if (switchDatasetModal && switchDatasetModal.style.display !== 'none') {
        openSwitchDatasetModal();
      }
    } catch (err) {
      showToast('Error deleting dataset: ' + err.message);
    }
  }

  async function handleRemoveDatasetFromWorkspace() {
    if (!activeDataset) {
      showToast('No active dataset to remove');
      return;
    }
    const removedName = activeDataset.filename || 'Dataset';
    activeDataset = null;

    if (activeAnalysisId) {
      try {
        await fetch(`/conversations/${activeAnalysisId}`, {
          method: 'PATCH',
          headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
          body: JSON.stringify({ dataset_id: null }),
        });
        const current = analysesList.find((c) => c.id === activeAnalysisId);
        if (current) current.dataset_id = null;
      } catch (err) {
        console.warn('Could not unbind dataset from conversation:', err);
      }
    }

    showWorkspaceEmpty();
    showToast(`Removed "${removedName}" from active workspace`);
  }

  async function openSwitchDatasetModal() {
    try {
      const res = await fetch('/datasets', {
        headers: getAuthHeaders(),
      });
      if (!res.ok) {
        triggerWorkspaceUpload();
        return;
      }
      const datasets = await res.json();
      if (!datasets || datasets.length === 0) {
        triggerWorkspaceUpload();
        return;
      }
      renderSwitchDatasetList(datasets);
      if (switchDatasetModal) switchDatasetModal.style.display = 'flex';
    } catch (_) {
      triggerWorkspaceUpload();
    }
  }

  function closeSwitchDatasetModal() {
    if (switchDatasetModal) switchDatasetModal.style.display = 'none';
  }

  function renderSwitchDatasetList(datasets) {
    if (!switchDatasetList) return;
    switchDatasetList.innerHTML = '';

    datasets.forEach((ds) => {
      const card = document.createElement('div');
      card.className = 'switch-dataset-card' + (activeDataset && activeDataset.dataset_id === ds.dataset_id ? ' active' : '');

      const info = document.createElement('div');
      info.className = 'switch-dataset-info';
      const name = document.createElement('div');
      name.className = 'switch-dataset-name';
      name.textContent = ds.filename || 'dataset.csv';
      const meta = document.createElement('div');
      meta.className = 'switch-dataset-meta';
      const rows = ds.row_count ? ds.row_count.toLocaleString() : '0';
      const cols = ds.column_count ? ds.column_count : '0';
      const format = (ds.format || 'CSV').toUpperCase();
      meta.textContent = `${format} • ${rows} rows • ${cols} columns`;

      info.appendChild(name);
      info.appendChild(meta);

      const actions = document.createElement('div');
      actions.className = 'switch-dataset-actions';

      const selectBtn = document.createElement('button');
      selectBtn.type = 'button';
      selectBtn.className = 'btn btn-secondary btn-sm';
      selectBtn.textContent = (activeDataset && activeDataset.dataset_id === ds.dataset_id) ? 'Active' : 'Select';
      if (activeDataset && activeDataset.dataset_id === ds.dataset_id) {
        selectBtn.disabled = true;
      } else {
        selectBtn.addEventListener('click', async () => {
          closeSwitchDatasetModal();
          renderWorkspaceDataset(ds);
          if (activeAnalysisId) {
            try {
              await fetch(`/conversations/${activeAnalysisId}`, {
                method: 'PATCH',
                headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
                body: JSON.stringify({
                  dataset_id: ds.dataset_id,
                  session_id: ds.session_id || currentSessionId,
                }),
              });
              const current = analysesList.find((c) => c.id === activeAnalysisId);
              if (current) current.dataset_id = ds.dataset_id;
            } catch (_) {}
          }
          showToast(`Switched active dataset to ${ds.filename}`);
        });
      }

      const delBtn = document.createElement('button');
      delBtn.type = 'button';
      delBtn.className = 'btn-micro danger';
      delBtn.title = 'Permanently delete this dataset';
      delBtn.setAttribute('aria-label', `Delete dataset ${ds.filename}`);
      delBtn.innerHTML = `
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="3 6 5 6 21 6"></polyline>
          <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
        </svg>
      `;
      delBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        promptDeleteDataset(ds.dataset_id, ds.filename, ds.session_id);
      });

      actions.appendChild(selectBtn);
      actions.appendChild(delBtn);

      card.appendChild(info);
      card.appendChild(actions);
      switchDatasetList.appendChild(card);
    });
  }

  async function confirmDelete() {
    if (pendingDeleteDatasetId) {
      const dsId = pendingDeleteDatasetId;
      const dsSessionId = pendingDeleteDatasetSessionId;
      pendingDeleteDatasetId = null;
      pendingDeleteDatasetSessionId = null;
      closeDeleteModal();
      await executeDeleteDataset(dsId, dsSessionId);
      return;
    }

    if (pendingDeleteAnalysisId) {
      const convId = pendingDeleteAnalysisId;
      closeDeleteModal();
      await executeDeleteAnalysis(convId);
      return;
    }

    if (pendingDeleteThreadId) {
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
  }

  function closeDeleteModal() {
    pendingDeleteThreadId = null;
    pendingDeleteAnalysisId = null;
    pendingDeleteDatasetId = null;
    pendingDeleteDatasetSessionId = null;
    if (deleteConfirmModal) {
      deleteConfirmModal.style.display = 'none';
    }
    restoreModalFocus();
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
    if (userProfileBtn) userProfileBtn.setAttribute('aria-expanded', !isVisible ? 'true' : 'false');
  }

  function closeProfileDropdown() {
    if (profileDropdown) {
      profileDropdown.style.display = 'none';
      if (userProfileBtn) userProfileBtn.setAttribute('aria-expanded', 'false');
    }
  }

  function toggleSettingsDrawer() {
    if (!sidebarRight) return;
    const isOpen = sidebarRight.classList.toggle('open');
    if (toggleSettingsBtn) toggleSettingsBtn.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
    if (isOpen) {
      if (apiKeyInput) {
        setTimeout(() => apiKeyInput.focus(), 100);
      }
    } else {
      if (toggleSettingsBtn) toggleSettingsBtn.focus();
    }
  }

  function closeSettingsDrawer(restoreFocus = false) {
    if (sidebarRight && sidebarRight.classList.contains('open')) {
      sidebarRight.classList.remove('open');
      if (toggleSettingsBtn) {
        toggleSettingsBtn.setAttribute('aria-expanded', 'false');
        if (restoreFocus) {
          try { toggleSettingsBtn.focus(); } catch (_) {}
        }
      }
    }
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

    // Render Table if present (Phase 5)
    if (data.table && data.table.columns && data.table.rows && data.table.rows.length > 0) {
      const tableWrap = document.createElement('div');
      tableWrap.className = 'table-scroll-wrap';
      tableWrap.style.marginTop = '12px';
      const tbl = document.createElement('table');
      tbl.className = 'analysis-result-table';
      tbl.innerHTML = `
        <thead><tr>${data.table.columns.map((c) => `<th>${escapeHtml(formatCleanDisplayLabel(String(c)))}</th>`).join('')}</tr></thead>
        <tbody>${data.table.rows.map((r) => `<tr>${r.map((v) => `<td>${v !== null && v !== undefined ? String(v) : ''}</td>`).join('')}</tr>`).join('')}</tbody>
      `;
      tableWrap.appendChild(tbl);
      bubble.appendChild(tableWrap);
    }

    // Render Chart if present (Phase 4 / 5)
    if (data.chart && data.chart.url) {
      const chartCard = document.createElement('div');
      chartCard.className = 'workspace-chart-card';
      chartCard.style.marginTop = '12px';
      chartCard.innerHTML = `
        <div class="chart-header-row">
          <div class="chart-title-wrap">
            <span class="chart-title-icon">📊</span>
            <span class="chart-title-text">${data.chart.title || 'Generated Chart'}</span>
          </div>
          <div class="chart-actions">
            <button type="button" class="chart-action-btn expand-btn" title="Expand Fullscreen">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="15 3 21 3 21 9"></polyline>
                <polyline points="9 21 3 21 3 15"></polyline>
                <line x1="21" y1="3" x2="14" y2="10"></line>
                <line x1="3" y1="21" x2="10" y2="14"></line>
              </svg>
              <span>Expand</span>
            </button>
            <a href="${data.chart.url}" download="${data.chart.filename || 'chart.png'}" class="chart-action-btn" title="Download PNG">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="7 10 12 15 17 10"></polyline>
                <line x1="12" y1="15" x2="12" y2="3"></line>
              </svg>
              <span>Download</span>
            </a>
          </div>
        </div>
        <div class="chart-image-wrap">
          <img src="${data.chart.url}" alt="${data.chart.title || 'Chart'}" class="chart-img-fluid" />
        </div>
      `;
      const expBtn = chartCard.querySelector('.expand-btn');
      if (expBtn) {
        expBtn.addEventListener('click', () => openChartModal(data.chart.url, data.chart.title));
      }
      bubble.appendChild(chartCard);
    }

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
      const headers = getAuthHeaders({ 'Content-Type': 'application/json' });
      const apiKey = getApiKey();
      if (apiKey && apiKey.trim().length > 0) {
        headers['X-API-Key'] = apiKey.trim();
      }

      const payload = {
        message: message,
        session_id: (thread && thread.id) ? thread.id : currentSessionId,
        workspace_mode: currentWorkspaceMode || WORKSPACE_MODE_SUPPORT,
      };

      if (thread && thread.attachment && thread.attachment.document_id) {
        payload.document_id = thread.attachment.document_id;
      }
      if (thread && thread.dataset && thread.dataset.dataset_id) {
        payload.dataset_id = thread.dataset.dataset_id;
      }

      const res = await fetch('/chat', {
        method: 'POST',
        headers: headers,
        body: JSON.stringify(payload),
      });

      const authHdr = res.headers.get('X-Session-Token');
      if (authHdr) {
        currentSessionToken = authHdr;
        try {
          localStorage.setItem(STORAGE_KEY_SESSION_TOKEN, authHdr);
        } catch (_) {}
      }

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

    // New Chat / New Analysis
    newChatBtn.addEventListener('click', async () => {
      if (currentWorkspaceMode === WORKSPACE_MODE_DATASET) {
        if (!activeDataset) {
          showToast('Please upload or select a dataset to begin analysis');
          if (workspaceUploadFileInput) workspaceUploadFileInput.click();
        } else {
          await createNewAnalysis();
        }
      } else {
        createNewThread();
        showToast('Started new conversation');
      }
    });

    if (sidebarNewAnalysisBtn) {
      sidebarNewAnalysisBtn.addEventListener('click', async () => {
        if (!activeDataset) {
          showToast('Please upload or select a dataset to begin analysis');
          if (workspaceUploadFileInput) workspaceUploadFileInput.click();
        } else {
          await createNewAnalysis();
        }
      });
    }

    if (workspaceNewAnalysisBtn) {
      workspaceNewAnalysisBtn.addEventListener('click', async () => {
        if (!activeDataset) {
          showToast('Please upload or select a dataset to begin analysis');
          if (workspaceUploadFileInput) workspaceUploadFileInput.click();
        } else {
          await createNewAnalysis();
        }
      });
    }

    // UI-2: Mode Selector Segmented Buttons
    if (tabSupportModeBtn) {
      tabSupportModeBtn.addEventListener('click', () => {
        switchWorkspaceMode(WORKSPACE_MODE_SUPPORT);
      });
    }
    if (tabDatasetModeBtn) {
      tabDatasetModeBtn.addEventListener('click', () => {
        switchWorkspaceMode(WORKSPACE_MODE_DATASET);
      });
    }

    // UI-2: Toggle Schema Table
    if (toggleSchemaTableBtn) {
      toggleSchemaTableBtn.addEventListener('click', () => {
        if (!workspaceSchemaSection) return;
        const isHidden = workspaceSchemaSection.style.display === 'none';
        workspaceSchemaSection.style.display = isHidden ? 'block' : 'none';
        toggleSchemaTableBtn.setAttribute('aria-expanded', isHidden ? 'true' : 'false');
        if (toggleSchemaTableBtnText) {
          toggleSchemaTableBtnText.textContent = isHidden ? 'Hide Schema' : 'Schema Table';
        }
      });
    }

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
      archivedToggleBtn.setAttribute('aria-expanded', !isVisible ? 'true' : 'false');
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
        if (action === 'datasets') {
          switchToWorkspaceView();
          return;
        } else {
          switchToChatView();
        }

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

    // Document attachment upload & remove controls
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

    // Dataset attachment in chat controls
    if (attachDatasetBtn && datasetFileInput) {
      attachDatasetBtn.addEventListener('click', () => {
        datasetFileInput.click();
      });
      datasetFileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
          handleDatasetUpload(e.target.files[0], false);
          datasetFileInput.value = '';
        }
      });
    }
    if (removeDatasetBtn) {
      removeDatasetBtn.addEventListener('click', handleRemoveDataset);
    }

    // Workspace upload triggers
    const triggerWorkspaceUpload = () => {
      if (workspaceUploadFileInput) workspaceUploadFileInput.click();
    };
    if (workspaceUploadBtn) workspaceUploadBtn.addEventListener('click', triggerWorkspaceUpload);
    if (workspaceUploadTriggerBtn) workspaceUploadTriggerBtn.addEventListener('click', triggerWorkspaceUpload);
    if (workspaceSwitchBtn) workspaceSwitchBtn.addEventListener('click', openSwitchDatasetModal);
    if (sidebarSwitchDatasetBtn) sidebarSwitchDatasetBtn.addEventListener('click', openSwitchDatasetModal);
    if (workspaceRemoveBtn) workspaceRemoveBtn.addEventListener('click', handleRemoveDatasetFromWorkspace);
    if (sidebarRemoveDatasetBtn) sidebarRemoveDatasetBtn.addEventListener('click', handleRemoveDatasetFromWorkspace);
    if (workspaceDeleteDatasetBtn) workspaceDeleteDatasetBtn.addEventListener('click', promptDeleteCurrentDataset);
    if (sidebarDatasetUploadBtn) sidebarDatasetUploadBtn.addEventListener('click', triggerWorkspaceUpload);
    if (closeSwitchDatasetModalBtn) closeSwitchDatasetModalBtn.addEventListener('click', closeSwitchDatasetModal);
    if (cancelSwitchDatasetBtn) cancelSwitchDatasetBtn.addEventListener('click', closeSwitchDatasetModal);
    if (switchUploadNewBtn) {
      switchUploadNewBtn.addEventListener('click', () => {
        closeSwitchDatasetModal();
        triggerWorkspaceUpload();
      });
    }
    if (switchDatasetModal) {
      switchDatasetModal.addEventListener('click', (e) => {
        if (e.target === switchDatasetModal) closeSwitchDatasetModal();
      });
    }
    if (workspaceUploadFileInput) {
      workspaceUploadFileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
          handleDatasetUpload(e.target.files[0], true);
          workspaceUploadFileInput.value = '';
        }
      });
    }

    // Workspace Query & Sample chips
    if (workspaceAnalyzeBtn) {
      workspaceAnalyzeBtn.addEventListener('click', () => {
        runWorkspaceAnalysis();
      });
    }
    if (workspaceQueryInput) {
      workspaceQueryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          runWorkspaceAnalysis();
        }
      });
    }
    document.querySelectorAll('.query-chip:not(.dataset-chip-action)').forEach((chip) => {
      chip.addEventListener('click', () => {
        const q = chip.getAttribute('data-query');
        if (q) {
          if (workspaceQueryInput) workspaceQueryInput.value = q;
          if (datasetChatInput) datasetChatInput.value = q;
          runWorkspaceAnalysis(q);
        }
      });
    });

    // UI-1: Dataset Intelligence Conversational Architecture Listeners
    if (datasetChatForm) {
      datasetChatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        runDatasetChat();
      });
    }
    if (datasetChatInput) {
      datasetChatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          runDatasetChat();
        }
      });
    }
    if (clearDatasetChatBtn) {
      clearDatasetChatBtn.addEventListener('click', clearDatasetChat);
    }
    if (deleteDatasetChatBtn) {
      deleteDatasetChatBtn.addEventListener('click', () => {
        if (!activeAnalysisId) {
          showToast('No active analysis to delete');
          return;
        }
        const current = analysesList.find((c) => c.id === activeAnalysisId);
        promptDeleteAnalysis(activeAnalysisId, current ? current.title : 'Active Analysis');
      });
    }
    document.querySelectorAll('.dataset-chip-action').forEach((chip) => {
      chip.addEventListener('click', (e) => {
        e.preventDefault();
        const q = chip.getAttribute('data-query');
        if (q) {
          if (datasetChatInput) datasetChatInput.value = q;
          runDatasetChat(q);
        }
      });
    });

    // UI-1: Full Result Table Modal Listeners
    if (modalTableSearchInput) {
      modalTableSearchInput.addEventListener('input', filterModalTable);
    }
    if (modalTableExportBtn) {
      modalTableExportBtn.addEventListener('click', () => {
        if (activeModalTableData && activeModalTableData.columns) {
          exportTableToCsv(activeModalTableData.columns, activeModalTableData.rows, 'dataset_results.csv');
        }
      });
    }
    if (closeTableModalBtn) closeTableModalBtn.addEventListener('click', closeTableModal);
    if (closeTableModalBottomBtn) closeTableModalBottomBtn.addEventListener('click', closeTableModal);
    if (tableExpandModal) {
      tableExpandModal.addEventListener('click', (e) => {
        if (e.target === tableExpandModal) closeTableModal();
      });
    }

    // Chart Fullscreen Modal
    if (workspaceChartExpandBtn) {
      workspaceChartExpandBtn.addEventListener('click', () => {
        if (workspaceChartImg && workspaceChartImg.src) {
          openChartModal(workspaceChartImg.src, workspaceChartTitle ? workspaceChartTitle.textContent : 'Chart View');
        }
      });
    }
    if (closeChartModalBtn) closeChartModalBtn.addEventListener('click', closeChartModal);
    if (closeChartModalBottomBtn) closeChartModalBottomBtn.addEventListener('click', closeChartModal);
    if (chartExpandModal) {
      chartExpandModal.addEventListener('click', (e) => {
        if (e.target === chartExpandModal) closeChartModal();
      });
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
      toggleSettingsBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        toggleSettingsDrawer();
      });
    }

    if (closeSettingsBtn) {
      closeSettingsBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        closeSettingsDrawer(true);
      });
    }

    // Outside click closes Settings drawer
    document.addEventListener('click', (e) => {
      if (
        sidebarRight &&
        sidebarRight.classList.contains('open') &&
        !sidebarRight.contains(e.target) &&
        toggleSettingsBtn &&
        !toggleSettingsBtn.contains(e.target)
      ) {
        closeSettingsDrawer(false);
      }
    });

    // Escape key closes modals and transient drawers
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        closeDeleteModal();
        closeProfileDropdown();
        closeChartModal();
        closeTableModal();
        closeSettingsDrawer(true);
      }
    });
  }

  // ==========================================================================
  // Desktop Sidebar Resizer (Splitter / Drag Handle)
  // ==========================================================================
  const DEFAULT_SIDEBAR_WIDTH = 260;
  const MIN_SIDEBAR_WIDTH = 200;
  const MAX_SIDEBAR_WIDTH = 520;

  function initSidebarResizer() {
    const sidebar = document.getElementById('sidebarLeft') || document.querySelector('.sidebar-left');
    const resizer = document.getElementById('sidebarResizer');
    if (!sidebar || !resizer) return;

    function applySavedWidth() {
      if (window.innerWidth <= 768) {
        sidebar.style.removeProperty('width');
        return;
      }
      const savedWidth = localStorage.getItem(STORAGE_KEY_SIDEBAR_WIDTH);
      if (savedWidth) {
        const parsed = parseInt(savedWidth, 10);
        if (!isNaN(parsed) && parsed >= MIN_SIDEBAR_WIDTH && parsed <= MAX_SIDEBAR_WIDTH) {
          sidebar.style.width = parsed + 'px';
          document.documentElement.style.setProperty('--sidebar-width', parsed + 'px');
          return;
        }
      }
      sidebar.style.width = DEFAULT_SIDEBAR_WIDTH + 'px';
      document.documentElement.style.setProperty('--sidebar-width', DEFAULT_SIDEBAR_WIDTH + 'px');
    }

    applySavedWidth();

    let isResizing = false;
    let startX = 0;
    let startWidth = 0;

    function onMouseDown(e) {
      if (window.innerWidth <= 768) return;
      if (e.button !== 0) return;

      e.preventDefault();
      isResizing = true;
      startX = e.clientX;
      startWidth = sidebar.getBoundingClientRect().width;

      resizer.classList.add('is-dragging');
      document.body.classList.add('sidebar-resizing');

      document.addEventListener('mousemove', onMouseMove, { passive: false });
      document.addEventListener('mouseup', onMouseUp);
    }

    function onMouseMove(e) {
      if (!isResizing) return;
      e.preventDefault();

      const deltaX = e.clientX - startX;
      let newWidth = startWidth + deltaX;

      const dynamicMax = Math.min(MAX_SIDEBAR_WIDTH, Math.floor(window.innerWidth * 0.5));
      newWidth = Math.max(MIN_SIDEBAR_WIDTH, Math.min(newWidth, dynamicMax));

      sidebar.style.width = newWidth + 'px';
      document.documentElement.style.setProperty('--sidebar-width', newWidth + 'px');
    }

    function onMouseUp(e) {
      if (!isResizing) return;
      isResizing = false;

      resizer.classList.remove('is-dragging');
      document.body.classList.remove('sidebar-resizing');

      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);

      const finalWidth = Math.round(sidebar.getBoundingClientRect().width);
      if (finalWidth >= MIN_SIDEBAR_WIDTH && finalWidth <= MAX_SIDEBAR_WIDTH) {
        localStorage.setItem(STORAGE_KEY_SIDEBAR_WIDTH, finalWidth.toString());
      }
    }

    resizer.addEventListener('dblclick', (e) => {
      if (window.innerWidth <= 768) return;
      e.preventDefault();
      sidebar.style.width = DEFAULT_SIDEBAR_WIDTH + 'px';
      document.documentElement.style.setProperty('--sidebar-width', DEFAULT_SIDEBAR_WIDTH + 'px');
      localStorage.setItem(STORAGE_KEY_SIDEBAR_WIDTH, DEFAULT_SIDEBAR_WIDTH.toString());
    });

    resizer.addEventListener('mousedown', onMouseDown);

    window.addEventListener('resize', () => {
      if (window.innerWidth <= 768) {
        sidebar.style.removeProperty('width');
      } else {
        applySavedWidth();
      }
    });
  }

  // ==========================================================================
  // App Bootstrapping
  // ==========================================================================
  function init() {
    initTheme();
    initApiKey();
    initSidebarResizer();
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

    const urlParams = new URLSearchParams(window.location.search);
    const urlMode = urlParams.get('mode');
    const savedMode = (urlMode === WORKSPACE_MODE_DATASET || urlMode === WORKSPACE_MODE_SUPPORT)
      ? urlMode
      : (localStorage.getItem(STORAGE_KEY_WORKSPACE_MODE) || WORKSPACE_MODE_SUPPORT);
    switchWorkspaceMode(savedMode);

    if (savedMode === WORKSPACE_MODE_SUPPORT) {
      loadRecentAnalyses();
    }

    initEventListeners();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
