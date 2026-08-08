/**
 * app.js – Zenvyrolabs Multimodal RAG Frontend
 *
 * Bug 3 Fix (Amnesia):
 *   - A `chatHistory` array per mode stores all turns as { role, content }.
 *   - Every /api/chat request includes the full history so the backend LLM
 *     has conversation context (resolved in rag_engine.py prompt).
 *
 * Additional improvements:
 *   - Auto-resize textarea
 *   - Keyboard shortcut (Enter to send, Shift+Enter for newline)
 *   - Mode-isolated history so switching tabs doesn't bleed context
 *   - Error display inline instead of alert()
 */

// ── DOM References ──────────────────────────────────────────────────────────
const fileInput     = document.getElementById('pdf-upload');
const uploadStatus  = document.getElementById('upload-status');
const uploadSuccess = document.getElementById('upload-success');
const successText   = document.getElementById('success-text');
const chatInput     = document.getElementById('chat-input');
const sendBtn       = document.getElementById('send-btn');
const chatBox       = document.getElementById('chat-box');

const API_BASE = window.location.origin;

// ── Mode & History State ────────────────────────────────────────────────────
let currentMode = 'coding';

/**
 * BUG 3 FIX: Each mode maintains its own independent chat history array.
 * History format: [{ role: 'user'|'assistant', content: '...' }, ...]
 */
const modeStates = {
  coding: { chatHtml: getDefaultWelcome(), successMsg: '', history: [] },
  novel:  { chatHtml: getDefaultWelcome(), successMsg: '', history: [] },
  manga:  { chatHtml: getDefaultWelcome(), successMsg: '', history: [] },
};

function getDefaultWelcome() {
  return `
    <div class="message system-msg">
      <div class="avatar"><i class="fa-solid fa-robot"></i></div>
      <div class="msg-bubble">
        <p>Welcome! Please upload a PDF document on the left, then ask me anything about it.</p>
      </div>
    </div>`;
}

// ── Mode Switching ──────────────────────────────────────────────────────────
window.setMode = function(mode) {
  // Save current state
  modeStates[currentMode].chatHtml   = chatBox.innerHTML;
  modeStates[currentMode].successMsg = uploadSuccess.classList.contains('hidden')
    ? '' : successText.innerText;

  // Apply new mode
  currentMode = mode;
  document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelector(`.mode-btn[data-mode="${mode}"]`).classList.add('active');

  // Restore saved state for new mode
  chatBox.innerHTML = modeStates[currentMode].chatHtml;

  if (modeStates[currentMode].successMsg) {
    successText.innerText = modeStates[currentMode].successMsg;
    uploadSuccess.classList.remove('hidden');
  } else {
    uploadSuccess.classList.add('hidden');
  }

  chatBox.scrollTop = chatBox.scrollHeight;
};

// ── File Upload ─────────────────────────────────────────────────────────────
fileInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  uploadStatus.classList.remove('hidden');
  uploadSuccess.classList.add('hidden');

  const formData = new FormData();
  formData.append('file', file);
  formData.append('book_type', currentMode);

  try {
    const response = await fetch(`${API_BASE}/api/upload`, {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();
    uploadStatus.classList.add('hidden');

    if (data.detail || data.error) {
      addMessage(`❌ Upload failed: ${data.detail || data.error}`);
    } else {
      successText.innerText = data.message || `Document embedded as [${currentMode}]!`;
      uploadSuccess.classList.remove('hidden');
    }
  } catch (err) {
    uploadStatus.classList.add('hidden');
    addMessage('❌ Failed to connect to the server. Is the backend running?');
  }

  // Reset so re-uploading the same file fires the event again
  fileInput.value = '';
});

// ── Markdown / Code Rendering ───────────────────────────────────────────────
marked.setOptions({
  highlight: function(code, lang) {
    const language = hljs.getLanguage(lang) ? lang : 'plaintext';
    return hljs.highlight(code, { language }).value;
  },
});

const renderer = new marked.Renderer();
renderer.code = function(code_or_token, language) {
  const code = typeof code_or_token === 'string' ? code_or_token : code_or_token.text;
  const lang = typeof code_or_token === 'string' ? language : code_or_token.lang;

  let validLang = 'plaintext';
  let highlighted = escapeHTML(code);

  try {
    if (lang && hljs.getLanguage(lang)) {
      validLang = lang;
      highlighted = hljs.highlight(code, { language: validLang }).value;
    } else {
      highlighted = hljs.highlightAuto(code).value;
    }
  } catch (_) { /* keep escaped fallback */ }

  const codeId = 'code-' + Math.random().toString(36).substr(2, 9);

  return `
    <div class="code-container">
      <div class="code-header">
        <span class="lang-label">${validLang}</span>
        <button class="copy-btn" onclick="copyCode('${codeId}')">
          <i class="fa-regular fa-copy"></i> Copy
        </button>
      </div>
      <div id="${codeId}-raw" class="hidden">${escapeHTML(code)}</div>
      <pre><code class="hljs language-${validLang}">${highlighted}</code></pre>
    </div>`;
};
marked.use({ renderer });

function escapeHTML(str) {
  return str.replace(/[&<>'"]/g,
    tag => ({ '&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;' }[tag] || tag)
  );
}

window.copyCode = function(codeId) {
  const raw = document.getElementById(codeId + '-raw').innerText;
  navigator.clipboard.writeText(raw).then(() => {
    const btn = document.querySelector(`[onclick="copyCode('${codeId}')"]`);
    const orig = btn.innerHTML;
    btn.innerHTML = '<i class="fa-solid fa-check"></i> Copied!';
    setTimeout(() => { btn.innerHTML = orig; }, 2000);
  });
};

// ── Message Rendering ───────────────────────────────────────────────────────
function addMessage(text, isUser = false) {
  const msgDiv = document.createElement('div');
  msgDiv.className = `message ${isUser ? 'user-msg' : 'system-msg'}`;

  const icon = isUser ? 'fa-user' : 'fa-robot';
  const htmlContent = isUser
    ? escapeHTML(text).replace(/\n/g, '<br>')
    : marked.parse(text);

  msgDiv.innerHTML = `
    <div class="avatar"><i class="fa-solid ${icon}"></i></div>
    <div class="msg-bubble">${htmlContent}</div>`;

  chatBox.appendChild(msgDiv);
  chatBox.scrollTop = chatBox.scrollHeight;
  return msgDiv;
}

function addLoadingBubble() {
  const div = document.createElement('div');
  div.className = 'message system-msg';
  div.id = 'loading-bubble';
  div.innerHTML = `
    <div class="avatar"><i class="fa-solid fa-robot"></i></div>
    <div class="msg-bubble">
      <div class="loader" style="width:15px;height:15px;border-width:2px;"></div>
    </div>`;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function removeLoadingBubble() {
  const el = document.getElementById('loading-bubble');
  if (el) el.remove();
}

// ── Send Message ────────────────────────────────────────────────────────────
async function sendMessage() {
  const text = chatInput.value.trim();
  if (!text) return;

  addMessage(text, true);
  chatInput.value = '';
  autoResizeTextarea();
  addLoadingBubble();
  sendBtn.disabled = true;

  // ── Bug 3 Fix: include full history for this mode ─────────────────────
  const history = [...modeStates[currentMode].history];

  try {
    const response = await fetch(`${API_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: text,
        book_type: currentMode,
        chat_history: history,   // ← full conversation history sent to backend
      }),
    });

    const data = await response.json();
    removeLoadingBubble();

    const answer = data.answer || data.detail || '❌ Unexpected response from server.';
    addMessage(answer);

    // ── Bug 3 Fix: append both turns to the mode's history ──────────────
    modeStates[currentMode].history.push({ role: 'user',      content: text   });
    modeStates[currentMode].history.push({ role: 'assistant', content: answer });

  } catch (err) {
    removeLoadingBubble();
    addMessage('❌ Failed to connect to the server. Make sure the backend is running.');
  }

  sendBtn.disabled = false;
  chatInput.focus();
}

sendBtn.addEventListener('click', sendMessage);

chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// ── Auto-resize textarea ────────────────────────────────────────────────────
function autoResizeTextarea() {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 160) + 'px';
}
chatInput.addEventListener('input', autoResizeTextarea);
