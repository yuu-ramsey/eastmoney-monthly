// popup 设置页 + 成本面板 + 分析历史

const $ = (id) => document.getElementById(id);

// provider 配置（与 lib/llm 一致，popup 不 import 模块，手写一份）
const PROVIDER_DEFS = {
  anthropic: {
    displayName: 'Anthropic Claude',
    defaultModel: 'claude-sonnet-4-6',
    apiKeyPattern: 'sk-ant-...',
    apiKeyHelp: '从 console.anthropic.com/settings/keys 获取（需海外支付）',
    docUrl: 'https://docs.claude.com/en/docs/about-claude/models/overview',
    knownModels: ['claude-opus-4-7', 'claude-sonnet-4-6', 'claude-haiku-4-5', 'claude-haiku-4-5-20251001'],
  },
  deepseek: {
    displayName: 'DeepSeek',
    defaultModel: 'deepseek-chat',
    apiKeyPattern: 'sk-...',
    apiKeyHelp: '从 platform.deepseek.com/api_keys 获取（支持支付宝/微信充值）',
    docUrl: 'https://api-docs.deepseek.com/',
    knownModels: ['deepseek-chat', 'deepseek-reasoner'],
  },
};

const DEFAULT_LIMIT = 60;

// ---------- Tab 切换 ----------

document.querySelectorAll('.tab-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach((c) => c.classList.remove('active'));
    btn.classList.add('active');
    const tabId = 'tab-' + btn.dataset.tab;
    const content = document.getElementById(tabId);
    if (content) content.classList.add('active');
    if (btn.dataset.tab === 'history') loadHistoryTab();
  });
});

// ---------- provider 切换后的 UI 刷新 ----------

function applyProviderUI(providerId) {
  const def = PROVIDER_DEFS[providerId] || PROVIDER_DEFS.anthropic;
  $('apiKey').placeholder = def.apiKeyPattern;
  $('apiKeyHint').innerHTML = `${def.apiKeyHelp} <a href="${def.docUrl}" target="_blank">文档参考</a>`;
  $('model').placeholder = def.defaultModel;
}

// ---------- 成本面板 ----------

function formatCost(cost) {
  if (cost < 0.01) return '<￥0.01';
  if (cost < 1) return `￥${cost.toFixed(2)}`;
  return `￥${cost.toFixed(2)}`;
}

function formatMonthKey(ym) {
  const [y, m] = ym.split('-');
  return `${y}年${parseInt(m, 10)}月`;
}

let costPanelProvider = 'anthropic';
let costPanelMonth = new Date().toISOString().slice(0, 7);

function currentMonthKey() {
  return new Date().toISOString().slice(0, 7);
}

function prevMonthKey(ym) {
  const d = new Date(ym + '-01');
  d.setMonth(d.getMonth() - 1);
  return d.toISOString().slice(0, 7);
}

function nextMonthKey(ym) {
  const d = new Date(ym + '-01');
  d.setMonth(d.getMonth() + 1);
  return d.toISOString().slice(0, 7);
}

async function loadCostPanel(providerId, monthKey) {
  costPanelProvider = providerId;
  costPanelMonth = monthKey || currentMonthKey();

  const def = PROVIDER_DEFS[providerId] || PROVIDER_DEFS.anthropic;
  const items = await chrome.storage.local.get([`usage:${providerId}:${costPanelMonth}`]);
  const record = items[`usage:${providerId}:${costPanelMonth}`];
  const isCurrentMonth = costPanelMonth === currentMonthKey();

  $('costLabel').textContent = `${def.displayName} ${formatMonthKey(costPanelMonth)}`;

  const hasNext = !isCurrentMonth && costPanelMonth < currentMonthKey();
  $('costNavPrev').style.visibility = 'visible';
  $('costNavNext').style.visibility = hasNext ? 'visible' : 'hidden';

  if (!record || record.callCount === 0) {
    $('costValue').textContent = '暂无调用记录';
    $('costDetailContent').innerHTML = '';
    $('costDetail').classList.remove('open');
    $('costOtherLink').style.display = 'none';
    return;
  }

  $('costValue').textContent = `${record.callCount} 次 · ${formatCost(record.totalCost)}`;

  let detailHtml = '';
  for (const [model, m] of Object.entries(record.byModel || {})) {
    detailHtml += `<div class="model-row"><span>${model}</span><span>${m.callCount} 次 · ${formatCost(m.totalCost)}</span></div>`;
  }

  const otherId = providerId === 'anthropic' ? 'deepseek' : 'anthropic';
  const otherDef = PROVIDER_DEFS[otherId];
  detailHtml += `<span class="other-link" data-other="${otherId}">查看 ${otherDef.displayName} 数据 →</span>`;

  $('costDetailContent').innerHTML = detailHtml;

  const link = $('costDetailContent').querySelector('[data-other]');
  if (link) {
    link.addEventListener('click', async (e) => {
      e.stopPropagation();
      await loadCostPanel(link.dataset.other);
      $('costDetail').classList.add('open');
    });
  }
}

// 成本面板折叠
$('costPanel').addEventListener('click', (e) => {
  if (e.target.id === 'costNavPrev' || e.target.id === 'costNavNext') return;
  $('costDetail').classList.toggle('open');
});

// 月份导航
$('costNavPrev').addEventListener('click', (e) => {
  e.stopPropagation();
  loadCostPanel(costPanelProvider, prevMonthKey(costPanelMonth));
  $('costDetail').classList.add('open');
});

$('costNavNext').addEventListener('click', (e) => {
  e.stopPropagation();
  const nxt = nextMonthKey(costPanelMonth);
  if (nxt <= currentMonthKey()) {
    loadCostPanel(costPanelProvider, nxt);
    $('costDetail').classList.add('open');
  }
});

// ---------- 设置 ----------

async function loadSettings() {
  const allItems = await chrome.storage.local.get([
    'provider', 'klineLimit', 'analysisStyle', 'period', 'debateMode', 'decisionMode',
    'analysisDepth', 'template',
    'apiKey:anthropic', 'apiKey:deepseek',
    'model:anthropic', 'model:deepseek',
  ]);

  const providerId = allItems.provider || 'anthropic';
  $('provider').value = providerId;
  applyProviderUI(providerId);

  $('apiKey').value = '';
  const currentApiKey = allItems[`apiKey:${providerId}`] || '';
  $('apiKey').placeholder = currentApiKey ? `已保存 (${currentApiKey.length} 字符)` : PROVIDER_DEFS[providerId].apiKeyPattern;

  const currentModel = allItems[`model:${providerId}`] || '';
  $('model').value = currentModel || '';

  $('klineLimit').value = allItems.klineLimit || '';
  $('klineLimit').placeholder = String(DEFAULT_LIMIT);
  $('template').value = allItems.template || 'technical';
  $('analysisStyle').value = allItems.analysisStyle || 'technical';
  $('period').value = allItems.period || 'monthly';
  $('debateMode').checked = allItems.debateMode || false;
  $('decisionMode').checked = allItems.decisionMode || false;
  $('multiHint').style.display = allItems.period === 'multi' ? 'block' : 'none';

  const depthValue = allItems.analysisDepth || 'standard';
  const depthRadio = document.querySelector(`input[name="analysisDepth"][value="${depthValue}"]`);
  if (depthRadio) depthRadio.checked = true;

  await loadCostPanel(providerId);
}

async function saveSettings() {
  const providerId = $('provider').value;
  const apiKey = $('apiKey').value.trim();
  const model = $('model').value.trim();
  const template = $('template').value;
  const analysisStyle = $('analysisStyle').value;
  const period = $('period').value;
  const limitRaw = parseInt($('klineLimit').value, 10);

  const def = PROVIDER_DEFS[providerId] || PROVIDER_DEFS.anthropic;

  const updates = {};
  updates.provider = providerId;
  if (apiKey) updates[`apiKey:${providerId}`] = apiKey;
  updates[`model:${providerId}`] = model || def.defaultModel;
  updates.template = template || 'technical';
  updates.analysisStyle = analysisStyle || 'technical';
  updates.period = period || 'monthly';
  updates.debateMode = !!$('debateMode').checked;
  updates.decisionMode = !!$('decisionMode').checked;
  updates.klineLimit = Number.isInteger(limitRaw) && limitRaw > 0 ? limitRaw : DEFAULT_LIMIT;

  const depthEl = document.querySelector('input[name="analysisDepth"]:checked');
  if (depthEl) updates.analysisDepth = depthEl.value;

  await chrome.storage.local.set(updates);
  setStatus('已保存');
  $('apiKey').value = '';
  await loadSettings();
}

async function clearCurrentProviderKey() {
  const providerId = $('provider').value;
  await chrome.storage.local.remove([`apiKey:${providerId}`]);
  setStatus('Key 已清除');
  await loadSettings();
}

function setStatus(msg) {
  const el = $('status');
  el.textContent = msg;
  setTimeout(() => { if (el.textContent === msg) el.textContent = ''; }, 2500);
}

function validateModel() {
  const providerId = $('provider').value;
  const modelValue = $('model').value.trim();
  const def = PROVIDER_DEFS[providerId] || PROVIDER_DEFS.anthropic;
  const hintEl = $('styleHint');
  if (modelValue && !def.knownModels.includes(modelValue)) {
    hintEl.textContent = `⚠️ '${modelValue}' 不是 ${def.displayName} 的已知模型名，可能导致 API 静默回退到默认模型，输出质量不可控`;
    hintEl.style.display = 'block';
    hintEl.style.color = '#d97706';
  } else {
    hintEl.style.display = 'none';
  }
}

// provider 切换监听
$('provider').addEventListener('change', async () => {
  const newProvider = $('provider').value;
  applyProviderUI(newProvider);

  const items = await chrome.storage.local.get([`apiKey:${newProvider}`, `model:${newProvider}`]);
  const def = PROVIDER_DEFS[newProvider] || PROVIDER_DEFS.anthropic;

  const apiKey = items[`apiKey:${newProvider}`] || '';
  $('apiKey').value = '';
  $('apiKey').placeholder = apiKey ? `已保存 (${apiKey.length} 字符)` : def.apiKeyPattern;

  validateModel();

  const model = items[`model:${newProvider}`] || '';
  $('model').value = model;
  $('model').placeholder = def.defaultModel;

  await loadCostPanel(newProvider);
});

// ---------- 历史 Tab ----------

const TEMPLATE_LABELS = { technical: '技术面', trend: '趋势判断', valuation: '估值面', sentiment: '情绪面' };
const PROVIDER_NAME = { anthropic: 'Claude', deepseek: 'DeepSeek' };

let historyListCache = [];

async function loadHistoryTab() {
  try {
    const resp = await chrome.runtime.sendMessage({ type: 'GET_HISTORY' });
    if (!resp || !resp.ok) {
      $('historyList').innerHTML = '<div class="history-empty">加载失败</div>';
      return;
    }
    historyListCache = resp.list || [];

    // 容量横幅
    const capBar = $('capacityBar');
    if (historyListCache.length > 0) {
      const pct = resp.maxItems ? Math.round(historyListCache.length / resp.maxItems * 100) : 0;
      const bytesMB = (resp.bytes / 1024 / 1024).toFixed(1);
      const maxMB = (resp.maxBytes / 1024 / 1024).toFixed(1);
      $('capacityText').textContent = `${historyListCache.length} / ${resp.maxItems} 条 · ${bytesMB} / ${maxMB} MB`;
      capBar.style.display = 'flex';
      capBar.className = pct > 80 ? 'capacity-bar warn' : 'capacity-bar';
    } else {
      capBar.style.display = 'none';
    }

    // 渲染列表
    if (historyListCache.length === 0) {
      $('historyList').innerHTML = '<div class="history-empty">暂无保存的分析记录</div>';
      $('historyGlobalActions').style.display = 'none';
    } else {
      renderHistoryCards();
      $('historyGlobalActions').style.display = 'flex';
    }
  } catch (_) {
    $('historyList').innerHTML = '<div class="history-empty">加载失败</div>';
  }
}

function renderHistoryCards() {
  $('historyList').innerHTML = historyListCache.map((entry, idx) => {
    const dtStr = entry.timestamp ? formatTime(entry.timestamp) : '?';
    const providerTag = PROVIDER_NAME[entry.provider] || (entry.provider || '?');
    const templateTag = TEMPLATE_LABELS[entry.template] || (entry.template || '?');
    const hasConv = entry.conversationHistory && entry.conversationHistory.length > 2;
    const badgeCls = hasConv ? 'hasconv' : '';

    return `
      <div class="history-item" data-idx="${idx}">
        <div class="history-header" data-action="toggle">
          <div class="hi-info">
            <div class="hi-title">${escapeHtml(entry.name || '?')} (${escapeHtml(entry.code || '?')})</div>
            <div class="hi-sub">${dtStr} · ${escapeHtml(providerTag)} / ${escapeHtml(entry.model || '?')} · ${escapeHtml(templateTag)}</div>
          </div>
          <span class="hi-badge ${badgeCls}">${hasConv ? '有追问' : '仅分析'}</span>
          <span class="hi-chevron">▶</span>
        </div>
        <div class="history-body">
          ${renderMarkdownSimple(entry.analysis || '')}
          ${renderConversationHistory(entry.conversationHistory)}
        </div>
        <div class="history-actions">
          <button class="h-btn h-btn-export" data-action="export" data-idx="${idx}">📥 导出</button>
          <button class="h-btn h-btn-del" data-action="delete" data-idx="${idx}">🗑 删除</button>
        </div>
      </div>
    `;
  }).join('');

  // 事件绑定
  $('historyList').querySelectorAll('.history-item').forEach((el) => {
    const idx = parseInt(el.dataset.idx, 10);
    el.querySelector('[data-action="toggle"]').addEventListener('click', () => toggleHistoryCard(el, idx));
    el.querySelector('[data-action="export"]').addEventListener('click', (e) => { e.stopPropagation(); exportSingle(idx); });
    el.querySelector('[data-action="delete"]').addEventListener('click', (e) => { e.stopPropagation(); deleteSingle(idx); });
  });
}

function toggleHistoryCard(el, idx) {
  const body = el.querySelector('.history-body');
  const chevron = el.querySelector('.hi-chevron');
  if (body.classList.contains('open')) {
    body.classList.remove('open');
    chevron.textContent = '▶';
  } else {
    body.classList.add('open');
    chevron.textContent = '▼';
  }
}

function renderConversationHistory(conv) {
  if (!conv || conv.length <= 2) return '';
  const msgs = conv.slice(2);
  if (msgs.length === 0) return '';
  let html = '<div class="conv-section"><h4>追问记录</h4>';
  for (const msg of msgs) {
    const cls = msg.role === 'user' ? 'user' : 'assistant';
    const label = msg.role === 'user' ? '🧑' : '🤖';
    html += `<div class="conv-msg ${cls}">${label} ${escapeHtml(msg.content.slice(0, 200))}${msg.content.length > 200 ? '...' : ''}</div>`;
  }
  html += '</div>';
  return html;
}

async function exportSingle(idx) {
  const entry = historyListCache[idx];
  if (!entry) return;
  try {
    const resp = await chrome.runtime.sendMessage({ type: 'EXPORT_HISTORY', id: entry.id });
    if (resp && resp.ok) {
      downloadBlob(resp.markdown, resp.filename, 'text/markdown');
    }
  } catch (_) { /* ignore */ }
}

async function deleteSingle(idx) {
  const entry = historyListCache[idx];
  if (!entry) return;
  if (!confirm(`删除 ${entry.name || '?'}(${entry.code || '?'}) 的分析记录?`)) return;
  try {
    await chrome.runtime.sendMessage({ type: 'DELETE_HISTORY', id: entry.id });
    await loadHistoryTab();
  } catch (_) { /* ignore */ }
}

async function exportAll() {
  if (historyListCache.length === 0) return;
  try {
    const resp = await chrome.runtime.sendMessage({ type: 'EXPORT_HISTORY' });
    if (resp && resp.ok) {
      downloadBlob(resp.markdown, resp.filename, 'text/markdown');
    }
  } catch (_) { /* ignore */ }
}

async function clearAllHistory() {
  if (historyListCache.length === 0) return;
  if (!confirm(`确认清空全部 ${historyListCache.length} 条分析记录? 此操作不可撤销。`)) return;
  try {
    await chrome.runtime.sendMessage({ type: 'CLEAR_HISTORY' });
    await loadHistoryTab();
  } catch (_) { /* ignore */ }
}

function downloadBlob(content, filename, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// ---------- 简易 Markdown 渲染（历史卡片内） ----------

function renderMarkdownSimple(md) {
  const lines = String(md).split('\n');
  const out = [];
  let inList = false;
  const closeList = () => { if (inList) { out.push('</ul>'); inList = false; } };

  for (const raw of lines) {
    if (raw.trim() === '```' || raw.trim() === '```json') continue;
    const h = raw.match(/^(#{1,4})\s+(.*)$/);
    if (h) {
      closeList();
      const level = Math.min(h[1].length + 2, 5);
      out.push(`<h${level}>${formatInlineSimple(h[2])}</h${level}>`);
      continue;
    }
    const li = raw.match(/^[-*]\s+(.*)$/);
    if (li) {
      if (!inList) { out.push('<ul>'); inList = true; }
      out.push(`<li>${formatInlineSimple(li[1])}</li>`);
      continue;
    }
    const nli = raw.match(/^\d+[\.\)]\s+(.*)$/);
    if (nli) {
      if (!inList) { out.push('<ul>'); inList = true; }
      out.push(`<li>${formatInlineSimple(nli[1])}</li>`);
      continue;
    }
    if (raw.trim() === '') { closeList(); continue; }
    closeList();
    out.push(`<p>${formatInlineSimple(raw)}</p>`);
  }
  closeList();
  return out.join('');
}

function formatInlineSimple(s) {
  let x = escapeHtml(s);
  x = x.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  x = x.replace(/`([^`]+)`/g, '<code>$1</code>');
  return x;
}

// ---------- 工具 ----------

function formatTime(ms) {
  const d = new Date(ms);
  const pad = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

function escapeAttr(s) {
  return String(s).replace(/"/g, '&quot;');
}

// ---------- 启动 ----------

document.addEventListener('DOMContentLoaded', async () => {
  await loadSettings();
  validateModel();

  $('saveBtn').addEventListener('click', saveSettings);
  $('clearBtn').addEventListener('click', clearCurrentProviderKey);
  $('model').addEventListener('blur', validateModel);
  $('period').addEventListener('change', () => {
    $('multiHint').style.display = $('period').value === 'multi' ? 'block' : 'none';
  });

  // 历史 Tab 全局按钮
  $('exportAllBtn').addEventListener('click', exportAll);
  $('clearAllHistoryBtn').addEventListener('click', clearAllHistory);

  // 结构化数据 debug 入口
  $('structuredDebugLink').addEventListener('click', async () => {
    const contentEl = $('structuredDebugContent');
    if (contentEl.style.display !== 'none') {
      contentEl.style.display = 'none';
      return;
    }
    const all = await chrome.storage.local.get(null);
    const structuredKeys = Object.keys(all).filter((k) => k.startsWith('structured:')).sort();
    if (structuredKeys.length === 0) {
      contentEl.innerHTML = '<div style="font-size:11px;color:#999;margin-top:6px;">暂无结构化数据缓存</div>';
    } else {
      contentEl.innerHTML = structuredKeys.map((k) => {
        const v = all[k];
        const dt = v.timestamp ? formatTime(v.timestamp) : '?';
        return `<details style="font-size:11px;margin:4px 0;"><summary style="cursor:pointer;color:#555;">${escapeHtml(k)} · ${dt}</summary><pre style="background:#f8f8f8;padding:8px;overflow-x:auto;font-size:11px;max-height:200px;">${escapeHtml(JSON.stringify(v.data, null, 2))}</pre></details>`;
      }).join('');
    }
    contentEl.style.display = 'block';
  });
});
