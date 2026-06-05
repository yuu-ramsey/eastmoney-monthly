// popup settings page + preset modes + cost panel + analysis history + debug panel

const $ = (id) => document.getElementById(id);

// provider configuration
const PROVIDER_DEFS = {
  anthropic: {
    displayName: 'Anthropic Claude',
    defaultModel: 'claude-sonnet-4-6',
    apiKeyPattern: 'sk-ant-...',
    apiKeyHelp: 'Get from console.anthropic.com/settings/keys (requires international payment)',
    docUrl: 'https://docs.claude.com/en/docs/about-claude/models/overview',
    knownModels: ['claude-opus-4-7', 'claude-sonnet-4-6', 'claude-haiku-4-5', 'claude-haiku-4-5-20251001'],
  },
  deepseek: {
    displayName: 'DeepSeek',
    defaultModel: 'deepseek-chat',
    apiKeyPattern: 'sk-...',
    apiKeyHelp: 'Get from platform.deepseek.com/api_keys (Alipay/WeChat top-up supported)',
    docUrl: 'https://api-docs.deepseek.com/',
    knownModels: ['deepseek-chat', 'deepseek-reasoner'],
  },
};

const DEFAULT_LIMIT = 60;

// preset definitions
const PRESETS = {
  quick: {
    label: 'Quick',
    provider: 'anthropic',
    analysisDepth: 'standard',
    model: 'claude-sonnet-4-6',
    debateMode: false,
    template: 'technical',
    analysisStyle: 'technical',
    enableSelfBacktest: false,
    enableThinking: false,
    enableDebugLog: false,
    summary: 'Sonnet single technical analysis, low cost, fast',
    estCost: '¥0.02',
  },
  deep: {
    label: 'Deep',
    provider: 'anthropic',
    analysisDepth: 'deep',
    model: 'claude-opus-4-7',
    debateMode: false,
    template: 'technical',
    analysisStyle: 'technical',
    enableSelfBacktest: true,
    enableThinking: true,
    enableDebugLog: false,
    summary: 'Opus single technical analysis with self-backtest + extended thinking',
    estCost: '¥0.5',
  },
  debate: {
    label: 'Debate',
    provider: 'anthropic',
    analysisDepth: 'deep',
    model: 'claude-opus-4-7',
    debateMode: true,
    template: 'technical',
    analysisStyle: 'technical',
    enableSelfBacktest: false,
    enableThinking: false,
    enableDebugLog: false,
    summary: 'Opus multi-agent debate, highest quality, most expensive',
    estCost: '¥1.5',
  },
};

// ---------- Tab switching ----------

document.querySelectorAll('.tab-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach((c) => c.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
    if (btn.dataset.tab === 'history') loadHistoryTab();
    if (btn.dataset.tab === 'debug') loadDebugTab();
  });
});

// ---------- Provider UI ----------

function applyProviderUI(providerId) {
  const def = PROVIDER_DEFS[providerId] || PROVIDER_DEFS.anthropic;
  $('apiKey').placeholder = def.apiKeyPattern;
  $('apiKeyHint').innerHTML = def.apiKeyHelp + ' <a href="' + def.docUrl + '" target="_blank">Docs</a>';
  $('model').placeholder = def.defaultModel;
}

// ---------- cost panel ----------

function formatCost(cost) {
  if (cost < 0.01) return '&lt;¥0.01';
  if (cost < 1) return '¥' + cost.toFixed(2);
  return '¥' + cost.toFixed(2);
}
function formatMonthKey(ym) {
  const [y, m] = ym.split('-');
  return y + '-' + m;
}
let costPanelProvider = 'anthropic';
let costPanelMonth = new Date().toISOString().slice(0, 7);
function currentMonthKey() { return new Date().toISOString().slice(0, 7); }
function prevMonthKey(ym) { const d = new Date(ym + '-01'); d.setMonth(d.getMonth() - 1); return d.toISOString().slice(0, 7); }
function nextMonthKey(ym) { const d = new Date(ym + '-01'); d.setMonth(d.getMonth() + 1); return d.toISOString().slice(0, 7); }

async function loadCostPanel(providerId, monthKey) {
  costPanelProvider = providerId;
  costPanelMonth = monthKey || currentMonthKey();
  const def = PROVIDER_DEFS[providerId] || PROVIDER_DEFS.anthropic;
  const items = await chrome.storage.local.get(['usage:' + providerId + ':' + costPanelMonth]);
  const record = items['usage:' + providerId + ':' + costPanelMonth];
  const isCurrentMonth = costPanelMonth === currentMonthKey();
  $('costLabel').textContent = def.displayName + ' ' + formatMonthKey(costPanelMonth);
  $('costNavPrev').style.visibility = 'visible';
  $('costNavNext').style.visibility = (!isCurrentMonth && costPanelMonth < currentMonthKey()) ? 'visible' : 'hidden';

  if (!record || record.callCount === 0) {
    $('costValue').textContent = 'No usage records';
    $('costDetailContent').innerHTML = '';
    $('costDetail').classList.remove('open');
    return;
  }
  $('costValue').textContent = record.callCount + ' calls . ' + formatCost(record.totalCost);
  let detailHtml = '';
  for (const [model, m] of Object.entries(record.byModel || {})) {
    detailHtml += '<div class="model-row"><span>' + model + '</span><span>' + m.callCount + ' calls . ' + formatCost(m.totalCost) + '</span></div>';
  }
  const otherId = providerId === 'anthropic' ? 'deepseek' : 'anthropic';
  detailHtml += '<span class="other-link" data-other="' + otherId + '">View ' + PROVIDER_DEFS[otherId].displayName + ' data</span>';
  $('costDetailContent').innerHTML = detailHtml;
  const link = $('costDetailContent').querySelector('[data-other]');
  if (link) link.addEventListener('click', async (e) => { e.stopPropagation(); await loadCostPanel(link.dataset.other); $('costDetail').classList.add('open'); });
}

$('costPanel').addEventListener('click', (e) => {
  if (e.target.id === 'costNavPrev' || e.target.id === 'costNavNext') return;
  $('costDetail').classList.toggle('open');
});
$('costNavPrev').addEventListener('click', (e) => { e.stopPropagation(); loadCostPanel(costPanelProvider, prevMonthKey(costPanelMonth)); $('costDetail').classList.add('open'); });
$('costNavNext').addEventListener('click', (e) => { e.stopPropagation(); const nxt = nextMonthKey(costPanelMonth); if (nxt <= currentMonthKey()) { loadCostPanel(costPanelProvider, nxt); $('costDetail').classList.add('open'); } });

// ---------- preset modes ----------

let currentPreset = 'deep';

function getCurrentPreset() {
  return currentPreset;
}

function setCurrentPreset(p) {
  currentPreset = p;
  document.querySelectorAll('.preset-btn').forEach((b) => b.classList.toggle('active', b.dataset.preset === p));
}

// write preset values to form
function applyPresetFields(preset) {
  const p = PRESETS[preset];
  if (!p) return; // custom does not modify form

  $('provider').value = p.provider;
  $('model').value = p.model;
  document.querySelector('input[name="analysisDepth"][value="' + p.analysisDepth + '"]').checked = true;
  $('debateMode').checked = p.debateMode;
  $('template').value = p.template;
  $('analysisStyle').value = p.analysisStyle;
  $('enableSelfBacktest').checked = p.enableSelfBacktest;
  $('enableThinking').checked = p.enableThinking;
  $('enableDebugLog').checked = p.enableDebugLog;
}

// dynamically show/hide options
function updateUIVisibility(preset) {
  const isCustom = preset === 'custom';
  const provider = $('provider').value;
  const isAnthropic = provider === 'anthropic';
  const isDebate = $('debateMode').checked;
  const depth = document.querySelector('input[name="analysisDepth"]:checked');
  const isStandard = depth && depth.value === 'standard';

  // detail settings collapse
  const group = $('settingsGroup');
  const toggleIcon = $('settingsToggleIcon');
  const toggleLabel = $('settingsToggleLabel');
  if (isCustom) {
    group.classList.add('open');
    toggleIcon.textContent = 'v';
    toggleLabel.textContent = 'Hide Advanced Settings';
  } else {
    group.classList.remove('open');
    toggleIcon.textContent = '>';
    toggleLabel.textContent = 'Show Advanced Settings';
  }

  // Provider-related
  $('rowDepth').style.display = isAnthropic ? '' : 'none';
  $('modelHint').textContent = isAnthropic
    ? 'Manually enter model ID. Leave blank to auto-select based on "Analysis Depth".'
    : 'DeepSeek always uses deepseek-chat';

  // debate mode-related
  $('rowTemplate').style.display = isDebate ? 'none' : '';
  $('rowAnalysisStyle').style.display = isDebate ? '' : 'none';
  $('rowSelfBacktest').style.display = isDebate ? 'none' : '';
  $('rowThinking').style.display = isDebate ? 'none' : '';

  // Thinking: only available for Opus, grayed out on standard
  if (isAnthropic && !isDebate) {
    const thinkingLabel = $('enableThinking').closest('label');
    if (isStandard) {
      $('enableThinking').disabled = true;
      $('enableThinking').checked = false;
      thinkingLabel.style.opacity = '0.5';
      $('rowThinking').querySelector('.hint').textContent = 'Only available for Opus (Deep mode). Currently in Standard mode.';
    } else {
      $('enableThinking').disabled = false;
      thinkingLabel.style.opacity = '1';
      $('rowThinking').querySelector('.hint').textContent = 'Only effective with Anthropic Opus model.';
    }
  }

  // DeepSeek model lock
  if (!isAnthropic) {
    $('model').value = 'deepseek-chat';
    $('model').readOnly = true;
    $('model').style.background = '#f5f5f5';
  } else {
    $('model').readOnly = false;
    $('model').style.background = '';
  }
}

// update summary row
function updateSummary() {
  const preset = getCurrentPreset();
  if (preset === 'custom') {
    $('presetSummary').style.display = 'none';
    return;
  }
  const p = PRESETS[preset];
  const providerLabel = PROVIDER_DEFS[p.provider].displayName;
  $('presetSummary').style.display = 'block';
  $('presetSummary').textContent = 'Current: ' + p.label + ' | ' + providerLabel + ' ' + p.model + ' | Est. cost: ' + p.estCost;
}

// preset button click
document.querySelectorAll('.preset-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    const preset = btn.dataset.preset;
    setCurrentPreset(preset);
    if (preset !== 'custom') {
      applyPresetFields(preset);
      applyProviderUI(PRESETS[preset].provider);
      // fill current saved API key
      chrome.storage.local.get(['apiKey:' + PRESETS[preset].provider]).then((items) => {
        const key = items['apiKey:' + PRESETS[preset].provider] || '';
        $('apiKey').value = '';
        $('apiKey').placeholder = key ? 'Saved (' + key.length + ' chars)' : PROVIDER_DEFS[PRESETS[preset].provider].apiKeyPattern;
      });
    }
    updateUIVisibility(preset);
    updateSummary();
    updateMultiHint();
  });
});

// expand/collapse detail settings
$('settingsToggle').addEventListener('click', () => {
  const group = $('settingsGroup');
  const open = !group.classList.contains('open');
  group.classList.toggle('open', open);
  $('settingsToggleIcon').textContent = open ? 'v' : '>';
  $('settingsToggleLabel').textContent = open ? 'Hide Advanced Settings' : 'Show Advanced Settings';
});

// refresh UI when provider changes
$('provider').addEventListener('change', () => {
  updateUIVisibility(getCurrentPreset());
  const newProvider = $('provider').value;
  applyProviderUI(newProvider);
  chrome.storage.local.get(['apiKey:' + newProvider, 'model:' + newProvider]).then((items) => {
    $('apiKey').value = '';
    $('apiKey').placeholder = items['apiKey:' + newProvider] ? 'Saved (' + items['apiKey:' + newProvider].length + ' chars)' : PROVIDER_DEFS[newProvider].apiKeyPattern;
    $('model').value = items['model:' + newProvider] || '';
    validateModel();
  });
  loadCostPanel(newProvider);
});

// refresh when debateMode changes
$('debateMode').addEventListener('change', () => updateUIVisibility(getCurrentPreset()));

// refresh thinking gray-out when analysisDepth changes
document.querySelectorAll('input[name="analysisDepth"]').forEach((r) => {
  r.addEventListener('change', () => updateUIVisibility(getCurrentPreset()));
});

function updateMultiHint() {
  $('multiHint').style.display = $('period').value === 'multi' ? 'block' : 'none';
}

// ---------- settings loading ----------

async function loadSettings() {
  const allItems = await chrome.storage.local.get([
    'provider', 'klineLimit', 'analysisStyle', 'period', 'debateMode', 'decisionMode',
    'analysisDepth', 'template', 'enableSelfBacktest', 'enableThinking', 'enableDebugLog',
    'usagePreset',
    'apiKey:anthropic', 'apiKey:deepseek',
    'model:anthropic', 'model:deepseek',
  ]);

  const preset = allItems.usagePreset || 'deep';
  setCurrentPreset(preset);

  const providerId = allItems.provider || 'anthropic';
  $('provider').value = providerId;
  applyProviderUI(providerId);

  const currentApiKey = allItems['apiKey:' + providerId] || '';
  $('apiKey').value = '';
  $('apiKey').placeholder = currentApiKey ? 'Saved (' + currentApiKey.length + ' chars)' : PROVIDER_DEFS[providerId].apiKeyPattern;

  const currentModel = allItems['model:' + providerId] || '';
  $('model').value = currentModel || '';

  $('klineLimit').value = allItems.klineLimit || '';
  $('klineLimit').placeholder = String(DEFAULT_LIMIT);
  $('template').value = allItems.template || 'technical';
  $('analysisStyle').value = allItems.analysisStyle || 'technical';
  $('period').value = allItems.period || 'monthly';
  $('debateMode').checked = allItems.debateMode || false;
  $('decisionMode').checked = allItems.decisionMode || false;
  $('enableSelfBacktest').checked = allItems.enableSelfBacktest !== undefined ? !!allItems.enableSelfBacktest : true;
  $('enableThinking').checked = !!allItems.enableThinking;
  $('enableDebugLog').checked = !!allItems.enableDebugLog;
  $('multiHint').style.display = allItems.period === 'multi' ? 'block' : 'none';

  const depthValue = allItems.analysisDepth || 'standard';
  const depthRadio = document.querySelector('input[name="analysisDepth"][value="' + depthValue + '"]');
  if (depthRadio) depthRadio.checked = true;

  updateUIVisibility(preset);
  updateSummary();
  await loadCostPanel(providerId);
}

// ---------- save ----------

async function saveSettings() {
  const preset = getCurrentPreset();
  const providerId = $('provider').value;
  const apiKey = $('apiKey').value.trim();
  const model = $('model').value.trim();
  const template = $('template').value;
  const analysisStyle = $('analysisStyle').value;
  const period = $('period').value;
  const limitRaw = parseInt($('klineLimit').value, 10);
  const def = PROVIDER_DEFS[providerId] || PROVIDER_DEFS.anthropic;

  const depthEl = document.querySelector('input[name="analysisDepth"]:checked');
  const analysisDepth = depthEl ? depthEl.value : 'standard';

  const updates = {};
  updates.usagePreset = preset;
  updates.provider = providerId;
  if (apiKey) updates['apiKey:' + providerId] = apiKey;
  updates['model:' + providerId] = model || def.defaultModel;
  updates.template = template || 'technical';
  updates.analysisStyle = analysisStyle || 'technical';
  updates.period = period || 'monthly';
  updates.analysisDepth = analysisDepth;
  updates.debateMode = !!$('debateMode').checked;
  updates.decisionMode = !!$('decisionMode').checked;
  updates.enableSelfBacktest = !!$('enableSelfBacktest').checked;
  updates.enableThinking = !!$('enableThinking').checked;
  updates.enableDebugLog = !!$('enableDebugLog').checked;
  updates.klineLimit = Number.isInteger(limitRaw) && limitRaw > 0 ? limitRaw : DEFAULT_LIMIT;

  await chrome.storage.local.set(updates);
  setStatus('Saved');
  $('apiKey').value = '';
  await loadSettings();
}

async function clearCurrentProviderKey() {
  const providerId = $('provider').value;
  await chrome.storage.local.remove(['apiKey:' + providerId]);
  setStatus('Key cleared');
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
    hintEl.textContent = '\'' + modelValue + '\' is not a known model name for ' + def.displayName + '. May cause silent API fallback to default model.';
    hintEl.style.display = 'block';
    hintEl.style.color = '#d97706';
  } else {
    hintEl.style.display = 'none';
  }
}

// ---------- quick navigation ----------

function resolveMarket(code) {
  const c = String(code).trim();
  if (/^6/.test(c)) return 'sh';
  if (/^[03]/.test(c)) return 'sz';
  return null;
}
function buildQuoteUrl(code) {
  const market = resolveMarket(code);
  if (!market) return null;
  return 'https://quote.eastmoney.com/' + market + code + '.html';
}
function openStockPage() {
  const code = $('navCode').value.trim();
  const hint = $('navHint');
  if (!code) { hint.textContent = 'Please enter a stock code'; hint.style.color = '#e74c3c'; return; }
  const url = buildQuoteUrl(code);
  if (!url) { hint.textContent = 'Unrecognized code "' + code + '". Shanghai starts with 6, Shenzhen starts with 0/3'; hint.style.color = '#e74c3c'; return; }
  hint.textContent = '';
  chrome.tabs.create({ url });
}

// ---------- History Tab ----------

const TEMPLATE_LABELS = { technical: 'Technical', trend: 'Trend', valuation: 'Valuation', sentiment: 'Sentiment' };
const PROVIDER_NAME = { anthropic: 'Claude', deepseek: 'DeepSeek' };
let historyListCache = [];

async function loadHistoryTab() {
  try {
    const resp = await chrome.runtime.sendMessage({ type: 'GET_HISTORY' });
    if (!resp || !resp.ok) { $('historyList').innerHTML = '<div class="history-empty">Failed to load</div>'; return; }
    historyListCache = resp.list || [];
    const capBar = $('capacityBar');
    if (historyListCache.length > 0) {
      const pct = resp.maxItems ? Math.round(historyListCache.length / resp.maxItems * 100) : 0;
      const bytesMB = (resp.bytes / 1024 / 1024).toFixed(1);
      const maxMB = (resp.maxBytes / 1024 / 1024).toFixed(1);
      $('capacityText').textContent = historyListCache.length + ' / ' + resp.maxItems + ' items . ' + bytesMB + ' / ' + maxMB + ' MB';
      capBar.style.display = 'flex';
      capBar.className = pct > 80 ? 'capacity-bar warn' : 'capacity-bar';
    } else { capBar.style.display = 'none'; }
    if (historyListCache.length === 0) {
      $('historyList').innerHTML = '<div class="history-empty">No saved analysis records</div>';
      $('historyGlobalActions').style.display = 'none';
    } else { renderHistoryCards(); $('historyGlobalActions').style.display = 'flex'; }
  } catch (_) { $('historyList').innerHTML = '<div class="history-empty">Failed to load</div>'; }
}

function renderHistoryCards() {
  $('historyList').innerHTML = historyListCache.map((entry, idx) => {
    const dtStr = entry.timestamp ? formatTime(entry.timestamp) : '?';
    const providerTag = PROVIDER_NAME[entry.provider] || (entry.provider || '?');
    const templateTag = TEMPLATE_LABELS[entry.template] || (entry.template || '?');
    const hasConv = entry.conversationHistory && entry.conversationHistory.length > 2;
    return '<div class="history-item" data-idx="' + idx + '">'
      + '<div class="history-header" data-action="toggle">'
      + '<div class="hi-info"><div class="hi-title">' + escapeHtml(entry.name || '?') + ' (' + escapeHtml(entry.code || '?') + ')</div>'
      + '<div class="hi-sub">' + dtStr + ' . ' + escapeHtml(providerTag) + ' / ' + escapeHtml(entry.model || '?') + ' . ' + escapeHtml(templateTag) + '</div></div>'
      + '<span class="hi-badge ' + (hasConv ? 'hasconv' : '') + '">' + (hasConv ? 'Conversation' : 'Analysis') + '</span>'
      + '<span class="hi-chevron">></span></div>'
      + '<div class="history-body">' + renderMarkdownSimple(entry.analysis || '') + renderConversationHistory(entry.conversationHistory) + '</div>'
      + '<div class="history-actions"><button class="h-btn h-btn-export" data-action="export" data-idx="' + idx + '">Export</button>'
      + '<button class="h-btn h-btn-del" data-action="delete" data-idx="' + idx + '">Delete</button></div></div>';
  }).join('');
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
  if (body.classList.contains('open')) { body.classList.remove('open'); chevron.textContent = '>'; }
  else { body.classList.add('open'); chevron.textContent = 'v'; }
}
function renderConversationHistory(conv) {
  if (!conv || conv.length <= 2) return '';
  const msgs = conv.slice(2);
  if (msgs.length === 0) return '';
  let html = '<div class="conv-section"><h4>Conversation</h4>';
  for (const msg of msgs) {
    const cls = msg.role === 'user' ? 'user' : 'assistant';
    html += '<div class="conv-msg ' + cls + '">' + (msg.role === 'user' ? 'U:' : 'A:') + ' ' + escapeHtml(msg.content.slice(0, 200)) + (msg.content.length > 200 ? '...' : '') + '</div>';
  }
  return html + '</div>';
}
async function exportSingle(idx) {
  const entry = historyListCache[idx];
  if (!entry) return;
  try {
    const resp = await chrome.runtime.sendMessage({ type: 'EXPORT_HISTORY', id: entry.id });
    if (resp && resp.ok) downloadBlob(resp.markdown, resp.filename, 'text/markdown');
  } catch (_) { /* ignore */ }
}
async function deleteSingle(idx) {
  const entry = historyListCache[idx];
  if (!entry) return;
  if (!confirm('Delete analysis record for ' + (entry.name || '?') + '(' + (entry.code || '?') + ')?')) return;
  try { await chrome.runtime.sendMessage({ type: 'DELETE_HISTORY', id: entry.id }); await loadHistoryTab(); } catch (_) { /* ignore */ }
}
async function exportAll() {
  if (historyListCache.length === 0) return;
  try { const resp = await chrome.runtime.sendMessage({ type: 'EXPORT_HISTORY' }); if (resp && resp.ok) downloadBlob(resp.markdown, resp.filename, 'text/markdown'); } catch (_) { /* ignore */ }
}
async function clearAllHistory() {
  if (historyListCache.length === 0) return;
  if (!confirm('Clear all ' + historyListCache.length + ' analysis records? This cannot be undone.')) return;
  try { await chrome.runtime.sendMessage({ type: 'CLEAR_HISTORY' }); await loadHistoryTab(); } catch (_) { /* ignore */ }
}
function downloadBlob(content, filename, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

// ---------- simple Markdown rendering ----------
function renderMarkdownSimple(md) {
  const lines = String(md).split('\n');
  const out = [];
  let inList = false;
  const closeList = () => { if (inList) { out.push('</ul>'); inList = false; } };
  for (const raw of lines) {
    if (raw.trim() === '```' || raw.trim() === '```json') continue;
    const h = raw.match(/^(#{1,4})\s+(.*)$/);
    if (h) { closeList(); out.push('<h' + Math.min(h[1].length + 2, 5) + '>' + formatInlineSimple(h[2]) + '</h' + Math.min(h[1].length + 2, 5) + '>'); continue; }
    const li = raw.match(/^[-*]\s+(.*)$/);
    if (li) { if (!inList) { out.push('<ul>'); inList = true; } out.push('<li>' + formatInlineSimple(li[1]) + '</li>'); continue; }
    const nli = raw.match(/^\d+[\.\)]\s+(.*)$/);
    if (nli) { if (!inList) { out.push('<ul>'); inList = true; } out.push('<li>' + formatInlineSimple(nli[1]) + '</li>'); continue; }
    if (raw.trim() === '') { closeList(); continue; }
    closeList(); out.push('<p>' + formatInlineSimple(raw) + '</p>');
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

// ---------- Debug Tab ----------

async function loadDebugTab() {
  const items = await chrome.storage.local.get(['debug:lastAnalysis']);
  const record = items['debug:lastAnalysis'];
  if (!record) { $('debugEmpty').style.display = 'block'; $('debugContent').style.display = 'none'; return; }
  $('debugEmpty').style.display = 'none';
  $('debugContent').style.display = 'block';
  const dtStr = record.timestamp ? formatTime(record.timestamp) : '?';
  $('debugSummary').innerHTML = '<strong>' + escapeHtml(record.name || '?') + '(' + escapeHtml(record.code || '?') + ')</strong> . ' + dtStr + ' . ' + escapeHtml(record.provider || '?') + '/' + escapeHtml(record.model || '?') + ' . ' + escapeHtml(record.template || '?') + ' . Duration ' + (record.durationMs || '?') + 'ms . Cost ¥' + ((record.cost && record.cost.cny) || 0).toFixed(4);
  $('debugPromptContent').textContent = record.fullPrompt || '';
  if (record.toolCalls && record.toolCalls.length > 0) {
    $('debugToolCalls').innerHTML = record.toolCalls.map((tc) => '<div class="debug-tool-item"><div class="debug-tool-name">' + escapeHtml(tc.name) + '</div><div class="debug-tool-meta">Params: ' + escapeHtml(JSON.stringify(tc.input)) + ' . Duration ' + (tc.durationMs || '?') + 'ms</div><div class="debug-tool-result">' + (tc.result ? escapeHtml(String(tc.result).slice(0, 500)) : '(no result)') + '</div></div>').join('');
  } else { $('debugToolCalls').innerHTML = '<div style="padding:8px 10px;font-size:12px;color:#888;">No tool calls</div>'; }
  $('debugRawResponseContent').textContent = record.rawResponse || '';
  const u = record.usage || {};
  $('debugUsage').innerHTML = '<div>Input tokens: ' + (u.inputTokens || 0) + ' (' + ((u.inputTokens || 0) / 1000).toFixed(1) + 'K)</div><div>Output tokens: ' + (u.outputTokens || 0) + ' (' + ((u.outputTokens || 0) / 1000).toFixed(1) + 'K)</div><div>Cost: ¥' + ((record.cost && record.cost.cny) || 0).toFixed(4) + '</div>';
}

// ---------- utilities ----------
function formatTime(ms) {
  const d = new Date(ms);
  const pad = (n) => String(n).padStart(2, '0');
  return d.getFullYear() + '-' + pad(d.getMonth() + 1) + '-' + pad(d.getDate()) + ' ' + pad(d.getHours()) + ':' + pad(d.getMinutes());
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

// ---------- startup ----------
document.addEventListener('DOMContentLoaded', async () => {
  await loadSettings();
  validateModel();

  $('navOpenBtn').addEventListener('click', openStockPage);
  $('navCode').addEventListener('keydown', (e) => { if (e.key === 'Enter') openStockPage(); });

  $('saveBtn').addEventListener('click', saveSettings);
  $('clearBtn').addEventListener('click', clearCurrentProviderKey);
  $('model').addEventListener('blur', validateModel);
  $('period').addEventListener('change', updateMultiHint);

  $('exportAllBtn').addEventListener('click', exportAll);
  $('clearAllHistoryBtn').addEventListener('click', clearAllHistory);
  $('clearDebugBtn').addEventListener('click', async () => { await chrome.storage.local.remove(['debug:lastAnalysis']); loadDebugTab(); });

  // debug panel collapse/copy
  document.addEventListener('click', (e) => {
    if (e.target.dataset.action === 'toggle-debug') {
      const el = document.getElementById(e.target.dataset.target);
      if (el) el.style.display = el.style.display === 'none' ? 'block' : 'none';
    }
    if (e.target.dataset.copy) {
      const el = document.getElementById(e.target.dataset.copy);
      if (el) { navigator.clipboard.writeText(el.textContent || ''); e.target.textContent = 'Copied!'; setTimeout(() => { e.target.textContent = 'Copy'; }, 1500); }
    }
  });

  $('structuredDebugLink').addEventListener('click', async () => {
    const contentEl = $('structuredDebugContent');
    if (contentEl.style.display !== 'none') { contentEl.style.display = 'none'; return; }
    const all = await chrome.storage.local.get(null);
    const structuredKeys = Object.keys(all).filter((k) => k.startsWith('structured:')).sort();
    if (structuredKeys.length === 0) {
      contentEl.innerHTML = '<div style="font-size:11px;color:#999;margin-top:6px;">No structured data cached</div>';
    } else {
      contentEl.innerHTML = structuredKeys.map((k) => {
        const v = all[k];
        const dt = v.timestamp ? formatTime(v.timestamp) : '?';
        return '<details style="font-size:11px;margin:4px 0;"><summary style="cursor:pointer;color:#555;">' + escapeHtml(k) + ' . ' + dt + '</summary><pre style="background:#f8f8f8;padding:8px;overflow-x:auto;font-size:11px;max-height:200px;">' + escapeHtml(JSON.stringify(v.data, null, 2)) + '</pre></details>';
      }).join('');
    }
    contentEl.style.display = 'block';
  });
});
