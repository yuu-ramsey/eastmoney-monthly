// viewer page: read cache key from URL params, display corresponding historical analysis

const params = new URLSearchParams(location.search);
const cacheKey = params.get('key');

const metaEl = document.getElementById('meta');
const contentEl = document.getElementById('content');
const closeBtn = document.getElementById('closeBtn');

closeBtn.addEventListener('click', () => window.close());

(async () => {
  if (!cacheKey) {
    contentEl.innerHTML = '<div class="error">Missing key parameter</div>';
    return;
  }

  const items = await chrome.storage.local.get([cacheKey]);
  const record = items[cacheKey];

  if (!record) {
    contentEl.innerHTML = '<div class="error">Cache not found</div>';
    return;
  }

  document.title = `${record.name}(${record.code}) Monthly Analysis - ${record.monthKey}`;

  const dtStr = record.analyzedAt ? formatTime(record.analyzedAt) : '-';
  const periodLabel = PERIOD_LABELS[record.period] || '';
  const styleLabel = STYLE_LABELS[record.style] || '';
  const providerName = PROVIDER_NAMES[record.provider] || (record.provider || '');
  const modelText = providerName ? `${providerName}(${escapeHtml(record.model || '')})` : escapeHtml(record.model || '');
  const costText = (typeof record.cost === 'number' && record.cost > 0) ? ` · ${formatCost(record.cost)}` : '';
  const decisionBadge = record.decision === 'on' ? ' · Decision Mode' : '';
  metaEl.innerHTML = `
    <span class="name">${escapeHtml(record.name)}(${escapeHtml(record.code)})</span>
    <span class="secondary">${periodLabel} ${escapeHtml(record.monthKey || record.bucket || '')} · ${escapeHtml(dtStr)} · ${modelText}${styleLabel ? ' · ' + styleLabel : ''}${costText}${decisionBadge}</span>
  `;

  // cross-level consistency warning banner
  let html = '';
  if (record.crossLevelWarnings && Array.isArray(record.crossLevelWarnings) && record.crossLevelWarnings.length > 0) {
    html += `<div class="cross-level-warnings">
      <h4>Cross-Level Consistency Notice</h4>
      <ul>${record.crossLevelWarnings.map((w) => '<li>' + escapeHtml(String(w)) + '</li>').join('')}</ul>
      <p class="hint">This is a tool-generated comparison against the last higher-level analysis. It does not necessarily mean the current analysis is wrong — use your own judgment.</p>
    </div>`;
  }

  // main output
  html += renderMarkdown(record.analysis || '');

  // debate mode: append collapsible details
  if (record.debate) {
    const d = record.debate;
    const totalMs = d.totalDurationMs ? `Total duration ${(d.totalDurationMs / 1000).toFixed(1)}s` : '';
    const totalCost = d.totalCost ? `Total cost ${formatCost(d.totalCost)}` : '';
    html += `
<details class="debate-details">
<summary>View Debate Details (Bull/Bear/Predictor raw output)${totalMs ? ' · ' + totalMs : ''}${totalCost ? ' · ' + totalCost : ''}</summary>
<div class="debate-tabs">
  <button class="debate-tab active" data-tab="bull">Bullish</button>
  <button class="debate-tab" data-tab="bear">Bearish</button>
  <button class="debate-tab" data-tab="predictor">Levels</button>
</div>
<div class="debate-panel active" data-panel="bull">${renderAgentOutput(d.partials?.bull, 'bull')}</div>
<div class="debate-panel" data-panel="bear">${renderAgentOutput(d.partials?.bear, 'bear')}</div>
<div class="debate-panel" data-panel="predictor">${renderAgentOutput(d.partials?.predictor, 'predictor')}</div>
</details>`;
  }

  contentEl.innerHTML = html;

  // bind tab switching
  contentEl.querySelectorAll('.debate-tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      const role = tab.dataset.tab;
      contentEl.querySelectorAll('.debate-tab').forEach((t) => t.classList.toggle('active', t.dataset.tab === role));
      contentEl.querySelectorAll('.debate-panel').forEach((p) => p.classList.toggle('active', p.dataset.panel === role));
    });
  });
})();

const PERIOD_LABELS = { monthly: 'Monthly', weekly: 'Weekly', daily: 'Daily', multi: 'Multi-Period' };
const STYLE_LABELS = { technical: 'Technical Analysis', chanlun: 'Chanlun', value: 'Value Perspective', comprehensive: 'Comprehensive' };
const PROVIDER_NAMES = { anthropic: 'Claude', deepseek: 'DeepSeek' };

function renderAgentOutput(agentResult, role) {
  if (!agentResult) return `<div class="error">${role} Agent: no output</div>`;
  const costInfo = (typeof agentResult.cost === 'number' && agentResult.cost > 0) ? ` · Cost ${formatCost(agentResult.cost)}` : '';
  const timeInfo = agentResult.durationMs ? ` · Duration ${(agentResult.durationMs / 1000).toFixed(1)}s` : '';
  const tokenInfo = agentResult.usage ? ` · ${agentResult.usage.inputTokens}+${agentResult.usage.outputTokens} tokens` : '';
  return `<div class="agent-meta">${costInfo}${timeInfo}${tokenInfo}</div>${renderMarkdown(agentResult.text || '')}`;
}

function formatCost(cost) {
  if (cost < 0.01) return '<¥0.01';
  if (cost < 1) return `¥${cost.toFixed(2)}`;
  return `¥${cost.toFixed(2)}`;
}

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

// minimal Markdown rendering (same as content.js, inlined to avoid content script module complexity)
function renderMarkdown(md) {
  const lines = String(md).split('\n');
  const out = [];
  let inList = false;
  const closeList = () => { if (inList) { out.push('</ul>'); inList = false; } };

  for (const raw of lines) {
    const h = raw.match(/^(#{1,6})\s+(.*)$/);
    if (h) {
      closeList();
      const level = Math.min(h[1].length, 4);
      out.push(`<h${level}>${formatInline(h[2])}</h${level}>`);
      continue;
    }
    const li = raw.match(/^[-*]\s+(.*)$/);
    if (li) {
      if (!inList) { out.push('<ul>'); inList = true; }
      out.push(`<li>${formatInline(li[1])}</li>`);
      continue;
    }
    if (raw.trim() === '') { closeList(); continue; }
    closeList();
    out.push(`<p>${formatInline(raw)}</p>`);
  }
  closeList();
  return out.join('');
}

function formatInline(s) {
  let x = escapeHtml(s);
  x = x.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  x = x.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, '$1<em>$2</em>');
  x = x.replace(/`([^`]+)`/g, '<code>$1</code>');
  return x;
}
