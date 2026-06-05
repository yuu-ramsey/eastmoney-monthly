// content script: inject Shadow DOM side panel, manage content-side cache, monitor URL changes to auto-trigger analysis

(function () {
  if (window.__monthlyAIInjected) return;
  window.__monthlyAIInjected = true;

  // ---------- create host + shadow root ----------
  const host = document.createElement('div');
  host.id = 'monthly-ai-host';
  document.documentElement.appendChild(host);
  const root = host.attachShadow({ mode: 'closed' });

  const link = document.createElement('link');
  link.rel = 'stylesheet';
  link.href = chrome.runtime.getURL('content.css');
  root.appendChild(link);

  // ---------- DOM structure ----------
  const wrapper = document.createElement('div');
  wrapper.innerHTML = `
    <button class="fab" data-role="fab" title="AI Analysis" type="button">AI</button>
    <div class="panel" data-role="panel">
      <div class="panel-header">
        <div class="title" data-role="title">AI Analysis</div>
        <div class="actions">
          <button class="reanalyze" data-role="reanalyze" type="button" title="Re-analyze (consumes tokens)" style="display:none;">Re-analyze</button>
          <button class="close" data-role="close" type="button" title="Close">x</button>
        </div>
      </div>
      <div class="panel-overview" data-role="overview" style="display:none;">
        <div class="overview-left">
          <span class="ov-name" data-role="ovName"></span>
          <span class="ov-code" data-role="ovCode"></span>
        </div>
        <div class="overview-right">
          <span class="ov-price" data-role="ovPrice"></span>
          <span class="ov-percentile" data-role="ovPercentile"></span>
        </div>
      </div>
      <div class="panel-meta" data-role="meta" style="display:none;"></div>
      <div class="dashboard-card" data-role="dashboardCard" style="display:none;">
        <div class="dashboard-main">
          <div class="dashboard-score">
            <div class="score-number" data-role="scoreNumber">--</div>
            <div class="score-label">Score</div>
          </div>
          <div class="dashboard-signal" data-role="signalBadge"></div>
          <div class="dashboard-confidence" data-role="confidenceBadge"></div>
        </div>
        <div class="dashboard-summary" data-role="summaryText"></div>
        <div class="dashboard-levels">
          <div class="level-item"><span class="level-label">Support</span><span class="level-values" data-role="supportLevels"></span></div>
          <div class="level-item"><span class="level-label">Resist</span><span class="level-values" data-role="resistanceLevels"></span></div>
          <div class="level-item"><span class="level-label">Stop</span><span class="level-values" data-role="stopLoss"></span></div>
        </div>
        <div class="dashboard-meta">
          <span data-role="positionPercentile"></span>
          <span data-role="trendLabel"></span>
        </div>
      </div>
      <div class="panel-body" data-role="body">
        <div class="hint">Loading...</div>
      </div>
      <div class="thinking-stream" data-role="thinkingStream" style="display:none;">
        <div class="thinking-header" data-role="thinkingHeader">
          <span class="thinking-icon">Brain</span>
          <span class="thinking-status" data-role="thinkingStatus">AI is analyzing...</span>
          <button class="thinking-toggle" data-role="thinkingToggle">Collapse</button>
        </div>
        <div class="thinking-body" data-role="thinkingBody"></div>
      </div>
      <div class="chat-area" data-role="chatArea" style="display:none;">
        <div class="chat-messages" data-role="chatMessages"></div>
        <div class="chat-warning" data-role="chatWarning" style="display:none;"></div>
        <div class="chat-input-row">
          <input type="text" data-role="chatInput" placeholder="Ask a follow-up based on current analysis..." autocomplete="off">
          <button data-role="chatSend" type="button">Send</button>
          <button data-role="chatClear" type="button">Clear</button>
          <button data-role="chatSave" type="button" title="Save conversation to history">Save</button>
        </div>
      </div>
    </div>
  `;
  root.appendChild(wrapper);

  const fabEl = root.querySelector('[data-role="fab"]');
  const panelEl = root.querySelector('[data-role="panel"]');
  const titleEl = root.querySelector('[data-role="title"]');
  const bodyEl = root.querySelector('[data-role="body"]');
  const metaEl = root.querySelector('[data-role="meta"]');
  const closeEl = root.querySelector('[data-role="close"]');
  const reanalyzeEl = root.querySelector('[data-role="reanalyze"]');
  const overviewEl = root.querySelector('[data-role="overview"]');
  const ovNameEl = root.querySelector('[data-role="ovName"]');
  const ovCodeEl = root.querySelector('[data-role="ovCode"]');
  const ovPriceEl = root.querySelector('[data-role="ovPrice"]');
  const ovPercentileEl = root.querySelector('[data-role="ovPercentile"]');

  const chatAreaEl = root.querySelector('[data-role="chatArea"]');
  const chatMessagesEl = root.querySelector('[data-role="chatMessages"]');
  const chatWarningEl = root.querySelector('[data-role="chatWarning"]');
  const chatInputEl = root.querySelector('[data-role="chatInput"]');
  const chatSendEl = root.querySelector('[data-role="chatSend"]');
  const chatClearEl = root.querySelector('[data-role="chatClear"]');
  const chatSaveEl = root.querySelector('[data-role="chatSave"]');
  const thinkingStreamEl = root.querySelector('[data-role="thinkingStream"]');
  const dashboardCardEl = root.querySelector('[data-role="dashboardCard"]');
  const scoreNumberEl = root.querySelector('[data-role="scoreNumber"]');
  const signalBadgeEl = root.querySelector('[data-role="signalBadge"]');
  const confidenceBadgeEl = root.querySelector('[data-role="confidenceBadge"]');
  const summaryTextEl = root.querySelector('[data-role="summaryText"]');
  const supportLevelsEl = root.querySelector('[data-role="supportLevels"]');
  const resistanceLevelsEl = root.querySelector('[data-role="resistanceLevels"]');
  const stopLossEl = root.querySelector('[data-role="stopLoss"]');
  const positionPercentileEl = root.querySelector('[data-role="positionPercentile"]');
  const trendLabelEl = root.querySelector('[data-role="trendLabel"]');
  const thinkingBodyEl = root.querySelector('[data-role="thinkingBody"]');
  const thinkingStatusEl = root.querySelector('[data-role="thinkingStatus"]');
  const thinkingToggleEl = root.querySelector('[data-role="thinkingToggle"]');

  closeEl.addEventListener('click', () => panelEl.classList.remove('open'));
  fabEl.addEventListener('click', () => {
    if (!panelEl.classList.contains('open')) {
      panelEl.classList.add('open');
      analyze(false);
    }
  });
  reanalyzeEl.addEventListener('click', () => {
    if (!confirm('Re-analyzing will consume tokens. Are you sure?')) return;
    analyze(true);
  });

  let busy = false;
  let lastAnalyzedCode = ''; // for URL change detection
  let conversationHistory = [];
  let currentAnalysisId = null; // history record ID of current analysis

  // ---------- thinking stream UI ----------

  thinkingToggleEl.addEventListener('click', () => {
    const body = thinkingBodyEl;
    if (body.style.display === 'none') {
      body.style.display = 'block';
      thinkingToggleEl.textContent = 'Collapse';
    } else {
      body.style.display = 'none';
      thinkingToggleEl.textContent = 'Expand';
    }
  });

  function showThinkingStream() {
    thinkingStreamEl.style.display = 'block';
    thinkingBodyEl.innerHTML = '';
    thinkingBodyEl.style.display = 'block';
    thinkingStatusEl.textContent = 'AI is analyzing...';
    thinkingToggleEl.textContent = 'Collapse';
  }

  function appendThinkingLine(text, cls) {
    const el = document.createElement('div');
    el.className = cls || 'stream-text';
    el.textContent = text;
    thinkingBodyEl.appendChild(el);
    thinkingBodyEl.scrollTop = thinkingBodyEl.scrollHeight;
  }

  function hideThinkingStream() {
    thinkingStatusEl.textContent = 'Analysis complete';
    thinkingBodyEl.style.display = 'none';
    thinkingToggleEl.textContent = 'Expand';
  }

  // listen for STREAM_PROGRESS from background
  chrome.runtime.onMessage.addListener((msg) => {
    if (msg?.type !== 'STREAM_PROGRESS' || !msg.event) return;
    const { type, text, name, input, result } = msg.event;

    switch (type) {
      case 'thinking':
        appendThinkingLine(text || '', 'stream-thinking');
        break;
      case 'text':
        appendThinkingLine(text || '', 'stream-text');
        break;
      case 'tool_start':
        appendThinkingLine(`Calling ${name}(${input})...`, 'stream-tool');
        break;
      case 'tool_result':
        // update the corresponding tool call line
        updateLastToolLine(name, result);
        break;
      case 'done':
        hideThinkingStream();
        break;
    }
  });

  function updateLastToolLine(toolName, result) {
    const lines = thinkingBodyEl.querySelectorAll('.stream-tool');
    for (let i = lines.length - 1; i >= 0; i--) {
      if (lines[i].textContent.includes(toolName)) {
        const summary = String(result || '').slice(0, 80);
        lines[i].textContent = `${toolName} returned: ${summary}${result.length > 80 ? '...' : ''}`;
        lines[i].className = 'stream-tool-done';
        break;
      }
    }
  }

  // ---------- extract current price from page ----------
  function extractCurrentPrice() {
    const selectors = [
      '.cur_price', '.price', '#price9', '.stock_price .cur',
      '[class*="price" i] span', '.zxj span', '.spn_price',
    ];
    for (const sel of selectors) {
      try {
        const el = document.querySelector(sel);
        if (!el) continue;
        const text = (el.textContent || '').trim();
        const num = parseFloat(text);
        if (!isNaN(num) && num > 0) return num;
      } catch (_) { /* skip */ }
    }
    // try parsing from page title
    const title = document.title;
    const match = title.match(/(\d+\.\d{2})/);
    if (match) return parseFloat(match[1]);
    return null;
  }

  // ---------- extract position percentile from analysis text ----------
  function extractPositionPercentile(analysisText) {
    // match patterns like "XX percentile" or "XX% position"
    const patterns = [
      /(\d+(?:\.\d+)?)\s*percentile/i,
      /percentile[：:]\s*(\d+(?:\.\d+)?)/i,
      /at\s+(\d+(?:\.\d+)?)\s*%/,
      /position[^\d]*(\d+(?:\.\d+)?)\s*percentile/i,
    ];
    for (const p of patterns) {
      const m = analysisText.match(p);
      if (m) return parseFloat(m[1]);
    }
    return null;
  }

  // ---------- content-side cache ----------
  function getCacheKey(code, template) {
    const monthKey = new Date().toISOString().slice(0, 7); // YYYY-MM
    const tpl = template || 'unknown';
    return `contentCache:${code}:${monthKey}:${tpl}`;
  }

  async function getContentCache(code, template) {
    const key = getCacheKey(code, template);
    await cleanExpiredCache();
    const items = await chrome.storage.local.get([key]);
    const entry = items[key];
    if (!entry) return null;
    // 30 day expiry
    if (Date.now() - entry.timestamp > 30 * 24 * 60 * 60 * 1000) {
      await chrome.storage.local.remove([key]);
      return null;
    }
    return entry;
  }

  async function setContentCache(code, template, data) {
    const key = getCacheKey(code, template);
    await chrome.storage.local.set({
      [key]: { ...data, timestamp: Date.now() },
    });
  }

  async function deleteContentCache(code, template) {
    const key = getCacheKey(code, template);
    await chrome.storage.local.remove([key]);
  }

  async function cleanExpiredCache() {
    const EXPIRY = 30 * 24 * 60 * 60 * 1000;
    const now = Date.now();
    const all = await chrome.storage.local.get(null);
    const expiredKeys = [];
    for (const [k, v] of Object.entries(all)) {
      if (k.startsWith('contentCache:') && v && v.timestamp && (now - v.timestamp > EXPIRY)) {
        expiredKeys.push(k);
      }
    }
    if (expiredKeys.length > 0) {
      await chrome.storage.local.remove(expiredKeys);
    }
  }

  // ---------- scrape events from page ----------
  function collectPageEvents() {
    const rows = document.querySelectorAll('.siderstockcalendarcontent table tbody tr');
    const events = [];
    for (const row of rows) {
      try {
        const timeEl = row.querySelector('.time');
        const eventEl = row.querySelector('.event');
        if (!timeEl || !eventEl) continue;
        const dateText = (timeEl.textContent || '').trim();
        if (!dateText) continue;
        const date = dateText.slice(0, 5);
        const firstChild = eventEl.firstElementChild;
        let type, rawTitle;
        if (firstChild) {
          type = (firstChild.textContent || '').trim();
          rawTitle = (eventEl.textContent || '').replace(firstChild.textContent || '', '').trim();
        } else {
          const text = (eventEl.textContent || '').trim();
          const idx = text.search(/[\s　]/);
          if (idx > 0) {
            type = text.slice(0, idx);
            rawTitle = text.slice(idx + 1).trim();
          } else {
            type = text;
            rawTitle = '';
          }
        }
        const hadDatePrefix = /^\d{4}年\d{2}月\d{2}日/.test(rawTitle);
        let title = rawTitle
          .replace(/^\d{4}年\d{2}月\d{2}日(?:发布)?[《【]?|[》】]/g, '')
          .replace(/^截止\d{4}年\d{2}月\d{2}日\s*/g, '')
          .trim();
        if (hadDatePrefix && type && title.endsWith(type)) title = title.slice(0, -type.length).trim();
        if (!title) title = rawTitle;
        events.push({ date, type, title });
      } catch (_) { /* skip */ }
    }
    return events.slice(0, 10);
  }

  // ---------- parse stock code ----------
  function parseCodeFromUrl(url) {
    // Eastmoney URL actual format: quote.eastmoney.com/sh600519.html (no slash separator)
    const m = (url || location.href).match(/\b(sh|sz|bj)\.?(\d{4,6})\b/i);
    if (m) return m[2];
    return null;
  }

  // ---------- main analysis flow ----------
  async function analyze(force) {
    if (busy) return;
    busy = true;
    fabEl.disabled = true;
    reanalyzeEl.disabled = true;
    panelEl.classList.add('open');
    titleEl.textContent = 'AI Analysis';
    overviewEl.style.display = 'none';
    metaEl.style.display = 'none';
    reanalyzeEl.style.display = 'none';
    showThinkingStream();
    setBody('<div class="loading">Loading...</div>');

    try {
      const code = parseCodeFromUrl();
      if (!code) {
        setBody('<div class="error">Unable to parse stock code from current page</div>');
        return;
      }
      lastAnalyzedCode = code;

      // 1. get current template setting (for cache key)
      const settingsItems = await chrome.storage.local.get(['template']);
      const currentTemplate = settingsItems.template || 'technical';

      // 2. check content-side cache first (non-force mode)
      if (!force) {
        const cached = await getContentCache(code, currentTemplate);
        if (cached?.analysis) {
          renderResult({ ...cached, cached: true });
          return;
        }
      }

      // 3. call SW
      const pageEvents = collectPageEvents();
      const resp = await chrome.runtime.sendMessage({
        type: 'ANALYZE',
        url: location.href,
        force,
        pageEvents,
      });
      if (!resp) {
        setBody('<div class="error">Service worker is not responding. The extension may not be loaded.</div>');
        return;
      }
      if (!resp.ok) {
        setBody(`<div class="error">${escapeHtml(resp.error || 'Unknown error')}</div>`);
        return;
      }

      // 4. write to content-side cache
      await setContentCache(code, currentTemplate, {
        code,
        name: resp.name,
        analysis: resp.analysis,
        period: resp.period,
        style: resp.style,
        template: resp.template || currentTemplate,
        provider: resp.provider,
        model: resp.model,
        mode: resp.mode,
        decision: resp.decision,
        prompt: resp.prompt,
        analyzedAt: resp.analyzedAt || Date.now(),
      });

      renderResult({ ...resp, cached: false });
    } catch (err) {
      setBody(`<div class="error">Communication error: ${escapeHtml(err.message || String(err))}</div>`);
    } finally {
      hideThinkingStream();
      busy = false;
      fabEl.disabled = false;
      reanalyzeEl.disabled = false;
    }
  }

  // ---------- dashboard JSON parsing ----------

  function parseDashboardJSON(text) {
    if (!text) return null;
    const blocks = [];
    const re = /```json\s*([\s\S]*?)```/g;
    let m;
    while ((m = re.exec(text)) !== null) blocks.push(m[1].trim());
    if (blocks.length === 0) return null;
    try { return JSON.parse(blocks[blocks.length - 1]); } catch (_) { return null; }
  }

  function renderDashboard(scoreData) {
    if (!scoreData) {
      dashboardCardEl.style.display = 'block';
      scoreNumberEl.textContent = '?';
      scoreNumberEl.style.color = '#888';
      signalBadgeEl.textContent = 'No structured data generated for this analysis — text only';
      signalBadgeEl.style.background = '#fff3cd';
      confidenceBadgeEl.textContent = '';
      summaryTextEl.textContent = '';
      supportLevelsEl.textContent = '';
      resistanceLevelsEl.textContent = '';
      stopLossEl.textContent = '';
      positionPercentileEl.textContent = '';
      trendLabelEl.textContent = '';
      return;
    }

    dashboardCardEl.style.display = 'block';
    const s = scoreData;

    // score number coloring
    const colors = { strong_bull: '#00aa44', bull: '#4ec77b', neutral: '#888', bear: '#e88880', strong_bear: '#cc0000' };
    scoreNumberEl.textContent = s.score != null ? Math.round(s.score) : '?';
    scoreNumberEl.style.color = colors[s.signal] || '#888';

    // signal
    const labels = { strong_bull: 'Strong Bull', bull: 'Bull', neutral: 'Neutral', bear: 'Bear', strong_bear: 'Strong Bear' };
    signalBadgeEl.textContent = labels[s.signal] || s.signal || '?';
    signalBadgeEl.style.background = '';

    // confidence
    const confLabels = { high: 'Confidence: High', medium: 'Confidence: Medium', low: 'Confidence: Low' };
    const confBg = { high: '#d4edda', medium: '#fff3cd', low: '#f0f0f0' };
    confidenceBadgeEl.textContent = confLabels[s.confidence] || '';
    confidenceBadgeEl.style.background = confBg[s.confidence] || '';

    // summary
    summaryTextEl.textContent = s.one_line_summary || '';

    // key_levels
    const kl = s.key_levels || {};
    supportLevelsEl.textContent = (kl.support || []).map(function(v) { return v.toFixed(2); }).join(' / ') || '-';
    resistanceLevelsEl.textContent = (kl.resistance || []).map(function(v) { return v.toFixed(2); }).join(' / ') || '-';
    if (kl.stop_loss != null) {
      stopLossEl.textContent = kl.stop_loss.toFixed(2);
      stopLossEl.style.color = '#cc0000';
      stopLossEl.style.fontWeight = 'bold';
    } else {
      stopLossEl.textContent = '-';
      stopLossEl.style.color = '';
    }

    // position_percentile
    if (s.position_percentile != null) {
      var pp = s.position_percentile;
      positionPercentileEl.textContent = 'Pos ' + pp.toFixed(1) + '%';
      positionPercentileEl.style.background = pp < 30 ? '#d4edda' : pp > 70 ? '#f8d7da' : '#f0f0f0';
      positionPercentileEl.style.padding = '2px 6px';
      positionPercentileEl.style.borderRadius = '3px';
    }

    // trend
    var trendMap = { uptrend: 'Uptrend', downtrend: 'Downtrend', sideways: 'Sideways', reversing: 'Reversing' };
    trendLabelEl.textContent = trendMap[s.trend] || s.trend || '';
  }

  // ---------- render result ----------
  function renderResult(resp) {
    titleEl.textContent = `${escapeHtml(resp.name)} (${escapeHtml(resp.code)})`;
    reanalyzeEl.style.display = 'inline-block';

    // meta info bar
    const dtStr = resp.analyzedAt ? formatTime(resp.analyzedAt) : '';
    const badge = resp.cached
      ? '<span class="badge badge-cache">Cached</span>'
      : '<span class="badge badge-fresh">Fresh</span>';
    const periodLabel = PERIOD_LABELS[resp.period] || '';
    const styleLabel = STYLE_LABELS[resp.style] || '';
    const providerLabel = PROVIDER_LABELS[resp.provider] || (resp.provider || '');
    const dbBadge = resp.mode === 'debate' ? ' . Debate' : '';
    const dcBadge = resp.decision === 'on' ? ' . Decision' : '';
    metaEl.innerHTML = `${badge} ${periodLabel} ${escapeHtml(resp.monthKey || resp.bucket || '')} . ${dtStr} . ${providerLabel}(${escapeHtml(resp.model || '')})${styleLabel ? ' . ' + styleLabel : ''}${dbBadge}${dcBadge}`;
    metaEl.style.display = 'block';

    // overview bar: stock name + code + current price + position percentile
    const currentPrice = extractCurrentPrice();
    const percentile = extractPositionPercentile(resp.analysis || '');
    ovNameEl.textContent = resp.name || '';
    ovCodeEl.textContent = resp.code || '';
    ovPriceEl.textContent = currentPrice !== null ? `¥${currentPrice.toFixed(2)}` : '';
    ovPercentileEl.textContent = percentile !== null ? `P${Math.round(percentile)}` : '';
    overviewEl.style.display = 'flex';

    // dashboard
    const scoreData = parseDashboardJSON(resp.analysis || '');
    renderDashboard(scoreData);

    // card rendering
    setBody(renderCards(resp.analysis || ''));

    // record history ID for later conversation save
    currentAnalysisId = resp.historyId || null;

    // initialize conversation history
    const firstUserMsg = resp.prompt
      || `Analyze ${resp.name || ''}(${resp.code || ''}) ${PERIOD_LABELS[resp.period] || resp.period || ''}`;
    conversationHistory = [
      { role: 'user', content: firstUserMsg },
      { role: 'assistant', content: resp.analysis || '' },
    ];

    // show conversation area
    chatAreaEl.style.display = 'flex';
    chatWarningEl.style.display = 'none';
    chatInputEl.value = '';
    renderChatMessages();
  }

  // ---------- card foldable rendering ----------
  function renderCards(md) {
    const sections = parseSections(md);
    if (sections.length === 0) {
      return `<div class="analysis">${renderMarkdown(md)}</div>`;
    }

    return sections.map((sec) => {
      const category = classifySection(sec.title);
      const isExpanded = (category === 'position' || category === 'ma' || category === 'price' || category === 'conclusion');
      const icon = isExpanded ? 'v' : '>';
      const expandedClass = isExpanded ? 'expanded' : '';
      const bodyClass = isExpanded ? '' : 'collapsed';

      return `
        <div class="card">
          <div class="card-header ${expandedClass}" data-card-toggle>
            <span class="card-icon">${icon}</span>
            <span class="card-title">${escapeHtml(sec.title)}</span>
            ${category !== '' ? `<span class="card-tag">${CATEGORY_LABELS[category] || ''}</span>` : ''}
          </div>
          <div class="card-body ${bodyClass}">
            ${renderMarkdown(sec.body)}
          </div>
        </div>
      `;
    }).join('');
  }

  // split Markdown by ## / ### headings
  function parseSections(md) {
    const lines = String(md).split('\n');
    const sections = [];
    let currentTitle = '';
    let currentLines = [];
    let skipJson = false;

    for (const line of lines) {
      // inside a JSON skip block, wait for closing ```
      if (skipJson) {
        if (line.trim() === '```') skipJson = false;
        continue;
      }

      const h2 = line.match(/^##\s+(.*)$/);
      const h3 = line.match(/^###\s+(.*)$/);
      const h1 = line.match(/^#\s+(.*)$/);

      if (h1 || h2 || h3) {
        // save previous section
        const bodyText = currentLines.join('\n').trim();
        if (currentTitle || bodyText) {
          sections.push({ title: currentTitle || 'Overview', body: bodyText });
        }
        currentTitle = (h1 || h2 || h3)[1].trim();
        currentLines = [];
        continue;
      }

      // skip structured JSON block (only effective at start of new section)
      if (currentLines.length === 0 && line.trim() === '```json') {
        skipJson = true;
        continue;
      }

      currentLines.push(line);
    }

    // last section
    const bodyText = currentLines.join('\n').trim();
    if (currentTitle || bodyText) {
      sections.push({ title: currentTitle || 'Overview', body: bodyText });
    }

    return sections;
  }

  function classifySection(title) {
    const t = title.toLowerCase();
    if (/position|percentile|range|valuation/i.test(t)) return 'position';
    if (/ma|ema|moving average/i.test(t)) return 'ma';
    if (/support|resistance|level|central hub|price|target|stop|entry|exit/i.test(t)) return 'price';
    if (/trend|direction|momentum/i.test(t)) return 'trend';
    if (/risk|counter|downside|bear case/i.test(t)) return 'counter';
    if (/action|advice|position sizing|entry|strategy/i.test(t)) return 'action';
    if (/conclusion|summary|verdict/i.test(t)) return 'conclusion';
    return '';
  }

  const CATEGORY_LABELS = {
    position: 'Position', ma: 'MA', price: 'Levels',
    trend: 'Trend', counter: 'Risks', action: 'Action', conclusion: 'Conclusion',
  };

  // card fold toggle click
  root.addEventListener('click', (e) => {
    const header = e.target.closest('[data-card-toggle]');
    if (!header) return;
    const card = header.parentElement;
    const body = card.querySelector('.card-body');
    const icon = header.querySelector('.card-icon');
    if (body.classList.contains('collapsed')) {
      body.classList.remove('collapsed');
      header.classList.add('expanded');
      icon.textContent = 'v';
    } else {
      body.classList.add('collapsed');
      header.classList.remove('expanded');
      icon.textContent = '>';
    }
  });

  // ---------- Markdown rendering ----------
  function setBody(html) { bodyEl.innerHTML = html; }

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

  const STYLE_LABELS = { technical: 'Technical', chanlun: 'Chanlun', value: 'Value', comprehensive: 'Comprehensive' };
  const PERIOD_LABELS = { monthly: 'Monthly', weekly: 'Weekly', daily: 'Daily', multi: 'Multi' };
  const PROVIDER_LABELS = { anthropic: 'Claude', deepseek: 'DeepSeek' };

  function renderMarkdown(md) {
    const lines = String(md).split('\n');
    const out = [];
    let inList = false;
    const closeList = () => { if (inList) { out.push('</ul>'); inList = false; } };

    for (const raw of lines) {
      // skip bare ``` markers
      if (raw.trim() === '```' || raw.trim() === '```json') continue;
      const h = raw.match(/^(#{1,6})\s+(.*)$/);
      if (h) {
        closeList();
        const level = Math.min(h[1].length + 2, 5); // start from h3 inside cards
        out.push(`<h${level}>${formatInline(h[2])}</h${level}>`);
        continue;
      }
      const li = raw.match(/^[-*]\s+(.*)$/);
      if (li) {
        if (!inList) { out.push('<ul>'); inList = true; }
        out.push(`<li>${formatInline(li[1])}</li>`);
        continue;
      }
      // numbered list
      const nli = raw.match(/^\d+[\.\)]\s+(.*)$/);
      if (nli) {
        if (!inList) { out.push('<ul>'); inList = true; }
        out.push(`<li>${formatInline(nli[1])}</li>`);
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

  // ---------- multi-turn conversation ----------

  function renderChatMessages() {
    if (conversationHistory.length === 0) {
      chatMessagesEl.innerHTML = '';
      return;
    }
    chatMessagesEl.innerHTML = conversationHistory.map((msg, i) => {
      // skip first prompt (too long, don't render)
      if (i === 0 && msg.role === 'user') {
        return `<div class="chat-msg chat-msg-user chat-msg-first">
          <div class="chat-bubble">Initial analysis prompt (submitted, length ${msg.content.length} chars)</div>
        </div>`;
      }
      const cls = msg.role === 'user' ? 'chat-msg-user' : 'chat-msg-assistant';
      return `<div class="chat-msg ${cls}">
        <div class="chat-bubble">${renderMarkdown(msg.content)}</div>
      </div>`;
    }).join('');
    chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;

    // warning for 20+ messages
    if (conversationHistory.length >= 20) {
      chatWarningEl.style.display = 'block';
      chatWarningEl.textContent = `Conversation has reached ${conversationHistory.length} messages. Token usage is high. Consider clearing and restarting.`;
    } else {
      chatWarningEl.style.display = 'none';
    }
  }

  async function sendFollowUp() {
    const question = chatInputEl.value.trim();
    if (!question || busy) return;
    chatInputEl.value = '';
    busy = true;
    chatSendEl.disabled = true;

    // render user message
    conversationHistory.push({ role: 'user', content: question });
    renderChatMessages();

    // add loading placeholder
    const loadingIdx = conversationHistory.length;
    conversationHistory.push({ role: 'assistant', content: 'Thinking...' });
    renderChatMessages();

    try {
      const resp = await chrome.runtime.sendMessage({
        type: 'FOLLOW_UP',
        question,
        history: conversationHistory.filter((m) => m.content !== 'Thinking...').slice(0, -1),
      });
      // remove loading placeholder
      conversationHistory.splice(loadingIdx, 1);

      if (!resp || !resp.ok) {
        conversationHistory.push({ role: 'assistant', content: `Error: ${resp?.error || 'No response'}` });
      } else {
        conversationHistory.push({ role: 'assistant', content: resp.text });
      }
    } catch (err) {
      conversationHistory.splice(loadingIdx, 1);
      conversationHistory.push({ role: 'assistant', content: `Communication error: ${err.message || String(err)}` });
    }

    renderChatMessages();
    busy = false;
    chatSendEl.disabled = false;
    chatInputEl.focus();
  }

  function clearChat() {
    if (conversationHistory.length <= 2) return;
    // keep first two messages (initial prompt + initial analysis)
    conversationHistory = conversationHistory.slice(0, 2);
    renderChatMessages();
    chatInputEl.focus();
  }

  // save conversation to history
  async function saveConversationToHistory() {
    if (!currentAnalysisId) {
      chatSaveEl.textContent = 'x';
      setTimeout(() => { chatSaveEl.textContent = 'Save'; }, 1500);
      return;
    }
    chatSaveEl.disabled = true;
    chatSaveEl.textContent = '...';
    try {
      const resp = await chrome.runtime.sendMessage({
        type: 'SAVE_CONVERSATION',
        id: currentAnalysisId,
        conversationHistory,
      });
      if (resp && resp.ok) {
        chatSaveEl.textContent = 'OK';
      } else {
        chatSaveEl.textContent = 'ERR';
      }
    } catch (_) {
      chatSaveEl.textContent = 'ERR';
    }
    setTimeout(() => {
      chatSaveEl.textContent = 'Save';
      chatSaveEl.disabled = false;
    }, 1500);
  }

  // conversation events
  chatSendEl.addEventListener('click', sendFollowUp);
  chatClearEl.addEventListener('click', clearChat);
  chatSaveEl.addEventListener('click', saveConversationToHistory);
  chatInputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendFollowUp();
    }
  });

  // ---------- URL change monitoring ----------
  let lastUrl = location.href;

  function onUrlChange() {
    if (location.href === lastUrl) return;
    lastUrl = location.href;
    const newCode = parseCodeFromUrl();
    if (newCode && newCode !== lastAnalyzedCode) {
      analyze(false);
    }
  }

  window.addEventListener('popstate', onUrlChange);

  // monkey-patch pushState / replaceState
  const _pushState = history.pushState;
  history.pushState = function (...args) {
    _pushState.apply(this, args);
    onUrlChange();
  };
  const _replaceState = history.replaceState;
  history.replaceState = function (...args) {
    _replaceState.apply(this, args);
    onUrlChange();
  };

  // ---------- auto-trigger ----------
  // auto-analyze after page loads
  if (document.readyState === 'complete') {
    setTimeout(() => analyze(false), 800);
  } else {
    window.addEventListener('load', () => setTimeout(() => analyze(false), 800));
  }
})();
