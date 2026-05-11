// content script: 注入 Shadow DOM 侧边面板,管理内容侧缓存,监听 URL 变化自动触发分析

(function () {
  if (window.__monthlyAIInjected) return;
  window.__monthlyAIInjected = true;

  // ---------- 创建 host + shadow root ----------
  const host = document.createElement('div');
  host.id = 'monthly-ai-host';
  document.documentElement.appendChild(host);
  const root = host.attachShadow({ mode: 'closed' });

  const link = document.createElement('link');
  link.rel = 'stylesheet';
  link.href = chrome.runtime.getURL('content.css');
  root.appendChild(link);

  // ---------- DOM 结构 ----------
  const wrapper = document.createElement('div');
  wrapper.innerHTML = `
    <button class="fab" data-role="fab" title="AI 分析" type="button">AI</button>
    <div class="panel" data-role="panel">
      <div class="panel-header">
        <div class="title" data-role="title">AI 分析</div>
        <div class="actions">
          <button class="reanalyze" data-role="reanalyze" type="button" title="重新分析（消耗 token）" style="display:none;">🔄 重新分析</button>
          <button class="close" data-role="close" type="button" title="关闭">✕</button>
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
      <div class="panel-body" data-role="body">
        <div class="hint">加载中…</div>
      </div>
      <div class="chat-area" data-role="chatArea" style="display:none;">
        <div class="chat-messages" data-role="chatMessages"></div>
        <div class="chat-warning" data-role="chatWarning" style="display:none;"></div>
        <div class="chat-input-row">
          <input type="text" data-role="chatInput" placeholder="基于当前分析追问…" autocomplete="off">
          <button data-role="chatSend" type="button">发送</button>
          <button data-role="chatClear" type="button">清空</button>
          <button data-role="chatSave" type="button" title="保存对话到历史">💾</button>
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

  closeEl.addEventListener('click', () => panelEl.classList.remove('open'));
  fabEl.addEventListener('click', () => {
    if (!panelEl.classList.contains('open')) {
      panelEl.classList.add('open');
      analyze(false);
    }
  });
  reanalyzeEl.addEventListener('click', () => {
    if (!confirm('重新分析会消耗 token，确定？')) return;
    analyze(true);
  });

  let busy = false;
  let lastAnalyzedCode = ''; // 用于 URL 变化检测
  let conversationHistory = [];
  let currentAnalysisId = null; // 当前分析的历史记录 ID

  // ---------- 从页面提取当前价格 ----------
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
    // 尝试从页面标题解析
    const title = document.title;
    const match = title.match(/(\d+\.\d{2})/);
    if (match) return parseFloat(match[1]);
    return null;
  }

  // ---------- 从分析文本提取位置百分位 ----------
  function extractPositionPercentile(analysisText) {
    // 匹配 "第 XX 百分位" 或 "XX.XX% 的位置" 等模式
    const patterns = [
      /第\s*(\d+(?:\.\d+)?)\s*百分位/,
      /百分位[：:]\s*(\d+(?:\.\d+)?)\s*%/,
      /处于[^\d]*(\d+(?:\.\d+)?)\s*%/,
      /位置[^\d]*(\d+(?:\.\d+)?)\s*百分位/,
      /(\d+(?:\.\d+)?)\s*百分位/,
    ];
    for (const p of patterns) {
      const m = analysisText.match(p);
      if (m) return parseFloat(m[1]);
    }
    return null;
  }

  // ---------- 内容侧缓存 ----------
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
    // 30 天过期
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

  // ---------- 从页面抓取事件 ----------
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

  // ---------- 解析股票代码 ----------
  function parseCodeFromUrl(url) {
    // 东财 URL 实际格式: quote.eastmoney.com/sh600519.html（无斜杠分隔）
    const m = (url || location.href).match(/\b(sh|sz|bj)\.?(\d{4,6})\b/i);
    if (m) return m[2];
    return null;
  }

  // ---------- 主分析流程 ----------
  async function analyze(force) {
    if (busy) return;
    busy = true;
    fabEl.disabled = true;
    reanalyzeEl.disabled = true;
    panelEl.classList.add('open');
    titleEl.textContent = 'AI 分析';
    overviewEl.style.display = 'none';
    metaEl.style.display = 'none';
    reanalyzeEl.style.display = 'none';
    setBody('<div class="loading">正在加载…</div>');

    try {
      const code = parseCodeFromUrl();
      if (!code) {
        setBody('<div class="error">无法从当前页面解析股票代码</div>');
        return;
      }
      lastAnalyzedCode = code;

      // 1. 获取当前 template 设置（用于缓存 key）
      const settingsItems = await chrome.storage.local.get(['template']);
      const currentTemplate = settingsItems.template || 'technical';

      // 2. 先查内容侧缓存（非强制模式）
      if (!force) {
        const cached = await getContentCache(code, currentTemplate);
        if (cached?.analysis) {
          renderResult({ ...cached, cached: true });
          return;
        }
      }

      // 3. 调 SW
      const pageEvents = collectPageEvents();
      const resp = await chrome.runtime.sendMessage({
        type: 'ANALYZE',
        url: location.href,
        force,
        pageEvents,
      });
      if (!resp) {
        setBody('<div class="error">service worker 无响应，可能扩展未加载</div>');
        return;
      }
      if (!resp.ok) {
        setBody(`<div class="error">${escapeHtml(resp.error || '未知错误')}</div>`);
        return;
      }

      // 4. 写入内容侧缓存
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
      setBody(`<div class="error">通信错误: ${escapeHtml(err.message || String(err))}</div>`);
    } finally {
      busy = false;
      fabEl.disabled = false;
      reanalyzeEl.disabled = false;
    }
  }

  // ---------- 渲染结果 ----------
  function renderResult(resp) {
    titleEl.textContent = `${escapeHtml(resp.name)} (${escapeHtml(resp.code)})`;
    reanalyzeEl.style.display = 'inline-block';

    // 元信息条
    const dtStr = resp.analyzedAt ? formatTime(resp.analyzedAt) : '';
    const badge = resp.cached
      ? '<span class="badge badge-cache">缓存</span>'
      : '<span class="badge badge-fresh">新分析</span>';
    const periodLabel = PERIOD_LABELS[resp.period] || '';
    const styleLabel = STYLE_LABELS[resp.style] || '';
    const providerLabel = PROVIDER_LABELS[resp.provider] || (resp.provider || '');
    const dbBadge = resp.mode === 'debate' ? ' · 辩论模式' : '';
    const dcBadge = resp.decision === 'on' ? ' · 决策模式' : '';
    metaEl.innerHTML = `${badge} ${periodLabel} ${escapeHtml(resp.monthKey || resp.bucket || '')} · ${dtStr} · ${providerLabel}(${escapeHtml(resp.model || '')})${styleLabel ? ' · ' + styleLabel : ''}${dbBadge}${dcBadge}`;
    metaEl.style.display = 'block';

    // 概览条: 股票名+代码+当前价+位置百分位
    const currentPrice = extractCurrentPrice();
    const percentile = extractPositionPercentile(resp.analysis || '');
    ovNameEl.textContent = resp.name || '';
    ovCodeEl.textContent = resp.code || '';
    ovPriceEl.textContent = currentPrice !== null ? `¥${currentPrice.toFixed(2)}` : '';
    ovPercentileEl.textContent = percentile !== null ? `P${Math.round(percentile)}` : '';
    overviewEl.style.display = 'flex';

    // 卡片渲染
    setBody(renderCards(resp.analysis || ''));

    // 记录历史 ID 用于后续保存对话
    currentAnalysisId = resp.historyId || null;

    // 初始化对话历史
    const firstUserMsg = resp.prompt
      || `分析 ${resp.name || ''}(${resp.code || ''}) ${PERIOD_LABELS[resp.period] || resp.period || ''}`;
    conversationHistory = [
      { role: 'user', content: firstUserMsg },
      { role: 'assistant', content: resp.analysis || '' },
    ];

    // 显示对话区域
    chatAreaEl.style.display = 'flex';
    chatWarningEl.style.display = 'none';
    chatInputEl.value = '';
    renderChatMessages();
  }

  // ---------- 卡片折叠式渲染 ----------
  function renderCards(md) {
    const sections = parseSections(md);
    if (sections.length === 0) {
      return `<div class="analysis">${renderMarkdown(md)}</div>`;
    }

    return sections.map((sec) => {
      const category = classifySection(sec.title);
      const isExpanded = (category === 'position' || category === 'ma' || category === 'price' || category === 'conclusion');
      const icon = isExpanded ? '▼' : '▶';
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

  // Markdown 按 ## / ### 分段
  function parseSections(md) {
    const lines = String(md).split('\n');
    const sections = [];
    let currentTitle = '';
    let currentLines = [];
    let isFirstNonEmpty = true;

    for (const line of lines) {
      const h2 = line.match(/^##\s+(.*)$/);
      const h3 = line.match(/^###\s+(.*)$/);
      const h1 = line.match(/^#\s+(.*)$/);

      if (h1 || h2 || h3) {
        // 保存上一个 section
        const bodyText = currentLines.join('\n').trim();
        if (currentTitle || bodyText) {
          sections.push({ title: currentTitle || '概述', body: bodyText });
        }
        currentTitle = (h1 || h2 || h3)[1].trim();
        currentLines = [];
        isFirstNonEmpty = false;
        continue;
      }

      // 跳过结构化 JSON 块
      if (currentLines.length === 0 && line.trim() === '```json') {
        // 找到闭合的 ```
        const closeIdx = lines.indexOf('```', lines.indexOf(line) + 1);
        if (closeIdx > -1) {
          lines.splice(lines.indexOf(line), closeIdx - lines.indexOf(line) + 1);
          continue;
        }
      }

      currentLines.push(line);
    }

    // 最后一段
    const bodyText = currentLines.join('\n').trim();
    if (currentTitle || bodyText) {
      sections.push({ title: currentTitle || '概述', body: bodyText });
    }

    return sections;
  }

  function classifySection(title) {
    const t = title.toLowerCase();
    if (/位置|分位|区间|高位|低位|中继|估值阶段/.test(t)) return 'position';
    if (/均线|ma\d|排列|移动平均/.test(t)) return 'ma';
    if (/支撑|压力|阻力|价位|中枢|买卖点|加仓|减仓|止损/.test(t)) return 'price';
    if (/趋势|方向|走势|判断|共振/.test(t)) return 'trend';
    if (/反方|风险|反向|出错/.test(t)) return 'counter';
    if (/操作|建议|策略|仓位|入场|决策/.test(t)) return 'action';
    if (/结论|总结|综合/.test(t)) return 'conclusion';
    return '';
  }

  const CATEGORY_LABELS = {
    position: '位置', ma: '均线', price: '价位',
    trend: '趋势', counter: '反方', action: '操作', conclusion: '结论',
  };

  // 卡片折叠点击
  root.addEventListener('click', (e) => {
    const header = e.target.closest('[data-card-toggle]');
    if (!header) return;
    const card = header.parentElement;
    const body = card.querySelector('.card-body');
    const icon = header.querySelector('.card-icon');
    if (body.classList.contains('collapsed')) {
      body.classList.remove('collapsed');
      header.classList.add('expanded');
      icon.textContent = '▼';
    } else {
      body.classList.add('collapsed');
      header.classList.remove('expanded');
      icon.textContent = '▶';
    }
  });

  // ---------- Markdown 渲染 ----------
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

  const STYLE_LABELS = { technical: '技术分析', chanlun: '缠论', value: '价值视角', comprehensive: '综合' };
  const PERIOD_LABELS = { monthly: '月线', weekly: '周线', daily: '日线', multi: '多周期' };
  const PROVIDER_LABELS = { anthropic: 'Claude', deepseek: 'DeepSeek' };

  function renderMarkdown(md) {
    const lines = String(md).split('\n');
    const out = [];
    let inList = false;
    const closeList = () => { if (inList) { out.push('</ul>'); inList = false; } };

    for (const raw of lines) {
      // 跳过单个 ``` 标记
      if (raw.trim() === '```' || raw.trim() === '```json') continue;
      const h = raw.match(/^(#{1,6})\s+(.*)$/);
      if (h) {
        closeList();
        const level = Math.min(h[1].length + 2, 5); // 卡片内从 h3 开始
        out.push(`<h${level}>${formatInline(h[2])}</h${level}>`);
        continue;
      }
      const li = raw.match(/^[-*]\s+(.*)$/);
      if (li) {
        if (!inList) { out.push('<ul>'); inList = true; }
        out.push(`<li>${formatInline(li[1])}</li>`);
        continue;
      }
      // 编号列表
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

  // ---------- 多轮对话 ----------

  function renderChatMessages() {
    if (conversationHistory.length === 0) {
      chatMessagesEl.innerHTML = '';
      return;
    }
    chatMessagesEl.innerHTML = conversationHistory.map((msg, i) => {
      // 跳过首次 prompt（太长，不渲染）
      if (i === 0 && msg.role === 'user') {
        return `<div class="chat-msg chat-msg-user chat-msg-first">
          <div class="chat-bubble">📊 首次分析 prompt（已提交，长度 ${msg.content.length} 字符）</div>
        </div>`;
      }
      const cls = msg.role === 'user' ? 'chat-msg-user' : 'chat-msg-assistant';
      return `<div class="chat-msg ${cls}">
        <div class="chat-bubble">${renderMarkdown(msg.content)}</div>
      </div>`;
    }).join('');
    chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;

    // 20 条以上 warning
    if (conversationHistory.length >= 20) {
      chatWarningEl.style.display = 'block';
      chatWarningEl.textContent = `⚠️ 对话已达 ${conversationHistory.length} 条，token 消耗较大。建议清空对话后重新开始。`;
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

    // 渲染 user 消息
    conversationHistory.push({ role: 'user', content: question });
    renderChatMessages();

    // 添加加载占位
    const loadingIdx = conversationHistory.length;
    conversationHistory.push({ role: 'assistant', content: '⏳ 思考中…' });
    renderChatMessages();

    try {
      const resp = await chrome.runtime.sendMessage({
        type: 'FOLLOW_UP',
        question,
        history: conversationHistory.filter((m) => m.content !== '⏳ 思考中…').slice(0, -1),
      });
      // 移除加载占位
      conversationHistory.splice(loadingIdx, 1);

      if (!resp || !resp.ok) {
        conversationHistory.push({ role: 'assistant', content: `❌ 错误：${resp?.error || '无响应'}` });
      } else {
        conversationHistory.push({ role: 'assistant', content: resp.text });
      }
    } catch (err) {
      conversationHistory.splice(loadingIdx, 1);
      conversationHistory.push({ role: 'assistant', content: `❌ 通信错误：${err.message || String(err)}` });
    }

    renderChatMessages();
    busy = false;
    chatSendEl.disabled = false;
    chatInputEl.focus();
  }

  function clearChat() {
    if (conversationHistory.length <= 2) return;
    // 保留前两条（首次 prompt + 首次分析结果）
    conversationHistory = conversationHistory.slice(0, 2);
    renderChatMessages();
    chatInputEl.focus();
  }

  // 保存对话到历史
  async function saveConversationToHistory() {
    if (!currentAnalysisId) {
      chatSaveEl.textContent = '⚠️';
      setTimeout(() => { chatSaveEl.textContent = '💾'; }, 1500);
      return;
    }
    chatSaveEl.disabled = true;
    chatSaveEl.textContent = '⏳';
    try {
      const resp = await chrome.runtime.sendMessage({
        type: 'SAVE_CONVERSATION',
        id: currentAnalysisId,
        conversationHistory,
      });
      if (resp && resp.ok) {
        chatSaveEl.textContent = '✅';
      } else {
        chatSaveEl.textContent = '❌';
      }
    } catch (_) {
      chatSaveEl.textContent = '❌';
    }
    setTimeout(() => {
      chatSaveEl.textContent = '💾';
      chatSaveEl.disabled = false;
    }, 1500);
  }

  // 对话事件
  chatSendEl.addEventListener('click', sendFollowUp);
  chatClearEl.addEventListener('click', clearChat);
  chatSaveEl.addEventListener('click', saveConversationToHistory);
  chatInputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendFollowUp();
    }
  });

  // ---------- URL 变化监听 ----------
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

  // Monkey-patch pushState / replaceState
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

  // ---------- 自动触发 ----------
  // 页面加载完成后自动分析
  if (document.readyState === 'complete') {
    setTimeout(() => analyze(false), 800);
  } else {
    window.addEventListener('load', () => setTimeout(() => analyze(false), 800));
  }
})();
