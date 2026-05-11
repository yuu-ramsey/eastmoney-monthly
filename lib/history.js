// 分析历史纯函数（不依赖 chrome.storage）

export const HISTORY_KEY = 'history';
export const MAX_HISTORY_ITEMS = 100;
export const MAX_HISTORY_BYTES = 9 * 1024 * 1024; // 9MB

export function generateHistoryId() {
  return `h_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

export function trimHistory(list, maxItems = MAX_HISTORY_ITEMS) {
  while (list.length > maxItems) {
    list.shift();
  }
  return list;
}

export function formatHistoryDate(ms) {
  const d = new Date(ms);
  const pad = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

export function historyToMarkdown(entry) {
  const dt = entry.timestamp ? new Date(entry.timestamp).toISOString().replace('T', ' ').slice(0, 19) : '?';
  const providerLabel = entry.provider === 'deepseek' ? 'DeepSeek' : 'Claude';
  const templateLabels = { technical: '技术面', trend: '趋势判断', valuation: '估值面', sentiment: '情绪面' };
  const templateLabel = templateLabels[entry.template] || entry.template || '?';
  let md = `# ${entry.name || '?'} (${entry.code || '?'})\n\n`;
  md += `- 时间: ${dt}\n`;
  md += `- Provider: ${providerLabel} / ${entry.model || '?'}\n`;
  md += `- 分析维度: ${templateLabel}\n\n`;
  md += `## 分析结果\n\n${entry.analysis || ''}\n\n`;
  if (entry.conversationHistory && entry.conversationHistory.length > 2) {
    md += `## 追问记录\n\n`;
    for (let i = 2; i < entry.conversationHistory.length; i++) {
      const msg = entry.conversationHistory[i];
      const roleLabel = msg.role === 'user' ? '追问' : '回复';
      md += `### ${roleLabel}\n\n${msg.content}\n\n`;
    }
  }
  return md;
}

/**
 * 检查历史列表是否需要裁剪
 * @returns {{ trimmed: boolean, reason: string }}
 */
export function checkCapacity(list, maxItems = MAX_HISTORY_ITEMS, maxBytes = MAX_HISTORY_BYTES) {
  if (list.length > maxItems) {
    return { trimmed: true, reason: `条数超限 (${list.length} > ${maxItems})` };
  }
  let bytes;
  try {
    bytes = JSON.stringify(list).length;
  } catch (_) {
    bytes = 0;
  }
  if (bytes > maxBytes) {
    return { trimmed: true, reason: `体积超限 (${(bytes / 1024 / 1024).toFixed(1)}MB > ${(maxBytes / 1024 / 1024).toFixed(1)}MB)` };
  }
  return { trimmed: false, reason: '' };
}
