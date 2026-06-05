// Opportunity stock daily report — sorted by signal strength, generates markdown
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..', '..');
const REPORT_DIR = path.join(PROJECT_DIR, '.eastmoney-ai', 'daily-reports');

function ensureDir() {
  if (!fs.existsSync(REPORT_DIR)) fs.mkdirSync(REPORT_DIR, { recursive: true });
}

// Signal strength keywords
const BULL_KEYWORDS = ['突破', '底部', '反转', '金叉', '三买', '放量上涨', '多头排列', '底背离', '低估'];
const BEAR_KEYWORDS = ['顶背离', '天量', '见顶', '三卖', '放量下跌', '空头排列', '死叉', '破位', '高估'];

function scoreSignal(result) {
  if (!result.judgment) return { direction: 'neutral', score: 0, keywords: [] };

  const text = (result.text || '').toLowerCase();
  let score = 0;
  let keywords = [];

  if (result.judgment === 'bull') {
    score += 1;
    for (const kw of BULL_KEYWORDS) {
      if (text.includes(kw)) { score += 2; keywords.push(kw); }
    }
    return { direction: 'bull', score, keywords };
  }

  if (result.judgment === 'bear') {
    score += 1;
    for (const kw of BEAR_KEYWORDS) {
      if (text.includes(kw)) { score += 2; keywords.push(kw); }
    }
    return { direction: 'bear', score, keywords };
  }

  return { direction: 'neutral', score: 0, keywords: [] };
}

/**
 * @param {Array} scanResults — batch scan 返回的 results
 * @param {object} meta — { totalStocks, totalCost, monthSpent, monthBudget }
 * @param {Date} [date] — 默认今天
 * @returns {string} 日报路径
 */
export function generateDailyReport(scanResults, meta = {}, date = new Date()) {
  const today = date.toISOString().slice(0, 10);

  // 评分：优先用 scoreData.score，降级到关键词匹配
  const scored = scanResults
    .filter((r) => !r.error)
    .map((r) => {
      const kw = scoreSignal(r);
      // 如果解析到了 scoreData，用其 score 替换关键词评分
      if (r.scoreData && typeof r.scoreData.score === 'number') {
        return { ...r, direction: kw.direction, score: r.scoreData.score, keywords: kw.keywords };
      }
      return { ...r, ...kw };
    });

  const bulls = scored.filter((r) => r.direction === 'bull').sort((a, b) => b.score - a.score);
  const bears = scored.filter((r) => r.direction === 'bear').sort((a, b) => b.score - a.score);
  const neutrals = scored.filter((r) => r.direction === 'neutral');
  const errors = scanResults.filter((r) => r.error);

  const topBulls = bulls.slice(0, 10);
  const topBears = bears.slice(0, 10);

  let md = `# 机会股日报 ${today}\n\n`;
  md += `## 扫描概况\n\n`;
  md += `- 扫描股票：${meta.totalStocks || scanResults.length} 只\n`;
  md += `- 成功：${scored.length} / 失败：${errors.length}\n`;
  md += `- 总成本：¥${(meta.totalCost || 0).toFixed(4)}\n`;
  if (meta.monthSpent != null) {
    md += `- 本月累计：¥${meta.monthSpent.toFixed(2)} / ¥${(meta.monthBudget || 50)}\n`;
  }
  md += `- 看多信号：${bulls.length} / 看空信号：${bears.length} / 中性：${neutrals.length}\n\n`;

  if (topBears.length > 0) {
    md += `## 🔴 强看空信号（top ${Math.min(10, topBears.length)}）\n\n`;
    md += `| 代码 | 名称 | 判断 | 强度 | 关键词 |\n`;
    md += `|------|------|------|------|--------|\n`;
    for (const r of topBears) {
      md += `| ${r.code} | ${r.name || ''} | 偏空 | ${r.score} | ${r.keywords.join(', ') || '-'} |\n`;
    }
    md += '\n';
  }

  if (topBulls.length > 0) {
    md += `## 🟢 强看多信号（top ${Math.min(10, topBulls.length)}）\n\n`;
    md += `| 代码 | 名称 | 判断 | 强度 | 关键词 |\n`;
    md += `|------|------|------|------|--------|\n`;
    for (const r of topBulls) {
      md += `| ${r.code} | ${r.name || ''} | 偏多 | ${r.score} | ${r.keywords.join(', ') || '-'} |\n`;
    }
    md += '\n';
  }

  if (neutrals.length > 0) {
    md += `## ⚪ 中性信号（${neutrals.length} 只）\n\n`;
    md += neutrals.slice(0, 20).map((r) => `- ${r.code} ${r.name || ''}`).join('\n');
    if (neutrals.length > 20) md += `\n- ...等 ${neutrals.length - 20} 只`;
    md += '\n\n';
  }

  if (errors.length > 0) {
    md += `## ❌ 失败/跳过（${errors.length} 只）\n\n`;
    for (const r of errors.slice(0, 10)) {
      md += `- ${r.code || '?'} ${r.name || ''}：${r.error || '未知错误'}\n`;
    }
    if (errors.length > 10) md += `- ...等 ${errors.length - 10} 只\n`;
    md += '\n';
  }

  ensureDir();
  const reportPath = path.join(REPORT_DIR, `${today}.md`);
  fs.writeFileSync(reportPath, md, 'utf-8');
  return reportPath;
}
