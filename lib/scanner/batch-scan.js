// Batch scanner — batched concurrency + budget guard + progress logging
// Provider locked to deepseek-chat, no override
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { checkBudget, recordSpending } from '../evaluation/cost-guard.js';
import { extractJudgment } from '../evaluation/collector.js';
import { parseScoreBlock } from '../dashboard/parse-score.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..', '..');

// Average token estimate per analysis (technical template ~8K input + 2K output)
const AVG_PROMPT_TOKENS = 8000;
const AVG_COMPLETION_TOKENS = 2000;
const AVG_COST_CNY = (8000 / 1_000_000) * 1 + (2000 / 1_000_000) * 4; // DeepSeek chat 定价

/**
 * @param {Array} stockList — [{ code, market, name }]
 * @param {object} options
 * @returns {{ total, succeeded, failed, totalCostCny, results }}
 */
export async function runBatchScan(stockList, options = {}) {
  const {
    fetchKlines,          // (market, code) => klines[]
    callDeepSeek,          // (prompt) => { text, usage }
    template = 'technical',
    batchSize = 10,
    batchDelayMs = 5000,
    onProgress,
  } = options;

  if (!fetchKlines || !callDeepSeek) {
    throw new Error('runBatchScan 需要 fetchKlines 和 callDeepSeek');
  }

  const total = stockList.length;
  const estimatedCost = total * AVG_COST_CYN;

  // 预算检查
  try {
    checkBudget(estimatedCost);
  } catch (err) {
    throw new Error(`预算不足，无法启动批量扫描：${err.message}。预估成本 ¥${estimatedCost.toFixed(2)}`);
  }

  const results = [];
  let succeeded = 0;
  let failed = 0;
  let totalCostCny = 0;
  let budgetExhausted = false;

  // 分批
  const batches = [];
  for (let i = 0; i < stockList.length; i += batchSize) {
    batches.push(stockList.slice(i, i + batchSize));
  }

  let globalIdx = 0;
  for (const batch of batches) {
    if (budgetExhausted) break;

    // 批内并发
    const batchPromises = batch.map(async (stock) => {
      const idx = ++globalIdx;
      try {
        // 再次检查预算（每次调之前）
        try {
          checkBudget(AVG_COST_CYN);
        } catch (_) {
          budgetExhausted = true;
          return { code: stock.code, name: stock.name, error: '预算耗尽' };
        }

        // 拉K线
        let klines;
        try {
          const raw = await fetchKlines(stock.market, stock.code, 60, 'monthly');
          klines = raw;
        } catch (err) {
          failed++;
          onProgress?.({ idx, total, code: stock.code, status: 'kline_error', error: err.message });
          return { code: stock.code, name: stock.name, error: `K线获取失败: ${err.message}` };
        }

        if (!klines || klines.length < 12) {
          failed++;
          onProgress?.({ idx, total, code: stock.code, status: 'skip', reason: 'K线不足' });
          return { code: stock.code, name: stock.name, error: 'K线数据不足' };
        }

        // 拼简短 prompt（不算 MA/MACD，让 DeepSeek 自己看）
        const table = buildSimpleTable(klines, stock);
        const prompt = `你是 A 股技术分析师。以下是 ${stock.name}(${stock.code}) 近 ${klines.length} 个月月线数据：

${table}

请用中文简要分析：
1. 均线排列（多头/空头/交叉）
2. 当前位置（高位/中位/低位）
3. 方向判断：【偏多 / 偏空 / 中性震荡】
4. 一句话关键观察

用 Markdown 格式输出，控制在 300 字以内。`;

        const result = await callDeepSeek(prompt);
        const cost = (result.usage?.inputTokens || AVG_PROMPT_TOKENS) / 1_000_000 * 1
          + (result.usage?.outputTokens || AVG_COMPLETION_TOKENS) / 1_000_000 * 4;

        recordSpending(cost);
        totalCostCny += cost;

        const judgment = extractJudgment(result.text);
        const scoreData = parseScoreBlock(result.text);
        succeeded++;
        onProgress?.({ idx, total, code: stock.code, name: stock.name, status: 'ok', judgment });

        return { code: stock.code, name: stock.name, judgment, scoreData, text: result.text, cost };
      } catch (err) {
        failed++;
        onProgress?.({ idx, total, code: stock.code, status: 'error', error: err.message });
        return { code: stock.code, name: stock.name, error: err.message };
      }
    });

    const batchResults = await Promise.allSettled(batchPromises);
    for (const r of batchResults) {
      if (r.status === 'fulfilled' && r.value) results.push(r.value);
    }

    // 批次间延迟
    if (batches.indexOf(batch) < batches.length - 1 && !budgetExhausted) {
      await sleep(batchDelayMs);
    }
  }

  return {
    total,
    succeeded,
    failed,
    totalCostCny: +totalCostCny.toFixed(4),
    budgetExhausted,
    results,
  };
}

function buildSimpleTable(klines, stock) {
  const header = '日期\t开盘\t收盘\t最高\t最低\t成交量\t涨跌幅';
  const rows = klines.slice(-24).map((k) => [
    k.date, k.open, k.close, k.high, k.low, k.volume, k.changePercent,
  ].join('\t'));
  return [header, ...rows].join('\n');
}

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }
