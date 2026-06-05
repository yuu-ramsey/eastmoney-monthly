#!/usr/bin/env node
// ema CLI — 自学习闭环命令行工具
// 用法：
//   ema nightly                  — 跑夜间作业
//   ema scheduled                — 跑当日调度任务
//   ema sync check               — 检查 Native Messaging 同步状态
//   ema review draft             — 立即生成新草稿
//   ema review list              — 列所有草稿
//   ema review show {date}       — 看某天草稿
//   ema review approve {date}    — 审核通过，生成 Claude 精修
//   ema budget show              — 看预算
//   ema budget set --monthly 80  — 改限额
//   ema db init --scope hs300    — 建本地 K 线库（阶段 8）

import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..');
const DATA_DIR = path.join(PROJECT_DIR, '.eastmoney-ai');
const STORAGE_DIR = path.join(DATA_DIR, 'storage');

async function main() {
  const args = process.argv.slice(2);
  if (args.length === 0) {
    printUsage();
    process.exit(0);
  }

  const cmd = args[0];
  const sub = args[1];

  try {
    switch (cmd) {
      case 'nightly': {
        await handleNightly(args.slice(1));
        break;
      }

      case 'scheduled': {
        await handleScheduled(args.slice(1));
        break;
      }

      case 'sync': {
        await handleSyncCheck();
        break;
      }

      case 'review': {
        await handleReview(sub, args.slice(2));
        break;
      }

      case 'budget': {
        await handleBudget(sub, args.slice(2));
        break;
      }

      case 'db': {
        await handleDb(sub, args.slice(2));
        break;
      }

      case 'sector': {
        await handleSector(sub, args.slice(2));
        break;
      }

      case 'analyze': {
        await handleAnalyze(sub, args.slice(2));
        break;
      }

      default:
        console.error(`未知命令: ${cmd}`);
        printUsage();
        process.exit(1);
    }
  } catch (err) {
    console.error('错误:', err.message);
    if (process.env.EMA_DEBUG) console.error(err.stack);
    process.exit(1);
  }
}

// ---- nightly ----

async function handleNightly(args) {
  console.log('=== 夜间作业 ===');

  // 1. 读 history
  const historyPath = path.join(STORAGE_DIR, 'history.json');
  let historyEntries = [];
  if (fs.existsSync(historyPath)) {
    try {
      const raw = fs.readFileSync(historyPath, 'utf-8');
      historyEntries = JSON.parse(raw);
      if (!Array.isArray(historyEntries)) historyEntries = [];
      console.log(`读取 history: ${historyEntries.length} 条`);
    } catch (err) {
      console.warn(`读取 history 失败: ${err.message}`);
    }
  }

  if (historyEntries.length === 0) {
    console.warn('警告: 本地无 history 数据。');
    console.warn('');
    console.warn('可能原因:');
    console.warn('  1. Native Messaging 尚未配置（运行 ema sync check 检查）');
    console.warn('  2. Chrome 扩展尚未做任何分析');
    console.warn('  3. 手动从 Chrome 扩展导出 history 后放入:');
    console.warn(`     ${historyPath}`);
    return;
  }

  // 2. 构造 fetchers（本地库优先，在线 fallback）
  const fetchKlines = await makeKlinesFetcher();
  const fetchIndexKlines = () => fetchKlines('1', '000300');

  // 3. DeepSeek 调用（强制 deepseek-chat 控成本）
  const { getProvider } = await import('../lib/llm/index.js');

  async function callDeepSeek(prompt) {
    const provider = getProvider('deepseek');
    const apiKey = process.env.DEEPSEEK_API_KEY;
    if (!apiKey) throw new Error('请设置 DEEPSEEK_API_KEY 环境变量');

    try {
      const result = await provider.call(prompt, {
        model: 'deepseek-chat',
        apiKey,
        maxTokens: 2000,
      });
      return { text: result.text, usage: result.usage };
    } catch (err) {
      if (err.message.includes('401')) throw new Error('DeepSeek API key 无效，检查 DEEPSEEK_API_KEY');
      if (err.message.includes('429')) throw new Error('DeepSeek 限流，稍后重试');
      throw err;
    }
  }

  // 4. 跑夜间作业
  const { runNightlyJob } = await import('../lib/evaluation/nightly.js');

  try {
    const result = await runNightlyJob({
      historyEntries,
      fetchKlines,
      fetchIndexKlines,
      callDeepSeek,
    });

    console.log('');
    console.log('=== 夜间作业完成 ===');
    console.log('状态:', result.status);
    if (result.newEvals != null) console.log('新增评估:', result.newEvals, '条');
    if (result.totalEvals != null) console.log('评估总数:', result.totalEvals, '条');
    if (result.draftPath) console.log('草稿:', result.draftPath);
    if (result.cost != null) console.log('成本: ¥' + result.cost.toFixed(4));
  } catch (err) {
    console.error('夜间作业失败:', err.message);
    process.exit(1);
  }
}

// ---- scheduled ----

async function handleScheduled(args) {
  console.log('=== 当日调度 ===');

  const { getScheduleForDate, loadSafetyConfig } = await import('../lib/scanner/scheduler.js');

  const schedule = getScheduleForDate();
  const safety = loadSafetyConfig();

  console.log('日期:', new Date().toISOString().slice(0, 10));
  console.log('调度:', schedule.reason);
  console.log('安全:', JSON.stringify(safety));

  if (safety.emergencyStop) {
    console.log('emergencyStop=true，终止所有自动任务');
    return;
  }

  if (!safety.enabled) {
    console.log('enabled=false，跳过');
    return;
  }

  // 1. 跑 evaluation（始终先跑，免费）
  console.log('\n[1/4] 跑 evaluation...');
  try {
    // 先跑 nightly（内部做 evaluateBatch）
    await handleNightly([]);
  } catch (err) {
    console.warn('evaluation 失败:', err.message);
  }

  // 2. HS300 扫描
  if (schedule.runHs300 && !safety.skipHs300) {
    console.log('\n[2/4] 跑 HS300 批量扫描...');
    try {
      await runBatchScan('hs300');
    } catch (err) {
      console.warn('HS300 扫描失败:', err.message);
    }
  } else {
    console.log('\n[2/4] HS300 扫描跳过（非 HS300 周或安全开关关闭）');
  }

  // 3. 自选股扫描
  if (schedule.runWatchlist && !safety.skipWatchlist) {
    console.log('\n[3/4] 跑自选股批量扫描...');
    try {
      await runBatchScan('watchlist');
    } catch (err) {
      console.warn('自选股扫描失败:', err.message);
    }
  } else {
    console.log('\n[3/4] 自选股扫描跳过（非周日或安全开关关闭）');
  }

  // 4. 日报
  if (schedule.runDailyReport && !safety.skipDailyReport) {
    console.log('\n[4/4] 生成日报...');
    try {
      await generateDailyReport();
    } catch (err) {
      console.warn('日报生成失败:', err.message);
    }
  } else {
    console.log('\n[4/4] 日报跳过');
  }

  console.log('\n=== 调度完成 ===');
}

async function runBatchScan(scope) {
  const { fetchHS300Constituents } = await import('../lib/scanner/hs300.js');
  const { loadWatchlist } = await import('../lib/scanner/watchlist.js');
  const { runBatchScan } = await import('../lib/scanner/batch-scan.js');
  const { getProvider } = await import('../lib/llm/index.js');

  let stockList;
  if (scope === 'hs300') {
    stockList = await fetchHS300Constituents();
    console.log(`  HS300 成分股: ${stockList.length} 只`);
  } else {
    const wl = loadWatchlist();
    stockList = wl.stocks || [];
    console.log(`  自选股: ${stockList.length} 只`);
  }

  if (stockList.length === 0) {
    console.log('  无股票，跳过');
    return;
  }

  // 构造 fetchers（本地库优先）
  const fetchKlines = await makeKlinesFetcher();
  const fetchIndexKlines = () => fetchKlines('1', '000300');

  // 强制 deepseek-chat
  const provider = getProvider('deepseek');
  const apiKey = process.env.DEEPSEEK_API_KEY;
  if (!apiKey) throw new Error('请设置 DEEPSEEK_API_KEY 环境变量');

  const callLLM = async (prompt) => {
    const r = await provider.call(prompt, { model: 'deepseek-chat', apiKey, maxTokens: 2000 });
    return { text: r.text, usage: r.usage };
  };

  const options = {
    fetchKlines,
    fetchIndexKlines,
    callLLM,
    onProgress: ({ idx, total, code, status }) => {
      if (idx % 50 === 0) console.log(`  进度: ${idx}/${total}`);
    },
  };

  const result = await runBatchScan(stockList, options);
  console.log('  完成:', result.succeeded, '/', result.total, '成功');
}

async function generateDailyReport() {
  const { generateDailyReport } = await import('../lib/scanner/daily-report.js');
  const report = await generateDailyReport();
  const date = new Date().toISOString().slice(0, 10);
  const reportDir = path.join(DATA_DIR, 'daily-reports');
  if (!fs.existsSync(reportDir)) fs.mkdirSync(reportDir, { recursive: true });
  const reportPath = path.join(reportDir, `report-${date}.md`);
  fs.writeFileSync(reportPath, report, 'utf-8');
  console.log('  日报已保存:', reportPath);
}

// ---- sync check ----

async function handleSyncCheck() {
  console.log('=== 同步状态检查 ===');
  console.log('');

  const historyPath = path.join(STORAGE_DIR, 'history.json');
  const exists = fs.existsSync(historyPath);

  console.log('存储目录:', STORAGE_DIR);
  console.log('history 文件:', exists ? '存在' : '不存在');

  if (exists) {
    try {
      const stat = fs.statSync(historyPath);
      const hoursAgo = (Date.now() - stat.mtimeMs) / 3600000;
      const raw = fs.readFileSync(historyPath, 'utf-8');
      const entries = JSON.parse(raw);
      console.log('条数:', Array.isArray(entries) ? entries.length : '?');
      console.log('最后更新:', stat.mtime.toISOString());
      if (hoursAgo > 24) {
        console.warn(`警告: 超过 ${hoursAgo.toFixed(1)} 小时未更新，数据可能过时`);
      } else {
        console.log(`新鲜度: ${hoursAgo.toFixed(1)} 小时前（OK）`);
      }
    } catch (err) {
      console.warn('读取失败:', err.message);
    }
  } else {
    console.log('');
    console.log('同步未配置。请按以下步骤设置:');
    console.log('  1. 打开 chrome://extensions，找到本扩展');
    console.log('  2. 复制扩展 ID（32 位小写字母）');
    console.log('  3. 运行: node native-host/install.js <扩展ID>');
    console.log('  4. 重启 Chrome');
    console.log('  5. 在东方财富页面做一次分析');
    console.log('  6. 再次运行 ema sync check 验证');
  }

  // 检查所有存储文件
  if (fs.existsSync(STORAGE_DIR)) {
    const files = fs.readdirSync(STORAGE_DIR).filter(f => f.endsWith('.json'));
    if (files.length > 0) {
      console.log('');
      console.log('存储文件列表:');
      for (const f of files) {
        const fp = path.join(STORAGE_DIR, f);
        const size = fs.statSync(fp).size;
        console.log(`  ${f} (${size} bytes)`);
      }
    }
  }
}

// ---- db ----

async function handleDb(sub, args) {
  switch (sub) {
    case 'init': {
      const scopeIdx = args.indexOf('--scope');
      const scope = scopeIdx >= 0 ? args[scopeIdx + 1] : 'hs300';
      const periodsIdx = args.indexOf('--periods');
      const periods = periodsIdx >= 0
        ? args[periodsIdx + 1].split(',').map((s) => s.trim()).filter(Boolean)
        : ['monthly', 'weekly', 'daily'];
      const sourceIdx = args.indexOf('--source');
      const source = sourceIdx >= 0 ? args[sourceIdx + 1] : 'baidu';
      const onlyFailed = args.includes('--only-failed');
      const { runDbInit } = await import('./db-init.js');
      await runDbInit(scope, periods, source, onlyFailed);
      break;
    }

    case 'status': {
      const { showDbStatus } = await import('./db-status.js');
      await showDbStatus();
      break;
    }

    case 'update': {
      console.log('db update 将在后续实现');
      break;
    }

    default:
      console.error(`未知 db 子命令: ${sub}`);
      console.log('可用: init | status | update');
      process.exit(1);
  }
}

// ---- sector ----

async function handleSector(sub, args) {
  switch (sub) {
    case 'init': {
      const force = args.includes('--force');
      const { sectorInit } = await import('./sector-init.js');
      await sectorInit(force);
      break;
    }

    case 'alpha': {
      const code = args[0];
      if (!code) {
        console.error('用法: ema sector alpha <股票代码> [period] [lookback]');
        process.exit(1);
      }
      const period = args[1] || 'monthly';
      const lookback = parseInt(args[2] || '12', 10);
      const { getDb, closeDb } = await import('../lib/db/connection.js');
      const { calcSectorAlpha } = await import('../lib/sector/alpha.js');
      const db = getDb();
      try {
        const result = calcSectorAlpha(db, code, period, lookback);
        if (!result) {
          console.log(`${code} 无行业映射数据`);
        } else {
          console.log(JSON.stringify(result, null, 2));
        }
      } finally {
        closeDb();
      }
      break;
    }

    default:
      console.error(`未知 sector 子命令: ${sub}`);
      console.log('可用: init | alpha <code>');
      process.exit(1);
  }
}

// ---- review ----

async function handleReview(sub, args) {
  const reviewDir = path.join(PROJECT_DIR, '.eastmoney-ai', 'reviews');

  switch (sub) {
    case 'draft': {
      console.log('立即生成草稿需要 Chrome 运行时。');
      console.log('请用 `ema nightly` 触发夜间作业，或 `ema scheduled` 跑完整调度。');
      break;
    }

    case 'list': {
      if (!fs.existsSync(reviewDir)) {
        console.log('暂无草稿');
        return;
      }
      const files = fs.readdirSync(reviewDir).filter((f) => f.startsWith('draft-')).sort().reverse();
      if (files.length === 0) console.log('暂无草稿');
      else files.forEach((f) => console.log(' ', f));
      break;
    }

    case 'show': {
      const date = args[0] || new Date().toISOString().slice(0, 10);
      const p = path.join(reviewDir, `draft-${date}.md`);
      if (!fs.existsSync(p)) {
        console.error(`草稿不存在: ${date}`);
        process.exit(1);
      }
      console.log(fs.readFileSync(p, 'utf-8'));
      break;
    }

    case 'approve': {
      const date = args[0] || new Date().toISOString().slice(0, 10);
      const draftPath = path.join(reviewDir, `draft-${date}.md`);
      if (!fs.existsSync(draftPath)) {
        console.error(`草稿不存在: ${date}`);
        process.exit(1);
      }

      const { parseUserReview } = await import('../lib/evaluation/refine.js');
      const { approved } = parseUserReview(draftPath);

      if (approved.length === 0) {
        console.log('未找到审核通过的建议。请在草稿中将 [ ] 改为 [x] 并标注"通过"。');
        console.log(`草稿路径: ${draftPath}`);
        process.exit(0);
      }

      console.log(`找到 ${approved.length} 条通过的建议：`);
      approved.forEach((s, i) => console.log(`  ${i + 1}. ${s}`));
      console.log('');
      console.log('Claude Opus 精修需要 API key。请设置环境变量 ANTHROPIC_API_KEY。');
      break;
    }

    default:
      console.error(`未知 review 子命令: ${sub}`);
      console.log('可用: draft | list | show <date> | approve <date>');
      process.exit(1);
  }
}

// ---- budget ----

async function handleBudget(sub, args) {
  const { getBudgetSummary, setBudgetConfig } = await import('../lib/evaluation/cost-guard.js');

  switch (sub) {
    case 'show': {
      const summary = getBudgetSummary();
      console.log(`月度预算 (${summary.currentMonth}):`);
      console.log(`  已花: ¥${summary.monthSpent} / ¥${summary.monthlyBudget}`);
      console.log(`  剩余: ¥${summary.monthRemaining}`);
      console.log('今日:');
      console.log(`  已花: ¥${summary.todaySpent} / ¥${summary.dailyBudget}`);
      console.log(`  剩余: ¥${summary.todayRemaining}`);
      break;
    }

    case 'set': {
      const updates = {};
      for (let i = 0; i < args.length; i++) {
        if (args[i] === '--monthly' && args[i + 1]) {
          updates.monthlyBudgetCny = parseFloat(args[++i]);
        } else if (args[i] === '--daily' && args[i + 1]) {
          updates.dailyBudgetCny = parseFloat(args[++i]);
        }
      }
      if (Object.keys(updates).length === 0) {
        console.log('用法: ema budget set --monthly 80 --daily 5');
        process.exit(1);
      }
      setBudgetConfig(updates);
      console.log('预算已更新:', updates);
      break;
    }

    default:
      console.error(`未知 budget 子命令: ${sub}`);
      console.log('可用: show | set --monthly <n> --daily <n>');
      process.exit(1);
  }
}

// 构造本地优先的 K 线获取器
async function makeKlinesFetcher() {
  const { getKlines } = await import('../lib/db/klines-repo.js');
  const { fetchKlinesWithFallback } = await import('../lib/data-sources/dispatcher.js');

  const onlineFetcher = async (params) => {
    const result = await fetchKlinesWithFallback(params);
    return result;
  };

  return async function fetchKlines(market, code, period = 'monthly', limit = 60) {
    try {
      const result = await getKlines({ code, market, period, limit, onlineFetcher });
      return result.klines || [];
    } catch (err) {
      console.warn(`  K 线获取失败 ${code}: ${err.message}`);
      return [];
    }
  };
}

// ---- analyze ----

async function handleAnalyze(sub, args) {
  if (!sub) {
    console.error('用法: ema analyze <股票代码>');
    console.error('例如: ema analyze 600519');
    process.exit(1);
  }

  const code = sub;
  const { execSync } = await import('node:child_process');

  // 1. 获取股票名称
  let stockName = '';
  try {
    const { getDb, closeDb } = await import('../lib/db/connection.js');
    const db = getDb();
    try {
      const row = db.prepare('SELECT name FROM stocks WHERE code = ?').get(code);
      if (row) stockName = row.name;
    } finally {
      closeDb();
    }
  } catch (_) {
    // stocks 表可能不存在，忽略
  }

  // 2. 获取月线数据用于技术指标计算
  const fetchKlines = await makeKlinesFetcher();
  let klines = [];
  try {
    // 市场推断: 6 开头 → sh, 0/3 开头 → sz
    const market = code.startsWith('6') ? '1' : '0';
    klines = await fetchKlines(market, code, 'monthly', 60);
  } catch (err) {
    console.warn(`获取 K 线失败: ${err.message}`);
  }

  // 3. 计算技术信号
  let techSignal = '（无数据）';
  let techDirection = '';
  if (klines.length >= 20) {
    try {
      const { calculateAll } = await import('../lib/indicators/calculate.js');
      const indicators = calculateAll(klines);
      const summary = summarizeTechnicalSignal(indicators, klines);
      techSignal = summary.text;
      techDirection = summary.direction;
    } catch (err) {
      techSignal = `（指标计算失败: ${err.message}）`;
    }
  }

  // 4. 调用 Kronos Python CLI
  const pythonExe = path.join(
    PROJECT_DIR, 'kronos', 'venv', 'Scripts', 'python.exe'
  );
  const cliScript = path.join(PROJECT_DIR, 'kronos', 'cli_predict.py');

  let kronosSignal = null;
  let kronosError = null;
  try {
    const stdout = execSync(
      `"${pythonExe}" "${cliScript}" ${code} --json`,
      { timeout: 120000, encoding: 'utf-8', stdio: ['pipe', 'pipe', 'pipe'] }
    );
    kronosSignal = JSON.parse(stdout);
  } catch (err) {
    kronosError = err.stderr || err.message;
  }

  // 5. 综合研判
  let combined = '';
  if (kronosSignal && techDirection) {
    const aiDir = kronosSignal.direction;
    if (aiDir === 'up' && techDirection === 'bull') {
      combined = '技术 + AI 共振，强烈看多';
    } else if (aiDir === 'down' && techDirection === 'bear') {
      combined = '技术 + AI 共振，强烈看空';
    } else if (aiDir === techDirection) {
      combined = '技术 + AI 同向，温和看' + (aiDir === 'up' ? '多' : '空');
    } else if (aiDir === 'flat') {
      combined = 'AI 预测震荡，技术面' + (techDirection === 'bull' ? '偏多' : '偏空') + '，方向不明';
    } else {
      combined = '技术与 AI 信号分歧，建议观望';
    }
  } else if (kronosSignal) {
    const aiLabel = { up: '看多', down: '看空', flat: '震荡' };
    combined = '仅 AI 信号: ' + (aiLabel[kronosSignal.direction] || kronosSignal.direction);
  } else {
    combined = techDirection === 'bull' ? '仅技术信号: 偏多' :
               techDirection === 'bear' ? '仅技术信号: 偏空' : '信号不足';
  }

  // 6. 输出
  const heading = `=== ${code}${stockName ? ' ' + stockName : ''} 月线分析 ===`;
  console.log(heading);
  console.log(`【技术信号】${techSignal} → ${techDirection === 'bull' ? '看多' : techDirection === 'bear' ? '看空' : '待定'}`);

  if (kronosSignal) {
    const aiDirLabel = { up: '看多', down: '看空', flat: '震荡' };
    console.log(
      `【AI预测】未来${kronosSignal.pred_len}月预测涨幅 ${kronosSignal.predicted_change_pct}%，`
      + `置信度 ${(kronosSignal.confidence * 100).toFixed(0)}%`
      + ` → ${aiDirLabel[kronosSignal.direction] || kronosSignal.direction}`
    );
  } else {
    console.log(`【AI预测】不可用: ${kronosError || '请先下载权重并启动服务'}`);
  }

  console.log(`【综合研判】${combined}`);
}

/** 从指标数据中提取技术面摘要文本 */
function summarizeTechnicalSignal(indicators, klines) {
  const last = klines.length - 1;
  const prev = last - 1;

  const ma5 = indicators.ma5;
  const ma20 = indicators.ma20;
  const ma60 = indicators.ma60;
  const macdHist = indicators.macd?.hist;
  const rsi14 = indicators.rsi14;
  const kdj = indicators.kdj;

  const parts = [];

  // 均线排列
  if (ma5[last] != null && ma20[last] != null && ma60[last] != null) {
    if (ma5[last] > ma20[last] && ma20[last] > ma60[last]) {
      parts.push('均线多头排列');
    } else if (ma5[last] < ma20[last] && ma20[last] < ma60[last]) {
      parts.push('均线空头排列');
    } else {
      parts.push('均线交织');
    }
  }

  // MACD 金叉/死叉
  if (macdHist && macdHist[last] != null && macdHist[prev] != null) {
    if (macdHist[prev] <= 0 && macdHist[last] > 0) {
      parts.push('MACD 金叉');
    } else if (macdHist[prev] >= 0 && macdHist[last] < 0) {
      parts.push('MACD 死叉');
    }
  } else if (indicators.macd?.dif && indicators.macd?.dea) {
    const dif = indicators.macd.dif;
    const dea = indicators.macd.dea;
    if (dif[last] != null && dea[last] != null && dif[prev] != null && dea[prev] != null) {
      if (dif[prev] <= dea[prev] && dif[last] > dea[last]) parts.push('MACD 金叉');
      else if (dif[prev] >= dea[prev] && dif[last] < dea[last]) parts.push('MACD 死叉');
    }
  }

  // RSI 超买/超卖
  if (rsi14[last] != null) {
    if (rsi14[last] > 80) parts.push('RSI 超买');
    else if (rsi14[last] < 20) parts.push('RSI 超卖');
  }

  // KDJ 金叉/死叉
  if (kdj && kdj.k[last] != null && kdj.d[last] != null) {
    if (kdj.k[prev] <= kdj.d[prev] && kdj.k[last] > kdj.d[last]) parts.push('KDJ 金叉');
    else if (kdj.k[prev] >= kdj.d[prev] && kdj.k[last] < kdj.d[last]) parts.push('KDJ 死叉');
  }

  const text = parts.length > 0 ? parts.join('，') : '无明确信号';

  // 方向判定：统计看多/看空信号
  let bullScore = 0, bearScore = 0;
  if (text.includes('多头排列')) bullScore += 2;
  if (text.includes('空头排列')) bearScore += 2;
  if (text.includes('金叉')) bullScore += 1;
  if (text.includes('死叉')) bearScore += 1;
  if (text.includes('超卖')) bullScore += 1;
  if (text.includes('超买')) bearScore += 1;

  let direction = '';
  if (bullScore > bearScore) direction = 'bull';
  else if (bearScore > bullScore) direction = 'bear';

  return { text, direction };
}

function printUsage() {
  console.log('ema — 东方财富月线 AI 分析 CLI');
  console.log('');
  console.log('  ema analyze <code>            技术指标 + Kronos AI 综合研判');
  console.log('  ema nightly                    跑夜间作业');
  console.log('  ema scheduled                  跑当日调度任务');
  console.log('  ema sync check                 检查 Native Messaging 同步状态');
  console.log('  ema review draft               生成新草稿');
  console.log('  ema review list                列所有草稿');
  console.log('  ema review show [date]         看某天草稿');
  console.log('  ema review approve [date]      审核通过（生成精修）');
  console.log('  ema db init --scope hs300       建本地K线库');
  console.log('  ema db status                   数据库状态');
  console.log('  ema sector init                 抓取申万行业映射+市值快照');
  console.log('  ema sector init --force         强制重建映射');
  console.log('  ema budget show                看预算');
  console.log('  ema budget set --monthly 80    改限额');
}

main();
