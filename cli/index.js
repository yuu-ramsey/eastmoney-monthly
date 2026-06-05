#!/usr/bin/env node
// ema CLI — self-learning loop CLI tool
// Usage:
//   ema nightly                  — run nightly job
//   ema scheduled                — run daily scheduled tasks
//   ema sync check               — check Native Messaging sync status
//   ema review draft             — generate new draft immediately
//   ema review list              — list all drafts
//   ema review show {date}       — view draft for a date
//   ema review approve {date}    — approve and generate Claude refinement
//   ema budget show              — view budget
//   ema budget set --monthly 80  — change limit
//   ema db init --scope hs300    — initialize local K-line database (Phase 8)

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
        console.error(`Unknown command: ${cmd}`);
        printUsage();
        process.exit(1);
    }
  } catch (err) {
    console.error('Error:', err.message);
    if (process.env.EMA_DEBUG) console.error(err.stack);
    process.exit(1);
  }
}

// ---- nightly ----

async function handleNightly(args) {
  console.log('=== Nightly Job ===');

  // 1. read history
  const historyPath = path.join(STORAGE_DIR, 'history.json');
  let historyEntries = [];
  if (fs.existsSync(historyPath)) {
    try {
      const raw = fs.readFileSync(historyPath, 'utf-8');
      historyEntries = JSON.parse(raw);
      if (!Array.isArray(historyEntries)) historyEntries = [];
      console.log(`Loaded history: ${historyEntries.length} entries`);
    } catch (err) {
      console.warn(`Failed to load history: ${err.message}`);
    }
  }

  if (historyEntries.length === 0) {
    console.warn('Warning: no local history data.');
    console.warn('');
    console.warn('Possible causes:');
    console.warn('  1. Native Messaging not configured (run `ema sync check`)');
    console.warn('  2. Chrome extension has not performed any analysis yet');
    console.warn('  3. Manually export history from the Chrome extension and place it in:');
    console.warn(`     ${historyPath}`);
    return;
  }

  // 2. build fetchers (local DB first, online fallback)
  const fetchKlines = await makeKlinesFetcher();
  const fetchIndexKlines = () => fetchKlines('1', '000300');

  // 3. DeepSeek call (force deepseek-chat to control cost)
  const { getProvider } = await import('../lib/llm/index.js');

  async function callDeepSeek(prompt) {
    const provider = getProvider('deepseek');
    const apiKey = process.env.DEEPSEEK_API_KEY;
    if (!apiKey) throw new Error('Please set DEEPSEEK_API_KEY environment variable');

    try {
      const result = await provider.call(prompt, {
        model: 'deepseek-chat',
        apiKey,
        maxTokens: 2000,
      });
      return { text: result.text, usage: result.usage };
    } catch (err) {
      if (err.message.includes('401')) throw new Error('DeepSeek API key invalid, check DEEPSEEK_API_KEY');
      if (err.message.includes('429')) throw new Error('DeepSeek rate limited, retry later');
      throw err;
    }
  }

  // 4. run nightly job
  const { runNightlyJob } = await import('../lib/evaluation/nightly.js');

  try {
    const result = await runNightlyJob({
      historyEntries,
      fetchKlines,
      fetchIndexKlines,
      callDeepSeek,
    });

    console.log('');
    console.log('=== Nightly Job Complete ===');
    console.log('Status:', result.status);
    if (result.newEvals != null) console.log('New evaluations:', result.newEvals);
    if (result.totalEvals != null) console.log('Total evaluations:', result.totalEvals);
    if (result.draftPath) console.log('Draft:', result.draftPath);
    if (result.cost != null) console.log('Cost: ¥' + result.cost.toFixed(4));
  } catch (err) {
    console.error('Nightly job failed:', err.message);
    process.exit(1);
  }
}

// ---- scheduled ----

async function handleScheduled(args) {
  console.log('=== Daily Schedule ===');

  const { getScheduleForDate, loadSafetyConfig } = await import('../lib/scanner/scheduler.js');

  const schedule = getScheduleForDate();
  const safety = loadSafetyConfig();

  console.log('Date:', new Date().toISOString().slice(0, 10));
  console.log('Schedule:', schedule.reason);
  console.log('Safety:', JSON.stringify(safety));

  if (safety.emergencyStop) {
    console.log('emergencyStop=true, aborting all automated tasks');
    return;
  }

  if (!safety.enabled) {
    console.log('enabled=false, skipping');
    return;
  }

  // 1. run evaluation (always first, free)
  console.log('\n[1/4] Running evaluation...');
  try {
    // run nightly first (internally calls evaluateBatch)
    await handleNightly([]);
  } catch (err) {
    console.warn('Evaluation failed:', err.message);
  }

  // 2. HS300 scan
  if (schedule.runHs300 && !safety.skipHs300) {
    console.log('\n[2/4] Running HS300 batch scan...');
    try {
      await runBatchScan('hs300');
    } catch (err) {
      console.warn('HS300 scan failed:', err.message);
    }
  } else {
    console.log('\n[2/4] HS300 scan skipped (not HS300 week or safety switch off)');
  }

  // 3. Watchlist scan
  if (schedule.runWatchlist && !safety.skipWatchlist) {
    console.log('\n[3/4] Running watchlist batch scan...');
    try {
      await runBatchScan('watchlist');
    } catch (err) {
      console.warn('Watchlist scan failed:', err.message);
    }
  } else {
    console.log('\n[3/4] Watchlist scan skipped (not Sunday or safety switch off)');
  }

  // 4. Daily report
  if (schedule.runDailyReport && !safety.skipDailyReport) {
    console.log('\n[4/4] Generating daily report...');
    try {
      await generateDailyReport();
    } catch (err) {
      console.warn('Daily report generation failed:', err.message);
    }
  } else {
    console.log('\n[4/4] Daily report skipped');
  }

  console.log('\n=== Schedule Complete ===');
}

async function runBatchScan(scope) {
  const { fetchHS300Constituents } = await import('../lib/scanner/hs300.js');
  const { loadWatchlist } = await import('../lib/scanner/watchlist.js');
  const { runBatchScan } = await import('../lib/scanner/batch-scan.js');
  const { getProvider } = await import('../lib/llm/index.js');

  let stockList;
  if (scope === 'hs300') {
    stockList = await fetchHS300Constituents();
    console.log(`  HS300 constituents: ${stockList.length}`);
  } else {
    const wl = loadWatchlist();
    stockList = wl.stocks || [];
    console.log(`  Watchlist: ${stockList.length}`);
  }

  if (stockList.length === 0) {
    console.log('  No stocks, skipping');
    return;
  }

  // build fetchers (local DB first)
  const fetchKlines = await makeKlinesFetcher();
  const fetchIndexKlines = () => fetchKlines('1', '000300');

  // force deepseek-chat
  const provider = getProvider('deepseek');
  const apiKey = process.env.DEEPSEEK_API_KEY;
  if (!apiKey) throw new Error('Please set DEEPSEEK_API_KEY environment variable');

  const callLLM = async (prompt) => {
    const r = await provider.call(prompt, { model: 'deepseek-chat', apiKey, maxTokens: 2000 });
    return { text: r.text, usage: r.usage };
  };

  const options = {
    fetchKlines,
    fetchIndexKlines,
    callLLM,
    onProgress: ({ idx, total, code, status }) => {
      if (idx % 50 === 0) console.log(`  Progress: ${idx}/${total}`);
    },
  };

  const result = await runBatchScan(stockList, options);
  console.log('  Complete:', result.succeeded, '/', result.total, 'succeeded');
}

async function generateDailyReport() {
  const { generateDailyReport } = await import('../lib/scanner/daily-report.js');
  const report = await generateDailyReport();
  const date = new Date().toISOString().slice(0, 10);
  const reportDir = path.join(DATA_DIR, 'daily-reports');
  if (!fs.existsSync(reportDir)) fs.mkdirSync(reportDir, { recursive: true });
  const reportPath = path.join(reportDir, `report-${date}.md`);
  fs.writeFileSync(reportPath, report, 'utf-8');
  console.log('  Daily report saved:', reportPath);
}

// ---- sync check ----

async function handleSyncCheck() {
  console.log('=== Sync Status Check ===');
  console.log('');

  const historyPath = path.join(STORAGE_DIR, 'history.json');
  const exists = fs.existsSync(historyPath);

  console.log('Storage directory:', STORAGE_DIR);
  console.log('History file:', exists ? 'exists' : 'not found');

  if (exists) {
    try {
      const stat = fs.statSync(historyPath);
      const hoursAgo = (Date.now() - stat.mtimeMs) / 3600000;
      const raw = fs.readFileSync(historyPath, 'utf-8');
      const entries = JSON.parse(raw);
      console.log('Entries:', Array.isArray(entries) ? entries.length : '?');
      console.log('Last updated:', stat.mtime.toISOString());
      if (hoursAgo > 24) {
        console.warn(`Warning: not updated for ${hoursAgo.toFixed(1)} hours, data may be stale`);
      } else {
        console.log(`Freshness: ${hoursAgo.toFixed(1)} hours ago (OK)`);
      }
    } catch (err) {
      console.warn('Read failed:', err.message);
    }
  } else {
    console.log('');
    console.log('Sync not configured. Follow these steps to set up:');
    console.log('  1. Open chrome://extensions, find this extension');
    console.log('  2. Copy the extension ID (32 lowercase letters)');
    console.log('  3. Run: node native-host/install.js <extensionID>');
    console.log('  4. Restart Chrome');
    console.log('  5. Perform an analysis on an Eastmoney page');
    console.log('  6. Run `ema sync check` again to verify');
  }

  // check all storage files
  if (fs.existsSync(STORAGE_DIR)) {
    const files = fs.readdirSync(STORAGE_DIR).filter(f => f.endsWith('.json'));
    if (files.length > 0) {
      console.log('');
      console.log('Storage file list:');
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
      console.log('db update will be implemented later');
      break;
    }

    default:
      console.error(`Unknown db subcommand: ${sub}`);
      console.log('Available: init | status | update');
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
        console.error('Usage: ema sector alpha <stock_code> [period] [lookback]');
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
          console.log(`${code} has no industry mapping data`);
        } else {
          console.log(JSON.stringify(result, null, 2));
        }
      } finally {
        closeDb();
      }
      break;
    }

    default:
      console.error(`Unknown sector subcommand: ${sub}`);
      console.log('Available: init | alpha <code>');
      process.exit(1);
  }
}

// ---- review ----

async function handleReview(sub, args) {
  const reviewDir = path.join(PROJECT_DIR, '.eastmoney-ai', 'reviews');

  switch (sub) {
    case 'draft': {
      console.log('Immediate draft generation requires Chrome runtime.');
      console.log('Use `ema nightly` to trigger the nightly job, or `ema scheduled` to run the full schedule.');
      break;
    }

    case 'list': {
      if (!fs.existsSync(reviewDir)) {
        console.log('No drafts yet');
        return;
      }
      const files = fs.readdirSync(reviewDir).filter((f) => f.startsWith('draft-')).sort().reverse();
      if (files.length === 0) console.log('No drafts yet');
      else files.forEach((f) => console.log(' ', f));
      break;
    }

    case 'show': {
      const date = args[0] || new Date().toISOString().slice(0, 10);
      const p = path.join(reviewDir, `draft-${date}.md`);
      if (!fs.existsSync(p)) {
        console.error(`Draft not found: ${date}`);
        process.exit(1);
      }
      console.log(fs.readFileSync(p, 'utf-8'));
      break;
    }

    case 'approve': {
      const date = args[0] || new Date().toISOString().slice(0, 10);
      const draftPath = path.join(reviewDir, `draft-${date}.md`);
      if (!fs.existsSync(draftPath)) {
        console.error(`Draft not found: ${date}`);
        process.exit(1);
      }

      const { parseUserReview } = await import('../lib/evaluation/refine.js');
      const { approved } = parseUserReview(draftPath);

      if (approved.length === 0) {
        console.log('No approved suggestions found. In the draft, change [ ] to [x] and mark "approved".');
        console.log(`Draft path: ${draftPath}`);
        process.exit(0);
      }

      console.log(`Found ${approved.length} approved suggestions:`);
      approved.forEach((s, i) => console.log(`  ${i + 1}. ${s}`));
      console.log('');
      console.log('Claude Opus refinement requires an API key. Please set the ANTHROPIC_API_KEY environment variable.');
      break;
    }

    default:
      console.error(`Unknown review subcommand: ${sub}`);
      console.log('Available: draft | list | show <date> | approve <date>');
      process.exit(1);
  }
}

// ---- budget ----

async function handleBudget(sub, args) {
  const { getBudgetSummary, setBudgetConfig } = await import('../lib/evaluation/cost-guard.js');

  switch (sub) {
    case 'show': {
      const summary = getBudgetSummary();
      console.log(`Monthly budget (${summary.currentMonth}):`);
      console.log(`  Spent: ¥${summary.monthSpent} / ¥${summary.monthlyBudget}`);
      console.log(`  Remaining: ¥${summary.monthRemaining}`);
      console.log('Today:');
      console.log(`  Spent: ¥${summary.todaySpent} / ¥${summary.dailyBudget}`);
      console.log(`  Remaining: ¥${summary.todayRemaining}`);
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
        console.log('Usage: ema budget set --monthly 80 --daily 5');
        process.exit(1);
      }
      setBudgetConfig(updates);
      console.log('Budget updated:', updates);
      break;
    }

    default:
      console.error(`Unknown budget subcommand: ${sub}`);
      console.log('Available: show | set --monthly <n> --daily <n>');
      process.exit(1);
  }
}

// build local-first K-line fetcher
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
      console.warn(`  K-line fetch failed ${code}: ${err.message}`);
      return [];
    }
  };
}

// ---- analyze ----

async function handleAnalyze(sub, args) {
  if (!sub) {
    console.error('Usage: ema analyze <stock_code>');
    console.error('Example: ema analyze 600519');
    process.exit(1);
  }

  const code = sub;
  const { execSync } = await import('node:child_process');

  // 1. get stock name
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
    // stocks table may not exist, ignore
  }

  // 2. get monthly kline data for technical indicator computation
  const fetchKlines = await makeKlinesFetcher();
  let klines = [];
  try {
    // market inference: 6-prefix → SH, 0/3-prefix → SZ
    const market = code.startsWith('6') ? '1' : '0';
    klines = await fetchKlines(market, code, 'monthly', 60);
  } catch (err) {
    console.warn(`Failed to fetch K-lines: ${err.message}`);
  }

  // 3. compute technical signals
  let techSignal = '(no data)';
  let techDirection = '';
  if (klines.length >= 20) {
    try {
      const { calculateAll } = await import('../lib/indicators/calculate.js');
      const indicators = calculateAll(klines);
      const summary = summarizeTechnicalSignal(indicators, klines);
      techSignal = summary.text;
      techDirection = summary.direction;
    } catch (err) {
      techSignal = `(indicator calculation failed: ${err.message})`;
    }
  }

  // 4. call Kronos Python CLI
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

  // 5. combined assessment
  let combined = '';
  if (kronosSignal && techDirection) {
    const aiDir = kronosSignal.direction;
    if (aiDir === 'up' && techDirection === 'bull') {
      combined = 'Technical + AI resonance, strongly bullish';
    } else if (aiDir === 'down' && techDirection === 'bear') {
      combined = 'Technical + AI resonance, strongly bearish';
    } else if (aiDir === techDirection) {
      combined = 'Technical + AI aligned, moderately ' + (aiDir === 'up' ? 'bullish' : 'bearish');
    } else if (aiDir === 'flat') {
      combined = 'AI predicts sideways, technical ' + (techDirection === 'bull' ? 'bullish' : 'bearish') + ', direction unclear';
    } else {
      combined = 'Technical and AI signals diverge, suggest wait-and-see';
    }
  } else if (kronosSignal) {
    const aiLabel = { up: 'bullish', down: 'bearish', flat: 'sideways' };
    combined = 'AI only: ' + (aiLabel[kronosSignal.direction] || kronosSignal.direction);
  } else {
    combined = techDirection === 'bull' ? 'Technical only: bullish' :
               techDirection === 'bear' ? 'Technical only: bearish' : 'insufficient signals';
  }

  // 6. output
  const heading = `=== ${code}${stockName ? ' ' + stockName : ''} Monthly Analysis ===`;
  console.log(heading);
  console.log(`[Technical] ${techSignal} → ${techDirection === 'bull' ? 'Bullish' : techDirection === 'bear' ? 'Bearish' : 'Undecided'}`);

  if (kronosSignal) {
    const aiDirLabel = { up: 'Bullish', down: 'Bearish', flat: 'Sideways' };
    console.log(
      `[AI Prediction] ${kronosSignal.pred_len}mo forecast change ${kronosSignal.predicted_change_pct}%, `
      + `confidence ${(kronosSignal.confidence * 100).toFixed(0)}%`
      + ` → ${aiDirLabel[kronosSignal.direction] || kronosSignal.direction}`
    );
  } else {
    console.log(`[AI Prediction] Unavailable: ${kronosError || 'Please download weights and start the service first'}`);
  }

  console.log(`[Combined Assessment] ${combined}`);
}

/** Extract technical summary text from indicator data */
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

  // MA alignment
  if (ma5[last] != null && ma20[last] != null && ma60[last] != null) {
    if (ma5[last] > ma20[last] && ma20[last] > ma60[last]) {
      parts.push('MA bullish alignment');
    } else if (ma5[last] < ma20[last] && ma20[last] < ma60[last]) {
      parts.push('MA bearish alignment');
    } else {
      parts.push('MA intertwined');
    }
  }

  // MACD golden cross / death cross
  if (macdHist && macdHist[last] != null && macdHist[prev] != null) {
    if (macdHist[prev] <= 0 && macdHist[last] > 0) {
      parts.push('MACD golden cross');
    } else if (macdHist[prev] >= 0 && macdHist[last] < 0) {
      parts.push('MACD death cross');
    }
  } else if (indicators.macd?.dif && indicators.macd?.dea) {
    const dif = indicators.macd.dif;
    const dea = indicators.macd.dea;
    if (dif[last] != null && dea[last] != null && dif[prev] != null && dea[prev] != null) {
      if (dif[prev] <= dea[prev] && dif[last] > dea[last]) parts.push('MACD golden cross');
      else if (dif[prev] >= dea[prev] && dif[last] < dea[last]) parts.push('MACD death cross');
    }
  }

  // RSI overbought/oversold
  if (rsi14[last] != null) {
    if (rsi14[last] > 80) parts.push('RSI overbought');
    else if (rsi14[last] < 20) parts.push('RSI oversold');
  }

  // KDJ golden cross / death cross
  if (kdj && kdj.k[last] != null && kdj.d[last] != null) {
    if (kdj.k[prev] <= kdj.d[prev] && kdj.k[last] > kdj.d[last]) parts.push('KDJ golden cross');
    else if (kdj.k[prev] >= kdj.d[prev] && kdj.k[last] < kdj.d[last]) parts.push('KDJ death cross');
  }

  const text = parts.length > 0 ? parts.join(', ') : 'no clear signal';

  // Direction assessment: count bullish/bearish signals
  let bullScore = 0, bearScore = 0;
  if (text.includes('bullish alignment')) bullScore += 2;
  if (text.includes('bearish alignment')) bearScore += 2;
  if (text.includes('golden cross')) bullScore += 1;
  if (text.includes('death cross')) bearScore += 1;
  if (text.includes('oversold')) bullScore += 1;
  if (text.includes('overbought')) bearScore += 1;

  let direction = '';
  if (bullScore > bearScore) direction = 'bull';
  else if (bearScore > bullScore) direction = 'bear';

  return { text, direction };
}

function printUsage() {
  console.log('ema — Eastmoney Monthly AI Analysis CLI');
  console.log('');
  console.log('  ema analyze <code>            Technical indicators + Kronos AI combined assessment');
  console.log('  ema nightly                    Run nightly job');
  console.log('  ema scheduled                  Run daily scheduled tasks');
  console.log('  ema sync check                 Check Native Messaging sync status');
  console.log('  ema review draft               Generate new draft');
  console.log('  ema review list                List all drafts');
  console.log('  ema review show [date]         View draft for a date');
  console.log('  ema review approve [date]      Approve (generate refinement)');
  console.log('  ema db init --scope hs300       Initialize local K-line database');
  console.log('  ema db status                   Database status');
  console.log('  ema sector init                 Fetch Shenwan industry mapping + market cap snapshot');
  console.log('  ema sector init --force         Force rebuild mapping');
  console.log('  ema budget show                 View budget');
  console.log('  ema budget set --monthly 80     Change limit');
}

main();
