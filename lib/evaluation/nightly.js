// Nightly job main flow (DeepSeek only, locked to deepseek-chat for cost control)
// Provider does not accept options override, prevents accidental Claude usage

import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { checkBudget, recordSpending, estimateDeepSeekCost, loadBudget, BudgetExceededError } from './cost-guard.js';
import { evaluateBatch } from './collector.js';
import { computeStats } from './draft-review.js';
import { generateDraftReview } from './draft-review.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..', '..');
const DATA_DIR = path.join(PROJECT_DIR, '.eastmoney-ai');
const EVAL_PATH = path.join(DATA_DIR, 'evaluations.jsonl');
const LOG_DIR = path.join(DATA_DIR, 'logs');

function ensureDir(dir) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

// ---- Storage adapter (Node filesystem) ----

const nodeStorage = {
  async getEvaluations() {
    ensureDir(DATA_DIR);
    try {
      const raw = fs.readFileSync(EVAL_PATH, 'utf-8');
      return raw.trim().split('\n').filter(Boolean).map((l) => JSON.parse(l));
    } catch (_) { return []; }
  },
  async saveEvaluation(record) {
    ensureDir(DATA_DIR);
    fs.appendFileSync(EVAL_PATH, JSON.stringify(record) + '\n', 'utf-8');
  },
};

// ---- Logging ----

function log(msg, logFile) {
  const ts = new Date().toISOString();
  const line = `[${ts}] ${msg}`;
  console.log(line);
  if (logFile) fs.appendFileSync(logFile, line + '\n', 'utf-8');
}

// ---- Last review time ----

const LAST_REVIEW_PATH = path.join(DATA_DIR, 'last_review.txt');

function getLastReviewDate() {
  try { return fs.readFileSync(LAST_REVIEW_PATH, 'utf-8').trim(); } catch (_) { return null; }
}

function setLastReviewDate(date) {
  ensureDir(DATA_DIR);
  fs.writeFileSync(LAST_REVIEW_PATH, date, 'utf-8');
}

// ---- Main entry ----

/**
 * Execute nightly job
 * @param {object} opts
 * @param {Array} opts.historyEntries - From chrome.storage.local or file import
 * @param {Function} opts.fetchKlines — (market, code) => klines[]
 * @param {Function} opts.fetchIndexKlines — () => klines[]
 * @param {Function} opts.callDeepSeek — (prompt) => { text, usage }
 */
export async function runNightlyJob(opts) {
  const { historyEntries, fetchKlines, fetchIndexKlines, callDeepSeek } = opts;

  const today = new Date().toISOString().slice(0, 10);
  ensureDir(LOG_DIR);
  const logFile = path.join(LOG_DIR, `nightly-${today}.log`);

  // Force-lock provider/model
  const DEEPSEEK_MODEL = 'deepseek-chat';

  log('=== Nightly job started ===', logFile);

  try {
    // 1. Budget check (estimate starts at ¥1, under daily limit ¥3)
    const estimatedInitCost = 1.0;
    try {
      checkBudget(estimatedInitCost);
    } catch (err) {
      if (err instanceof BudgetExceededError) {
        log(`Budget exceeded, exiting: ${err.message}`, logFile);
        return { status: 'budget_exceeded', message: err.message };
      }
      throw err;
    }
    log('Budget check passed', logFile);

    // 2. Run evaluateBatch (pure computation, free)
    log('Starting evaluateBatch...', logFile);
    const newEvals = await evaluateBatch(historyEntries, { fetchKlines, fetchIndexKlines }, nodeStorage);
    log(`evaluateBatch complete, ${newEvals.length} new evaluations added`, logFile);

    // 3. Trigger condition check
    const allEvals = await nodeStorage.getEvaluations();
    const lastReview = getLastReviewDate();
    const daysSinceReview = lastReview
      ? Math.round((Date.now() - new Date(lastReview).getTime()) / 86400000)
      : 999;

    const shouldTrigger = (
      (allEvals.length >= 50 && daysSinceReview >= 7) ||
      daysSinceReview >= 14
    );

    log(`Total evals:${allEvals.length} days since last review:${daysSinceReview} trigger:${shouldTrigger}`, logFile);

    if (!shouldTrigger) {
      log('Review not triggered, exiting', logFile);
      return { status: 'no_review', newEvals: newEvals.length, totalEvals: allEvals.length };
    }

    // 4. Generate draft review
    log('Review triggered, starting generateDraftReview...', logFile);
    const budget = loadBudget();
    recordSpending(0); // 确保 dayLog 有今天

    const { draftPath, cost } = await generateDraftReview({
      evaluations: allEvals,
      historyEntries,
      callDeepSeek,
      recordSpending: (c) => {
        recordSpending(c);
        log(`DeepSeek draft cost: ¥${c.toFixed(4)}`, logFile);
      },
    });

    setLastReviewDate(today);
    log(`Draft generation complete: ${draftPath}, cost ¥${cost.toFixed(4)}`, logFile);
    log(`Monthly total: ¥${budget.monthSpentCny.toFixed(2)} / ¥${budget.monthlyBudgetCny}`, logFile);

    return { status: 'review_generated', draftPath, cost, newEvals: newEvals.length, totalEvals: allEvals.length };
  } catch (err) {
    log(`Abnormal exit: ${err.message}`, logFile);
    throw err;
  }
}
