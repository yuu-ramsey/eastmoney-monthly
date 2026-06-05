// Cost guard — monthly/daily budget management
// deepseek-chat: input ¥1/M tokens, output ¥4/M tokens
// deepseek-reasoner: input ¥4/M tokens, output ¥16/M tokens

import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..', '..');
const BUDGET_DIR = path.join(PROJECT_DIR, '.eastmoney-ai');
const BUDGET_PATH = path.join(BUDGET_DIR, 'budget.json');

const DEFAULT_CONFIG = {
  monthlyBudgetCny: 50,
  dailyBudgetCny: 3,
  perReviewBudgetCny: 5,
};

function ensureDir() {
  if (!fs.existsSync(BUDGET_DIR)) fs.mkdirSync(BUDGET_DIR, { recursive: true });
}

export class BudgetExceededError extends Error {
  constructor(msg, detail) {
    super(msg);
    this.name = 'BudgetExceededError';
    this.detail = detail;
  }
}

/**
 * Read budget file
 */
export function loadBudget() {
  ensureDir();
  let data;
  try {
    const raw = fs.readFileSync(BUDGET_PATH, 'utf-8');
    data = JSON.parse(raw);
  } catch (_) {
    data = {
      ...DEFAULT_CONFIG,
      currentMonth: new Date().toISOString().slice(0, 7),
      monthSpentCny: 0,
      dayLog: {},
    };
    saveBudget(data);
    return data;
  }

  // Clean expired dayOverrides (entries older than today)
  if (data.dayOverrides && typeof data.dayOverrides === 'object') {
    const today = new Date().toISOString().slice(0, 10);
    let cleaned = false;
    for (const d of Object.keys(data.dayOverrides)) {
      if (d < today) {
        delete data.dayOverrides[d];
        cleaned = true;
      }
    }
    if (cleaned && Object.keys(data.dayOverrides).length === 0) {
      delete data.dayOverrides;
    }
    if (cleaned) saveBudget(data);
  }

  return data;
}

function saveBudget(data) {
  ensureDir();
  fs.writeFileSync(BUDGET_PATH, JSON.stringify(data, null, 2), 'utf-8');
}

/**
 * Auto reset at month start
 */
function ensureMonthReset(data) {
  const now = new Date().toISOString().slice(0, 7);
  if (data.currentMonth !== now) {
    data.currentMonth = now;
    data.monthSpentCny = 0;
    data.dayLog = {};
    saveBudget(data);
  }
  return data;
}

/**
 * Check budget
 * @param {number} estimatedCostCny
 * @returns {{ ok: true, remaining: { daily: number, monthly: number } }}
 * @throws {BudgetExceededError}
 */
export function checkBudget(estimatedCostCny) {
  const data = ensureMonthReset(loadBudget());
  const today = new Date().toISOString().slice(0, 10);
  const todaySpent = data.dayLog[today] || 0;
  // Prefer today's override, otherwise use default daily limit
  const todayBudget = (data.dayOverrides && data.dayOverrides[today]) || data.dailyBudgetCny;

  const dailyAfter = todaySpent + estimatedCostCny;
  if (dailyAfter > todayBudget) {
    throw new BudgetExceededError(
      `今日预算超限：已花 ¥${todaySpent.toFixed(2)}，预估 ¥${estimatedCostCny.toFixed(2)}，日限额 ¥${todayBudget}`,
      { todaySpent, estimatedCostCny, dailyBudget: todayBudget },
    );
  }

  const monthlyAfter = data.monthSpentCny + estimatedCostCny;
  if (monthlyAfter > data.monthlyBudgetCny) {
    throw new BudgetExceededError(
      `月度预算超限：已花 ¥${data.monthSpentCny.toFixed(2)}，预估 ¥${estimatedCostCny.toFixed(2)}，月限额 ¥${data.monthlyBudgetCny}`,
      { monthSpent: data.monthSpentCny, estimatedCostCny, monthlyBudget: data.monthlyBudgetCny },
    );
  }

  return {
    ok: true,
    remaining: {
      daily: +(todayBudget - dailyAfter).toFixed(2),
      monthly: +(data.monthlyBudgetCny - monthlyAfter).toFixed(2),
    },
  };
}

/**
 * Record actual spending
 * @param {number} actualCostCny
 */
export function recordSpending(actualCostCny) {
  const data = ensureMonthReset(loadBudget());
  const today = new Date().toISOString().slice(0, 10);
  data.dayLog[today] = (data.dayLog[today] || 0) + actualCostCny;
  data.monthSpentCny += actualCostCny;
  saveBudget(data);
}

/**
 * Estimate DeepSeek cost
 * @param {number} promptTokens
 * @param {number} completionTokens
 * @param {string} model — 'deepseek-chat' | 'deepseek-reasoner'
 * @returns {number} CNY
 */
export function estimateDeepSeekCost(promptTokens, completionTokens, model = 'deepseek-chat') {
  const pricing = model === 'deepseek-reasoner'
    ? { input: 4, output: 16 }
    : { input: 1, output: 4 };
  return (promptTokens / 1_000_000) * pricing.input + (completionTokens / 1_000_000) * pricing.output;
}

/**
 * Get budget summary
 */
export function getBudgetSummary() {
  const data = ensureMonthReset(loadBudget());
  const today = new Date().toISOString().slice(0, 10);
  const todayOverride = (data.dayOverrides && data.dayOverrides[today]) || null;
  return {
    currentMonth: data.currentMonth,
    monthlyBudget: data.monthlyBudgetCny,
    monthSpent: +data.monthSpentCny.toFixed(4),
    monthRemaining: +(data.monthlyBudgetCny - data.monthSpentCny).toFixed(2),
    todaySpent: +(data.dayLog[today] || 0).toFixed(4),
    todayRemaining: +((todayOverride || data.dailyBudgetCny) - (data.dayLog[today] || 0)).toFixed(2),
    dailyBudget: todayOverride || data.dailyBudgetCny,
    todayOverride,
  };
}

/**
 * Set today's daily budget override (temporary increase, auto-expires next day)
 */
export function setTodayBudget(amountCny) {
  const data = ensureMonthReset(loadBudget());
  const today = new Date().toISOString().slice(0, 10);
  if (!data.dayOverrides) data.dayOverrides = {};
  data.dayOverrides[today] = amountCny;
  saveBudget(data);
  return { date: today, dailyBudget: amountCny };
}

/**
 * Clear today's override
 */
export function clearTodayBudget() {
  const data = ensureMonthReset(loadBudget());
  const today = new Date().toISOString().slice(0, 10);
  if (data.dayOverrides) {
    delete data.dayOverrides[today];
    if (Object.keys(data.dayOverrides).length === 0) delete data.dayOverrides;
  }
  saveBudget(data);
  return { date: today, cleared: true };
}

/**
 * Update budget configuration
 */
export function setBudgetConfig(updates) {
  const data = ensureMonthReset(loadBudget());
  if (updates.monthlyBudgetCny != null) data.monthlyBudgetCny = updates.monthlyBudgetCny;
  if (updates.dailyBudgetCny != null) data.dailyBudgetCny = updates.dailyBudgetCny;
  if (updates.perReviewBudgetCny != null) data.perReviewBudgetCny = updates.perReviewBudgetCny;
  saveBudget(data);
  return data;
}
