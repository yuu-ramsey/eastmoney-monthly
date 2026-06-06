// MC Dropout JS bridge — reads precomputed MC Dropout uncertainty data
// Dual mode: CLI (child_process calls Python on-demand) / precomputed cache (reads JSON)
// Returns format expected by buildLstmSignalBlock
//
// Usage:
//   import { getMcDropoutData, getMcDropoutForStock } from './lib/lstm/mc-bridge.js';
//   const data = await getMcDropoutForStock('000001', '2026-05-19');

import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..', '..');
const DATA_DIR = path.join(PROJECT_DIR, '.eastmoney-ai', 'lstm');
const CACHE_DIR = path.join(PROJECT_DIR, '.eastmoney-ai', 'storage', 'mc_dropout');
const PARQUET_PATH = path.join(DATA_DIR, 'mc_dropout_signals.parquet');

// Python path (override via PYTHON_PATH env var if needed)
const PYTHON_PATH = process.env.PYTHON_PATH || 'python';

/**
 * Query latest MC Dropout signal for a single stock from parquet file
 * @param {string} code - Stock code, e.g. '000001'
 * @param {string} [date] - Date string 'YYYY-MM-DD', default to latest
 * @returns {Promise<object|null>} signalData expected by buildLstmSignalBlock
 */
export async function getMcDropoutForStock(code, date) {
  try {
    // Check JSON cache first
    if (!date) {
      const cached = readJsonCache(code);
      if (cached) return cached;
    }

    // Call Python to query
    const filterExpr = date
      ? `df[(df['code']=='${code}') & (df['date']=='${date}')]`
      : `df[df['code']=='${code}'].iloc[-1:]`;

    const script = `
import pandas as pd
df = pd.read_parquet('${PARQUET_PATH}')
row = ${filterExpr}
if len(row) == 0:
    print('{}')
else:
    import json
    print(row.iloc[0].to_json())
`;

    const result = await pythonExec(script);
    return formatMcData(JSON.parse(result));
  } catch (e) {
    console.error(`[mc-bridge] Query ${code} failed:`, e.message);
    return null;
  }
}

/**
 * Batch fetch latest MC Dropout signals for multiple stocks
 * @param {string[]} codes
 * @returns {Promise<Map<string, object>>} code → signalData
 */
export async function getMcDropoutBatch(codes) {
  if (!fs.existsSync(PARQUET_PATH)) {
    console.warn('[mc-bridge] mc_dropout_signals.parquet not found, run cli/mc_dropout_predict.py first');
    return new Map();
  }

  try {
    const codeList = JSON.stringify(codes);
    const script = `
import pandas as pd
df = pd.read_parquet('${PARQUET_PATH}')
result = {}
for c in ${codeList}:
    sub = df[df['code'] == c]
    if len(sub) > 0:
        row = sub.iloc[-1]
        result[c] = row.to_dict()
import json
print(json.dumps(result))
`;

    const raw = await pythonExec(script);
    const rawMap = JSON.parse(raw);
    const result = new Map();
    for (const [code, row] of Object.entries(rawMap)) {
      const formatted = formatMcData(row);
      if (formatted) result.set(code, formatted);
    }
    return result;
  } catch (e) {
    console.error('[mc-bridge] Batch query failed:', e.message);
    return new Map();
  }
}

/**
 * Get latest MC Dropout data (all stocks).
 * Suitable for eval runner batch scenarios.
 * @returns {Promise<Map<string, object>>}
 */
export async function getAllLatestMcDropout() {
  if (!fs.existsSync(PARQUET_PATH)) return new Map();

  try {
    const script = `
import pandas as pd
df = pd.read_parquet('${PARQUET_PATH}')
latest = df.groupby('code').last().reset_index()
print(latest.to_json(orient='records'))
`;
    const raw = await pythonExec(script);
    const rows = JSON.parse(raw);
    const result = new Map();
    for (const row of rows) {
      const formatted = formatMcData(row);
      if (formatted) result.set(row.code || row['code'], formatted);
    }
    return result;
  } catch (e) {
    console.error('[mc-bridge] Full query failed:', e.message);
    return new Map();
  }
}

/**
 * Run MC Dropout prediction generation script (full precompute)
 * @param {object} opts - { limit?: number, codes?: string[], all?: boolean }
 * @returns {Promise<boolean>}
 */
export async function runMcDropoutPipeline(opts = {}) {
  const args = [path.join(PROJECT_DIR, 'cli', 'mc_dropout_predict.py')];
  if (opts.all) {
    args.push('--all');
  } else if (opts.codes && opts.codes.length > 0) {
    args.push('--codes', opts.codes.join(','));
  } else if (opts.limit) {
    args.push('--limit', String(opts.limit));
  } else {
    args.push('--limit', '50'); // default
  }
  args.push('--latest'); // latest day only, faster

  return new Promise((resolve) => {
    const py = spawn(PYTHON_PATH, args, { cwd: PROJECT_DIR });
    let stdout = '';
    let stderr = '';
    py.stdout.on('data', (d) => { stdout += d; });
    py.stderr.on('data', (d) => { stderr += d; });
    py.on('close', (code) => {
      if (code === 0) {
        console.log('[mc-bridge] MC Dropout pipeline completed');
        resolve(true);
      } else {
        console.error('[mc-bridge] MC Dropout pipeline failed:', stderr.slice(-500));
        resolve(false);
      }
    });
  });
}

// ---- Internal utility functions ----

function pythonExec(script) {
  return new Promise((resolve, reject) => {
    const py = spawn(PYTHON_PATH, ['-c', script], { cwd: PROJECT_DIR });
    let stdout = '';
    let stderr = '';
    py.stdout.on('data', (d) => { stdout += d; });
    py.stderr.on('data', (d) => { stderr += d; });
    py.on('close', (code) => {
      if (code === 0) resolve(stdout.trim());
      else reject(new Error(stderr.trim() || `exit ${code}`));
    });
  });
}

function readJsonCache(code) {
  const cachePath = path.join(CACHE_DIR, `${code}.json`);
  if (!fs.existsSync(cachePath)) return null;
  try {
    const raw = JSON.parse(fs.readFileSync(cachePath, 'utf-8'));
    return formatMcData(raw);
  } catch {
    return null;
  }
}

/**
 * Format Python dict into the format expected by buildLstmSignalBlock
 */
function formatMcData(row) {
  if (!row || row.overall_confidence == null && row.overall_confidence !== 0) return null;

  const ulevel = row.uncertainty_level || 'medium';
  const levelEmoji = { low: '🟢', medium: '🟡', high: '🔴' };
  const levelDesc = {
    low: 'Strong model prediction consensus, signal has high reliability',
    medium: 'Model predictions diverge, signal should be verified with technical analysis',
    high: 'Model predictions diverge significantly, signal unreliable, prioritize technical analysis',
  };

  return {
    lstm_signal: row.signal,
    lstm_signal_raw: row.signal_raw,
    y3_mean: row.y3_mean,
    y3_std: row.y3_std,
    y6_mean: row.y6_mean,
    y6_std: row.y6_std,
    overall_confidence: row.overall_confidence,
    uncertainty_level: ulevel,
    uncertainty_emoji: levelEmoji[ulevel] || '🟡',
    uncertainty_desc: levelDesc[ulevel] || '',
    mc_samples: 50,
  };
}

// Export format function for test use
export { formatMcData };
