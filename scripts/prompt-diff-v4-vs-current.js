// Step 1: v4 promptUsed vs current buildPromptByTemplate diff
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildPromptByTemplate } from '../lib/build-prompt.js';
import { computeMA } from '../lib/compute-ma.js';
import { computeMACD } from '../lib/compute-macd.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..');
const RUNS_DIR = path.join(PROJECT_DIR, '.eastmoney-ai', 'eval', 'runs');

// 1. Extract v4 first promptUsed
const v4Path = path.join(RUNS_DIR, 'v4-signals-2026-05-17-00-41.jsonl');
const v4Lines = fs.readFileSync(v4Path, 'utf-8').trim().split('\n').filter(Boolean);
const firstRecord = JSON.parse(v4Lines[0]);
const v4Prompt = firstRecord.promptUsed;

console.log('=== v4 First Record Metadata ===');
console.log(`code: ${firstRecord.code}`);
console.log(`cutoffDate: ${firstRecord.cutoffDate}`);
console.log(`template: ${firstRecord.template}`);
console.log(`prompt length: ${v4Prompt.length} chars`);

// 2. Generate prompt with current code (needs K-line data)
// Extract K-line table from promptUsed (rebuild klines array)
function extractKlinesFromPrompt(prompt) {
  // prompt format: "Date\tOpen\tClose\tHigh\tLow\tVolume\tChange%\tMA5\t..."
  const lines = prompt.split('\n');
  const tableStart = lines.findIndex(l => l.startsWith('日期\t'));
  if (tableStart < 0) return null;

  const klines = [];
  for (let i = tableStart + 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line || line.startsWith('---') || line.startsWith('以上数据') || line.startsWith('请根据') || line.startsWith('综合结论')) break;
    // Skip prompt instruction lines that don't look like data
    if (line.startsWith('你是') || line.startsWith('请') || line.startsWith('1.') || line.startsWith('2.') || line.startsWith('3.')) break;

    const parts = line.split('\t');
    if (parts.length < 7) continue;

    const date = parts[0];
    const open = parseFloat(parts[1]);
    const close = parseFloat(parts[2]);
    const high = parseFloat(parts[3]);
    const low = parseFloat(parts[4]);
    const volume = parseFloat(parts[5]);
    const change = parseFloat(parts[6]);

    if (isNaN(open) || isNaN(close)) continue;

    const ma5 = parts.length > 7 && parts[7] !== '-' ? parseFloat(parts[7]) || null : null;
    const ma20 = parts.length > 8 && parts[8] !== '-' ? parseFloat(parts[8]) || null : null;
    const ma60 = parts.length > 9 && parts[9] !== '-' ? parseFloat(parts[9]) || null : null;
    const dif = parts.length > 10 && parts[10] !== '-' ? parseFloat(parts[10]) || null : null;
    const dea = parts.length > 11 && parts[11] !== '-' ? parseFloat(parts[11]) || null : null;
    const hist = parts.length > 12 && parts[12] !== '-' ? parseFloat(parts[12]) || null : null;
    const turnover = parts.length > 13 && parts[13] !== '-' ? parseFloat(parts[13]) || null : null;

    klines.push({ date, open, close, high, low, volume, change, ma5, ma20, ma60, dif, dea, hist, turnover });
  }
  return klines;
}

const klines = extractKlinesFromPrompt(v4Prompt);
if (!klines || klines.length === 0) {
  console.log('FATAL: Cannot extract K-line data from v4 prompt');
  process.exit(1);
}
console.log(`Extracted K-lines: ${klines.length} bars`);

// Extract stock name from v4 prompt
const nameMatch = v4Prompt.match(/以下是\s*(\S+?)\(/);
const stockName = nameMatch ? nameMatch[1] : '000001';
console.log(`Stock name: ${stockName}`);

// 3. Regenerate with current buildPromptByTemplate
const currentPrompt = await buildPromptByTemplate({
  templateKey: firstRecord.template || 'trend',
  name: stockName,
  code: firstRecord.code || '000001',
  klines,
  period: 'monthly',
  provider: 'deepseek',
  decisionMode: false,
});

console.log(`Current prompt length: ${currentPrompt.length} chars`);
console.log(`Length difference: ${currentPrompt.length - v4Prompt.length} chars`);

// 4. Section-by-section diff
function extractHARDConstraints(prompt) {
  const match = prompt.match(/## 硬约束\s*\n([\s\S]*?)(?=\n## |\n---\s*\n|$)/);
  return match ? match[1].trim() : '(HARD_CONSTRAINTS section not found)';
}

function extractTemplateTasks(prompt) {
  const match = prompt.match(/## 分析任务\s*\n([\s\S]*?)(?=\n## 综合结论|\n## 硬约束|$)/);
  return match ? match[1].trim() : '(Analysis tasks section not found)';
}

function extractStructuredOutput(prompt) {
  const match = prompt.match(/## 结构化数据输出\s*\n([\s\S]*?)(?=\n## |\n---|$)/);
  return match ? match[1].trim() : '(Structured output section not found)';
}

function extractPersonalDecision(prompt) {
  const match = prompt.match(/## 个人决策视角[\s\S]*/);
  return match ? match[0].substring(0, 200) + '...' : '(No personal decision section)';
}

function extractSignalBlock(prompt) {
  const match = prompt.match(/## 已触发的结构化信号[\s\S]*?(?=\n## 综合结论|\n## 极端标签|\n## 硬约束|$)/);
  return match ? match[0].substring(0, 500) : '(No signal section)';
}

function extractExtremeLabel(prompt) {
  const match = prompt.match(/## 极端标签指引[\s\S]*?(?=\n## |\n---|$)/);
  return match ? match[0].substring(0, 500) : '(No extreme label section)';
}

const v4HC = extractHARDConstraints(v4Prompt);
const curHC = extractHARDConstraints(currentPrompt);

const v4Tasks = extractTemplateTasks(v4Prompt);
const curTasks = extractTemplateTasks(currentPrompt);

const v4Structured = extractStructuredOutput(v4Prompt);
const curStructured = extractStructuredOutput(currentPrompt);

const v4Decision = extractPersonalDecision(v4Prompt);
const curDecision = extractPersonalDecision(currentPrompt);

const v4Signals = extractSignalBlock(v4Prompt);
const curSignals = extractSignalBlock(currentPrompt);

const v4Extreme = extractExtremeLabel(v4Prompt);
const curExtreme = extractExtremeLabel(currentPrompt);

// Print key section comparison
console.log('\n' + '='.repeat(70));
console.log('=== Section-level diff ===');
console.log('='.repeat(70));

// HARD_CONSTRAINTS
console.log('\n--- HARD_CONSTRAINTS ---');
const v4HCLines = v4HC.split('\n');
const curHCLines = curHC.split('\n');
console.log(`v4: ${v4HCLines.length} lines`);
console.log(`cur: ${curHCLines.length} lines`);

// Check if specific constraints exist
const checks = [
  { pattern: /收盘.*K线.*含.*MA/, label: '#1 收盘K线含MA/MACD' },
  { pattern: /数字依据|反方.*观点/, label: '#2 数字依据+反方观点' },
  { pattern: /3.*定式|均线.*排列.*MACD.*位置.*成交量.*趋势/, label: '#3 三定式' },
  { pattern: /数据窗口|data.*window/i, label: '#4 数据窗口标注' },
  { pattern: /禁止表外|表外|不在.*表格/, label: '#5 禁止表外数据' },
  { pattern: /已收盘.*K.*线|区分.*收盘/, label: '#6 区分已收盘K线' },
  { pattern: /结构化.*JSON|score.*signal|```json/, label: '#7 结构化JSON输出' },
  { pattern: /禁止.*自行.*计算.*MA|禁止.*MACD.*自算|不要.*自行.*计算.*指标/, label: '#8 禁止自算指标' },
  { pattern: /极端.*标签.*触发|strong_bull.*触发|strong_bear.*触发/, label: '#9 极端标签指引' },
  { pattern: /signal.*一致|信号.*覆盖|factory.*signal/i, label: '#10 signal一致性' },
  { pattern: /共振|resonance|多周期.*共振|跨周期/, label: '#11 多周期共振' },
  { pattern: /sector.*alpha|行业.*alpha|行业内.*排名|申万/, label: '#12 sector alpha' },
];

console.log('\nHARD_CONSTRAINTS existence check:');
for (const check of checks) {
  const inV4 = check.pattern.test(v4HC);
  const inCur = check.pattern.test(curHC);
  const status = (inV4 && inCur) ? '✓ both have' : (!inV4 && !inCur) ? '- neither has' : inCur ? '▶ cur added' : '◀ v4 has, cur missing';
  if (inV4 !== inCur) {
    console.log(`  **${check.label}**: ${status} **`);
  } else {
    console.log(`  ${check.label}: ${status}`);
  }
}

// Analysis tasks section
console.log('\n--- Analysis Tasks Section ---');
const v4TaskLen = v4Tasks.length;
const curTaskLen = curTasks.length;
console.log(`v4: ${v4TaskLen} chars`);
console.log(`cur: ${curTaskLen} chars`);
console.log(`Δ: ${curTaskLen - v4TaskLen} chars`);

// Structured output
console.log('\n--- Structured Output Section ---');
console.log(`v4 has: ${v4Structured !== '(Structured output section not found)'}`);
console.log(`cur has: ${curStructured !== '(Structured output section not found)'}`);
if (v4Structured !== curStructured) {
  console.log('v4 first 200 chars:', v4Structured.substring(0, 200));
  console.log('cur first 200 chars:', curStructured.substring(0, 200));
}

// Personal decision
console.log('\n--- Personal Decision Section ---');
console.log(`v4 has: ${v4Decision !== '(No personal decision section)'}`);
console.log(`cur has: ${curDecision !== '(No personal decision section)'}`);

// Signal block
console.log('\n--- Structured Signal Section ---');
console.log(`v4 has: ${v4Signals !== '(No signal section)'}`);
console.log(`cur has: ${curSignals !== '(No signal section)'}`);

// Extreme labels
console.log('\n--- Extreme Labels Section ---');
console.log(`v4 has: ${v4Extreme !== '(No extreme label section)'}`);
console.log(`cur has: ${curExtreme !== '(No extreme label section)'}`);

// 5. Line-by-line text diff (first 30 differences)
console.log('\n' + '='.repeat(70));
console.log('=== Text-level diff (first 30 differences) ===');
console.log('='.repeat(70));

function simpleDiff(oldText, newText) {
  const oldLines = oldText.split('\n');
  const newLines = newText.split('\n');
  const maxLen = Math.max(oldLines.length, newLines.length);
  const diffs = [];

  for (let i = 0; i < maxLen; i++) {
    const ol = oldLines[i] || '';
    const nl = newLines[i] || '';
    if (ol !== nl && diffs.length < 30) {
      diffs.push({ line: i + 1, old: ol.substring(0, 150), new: nl.substring(0, 150) });
    }
  }
  return diffs;
}

const diffs = simpleDiff(v4Prompt, currentPrompt);
if (diffs.length === 0) {
  console.log('Identical! (may be extraction issue, skipping)');
} else {
  console.log(`Total ${diffs.length} differences (showing first 30):`);
  for (const d of diffs.slice(0, 30)) {
    console.log(`\n[Line ${d.line}]`);
    if (d.old) console.log(`  - ${d.old}`);
    if (d.new) console.log(`  + ${d.new}`);
  }
}

// Final verdict
console.log('\n' + '='.repeat(70));
console.log('=== Final Verdict ===');
console.log('='.repeat(70));

const hcChanges = checks.filter(c => c.pattern.test(v4HC) !== c.pattern.test(curHC));
if (hcChanges.length === 0) {
  console.log('HARD_CONSTRAINTS: identical');
} else {
  hcChanges.forEach(c => {
    const inV4 = c.pattern.test(v4HC);
    const inCur = c.pattern.test(curHC);
    console.log(`HARD_CONSTRAINTS change: ${c.label} — v4=${inV4} cur=${inCur}`);
  });
}

console.log(`\nTotal text differences: ${diffs.length} lines`);
console.log(`v4 length: ${v4Prompt.length} cur length: ${currentPrompt.length} Δ: ${currentPrompt.length - v4Prompt.length} chars`);
