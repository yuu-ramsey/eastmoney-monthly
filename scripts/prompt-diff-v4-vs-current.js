// 步骤 1: v4 promptUsed vs 当前 buildPromptByTemplate 的 diff
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildPromptByTemplate } from '../lib/build-prompt.js';
import { computeMA } from '../lib/compute-ma.js';
import { computeMACD } from '../lib/compute-macd.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..');
const RUNS_DIR = path.join(PROJECT_DIR, '.eastmoney-ai', 'eval', 'runs');

// 1. 提取 v4 第一条 promptUsed
const v4Path = path.join(RUNS_DIR, 'v4-signals-2026-05-17-00-41.jsonl');
const v4Lines = fs.readFileSync(v4Path, 'utf-8').trim().split('\n').filter(Boolean);
const firstRecord = JSON.parse(v4Lines[0]);
const v4Prompt = firstRecord.promptUsed;

console.log('=== v4 第一条元数据 ===');
console.log(`code: ${firstRecord.code}`);
console.log(`cutoffDate: ${firstRecord.cutoffDate}`);
console.log(`template: ${firstRecord.template}`);
console.log(`prompt 长度: ${v4Prompt.length} chars`);

// 2. 用当前代码生成 prompt（需要 K 线数据）
// 从 promptUsed 中提取 K 线表格（重建 klines 数组）
function extractKlinesFromPrompt(prompt) {
  // prompt 格式: "日期\t开盘\t收盘\t最高\t最低\t成交量\t涨跌幅\tMA5\t..."
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
  console.log('FATAL: 无法从 v4 prompt 中提取 K 线数据');
  process.exit(1);
}
console.log(`提取 K 线: ${klines.length} 根`);

// 从 v4 prompt 提取股票名
const nameMatch = v4Prompt.match(/以下是\s*(\S+?)\(/);
const stockName = nameMatch ? nameMatch[1] : '000001';
console.log(`股票名: ${stockName}`);

// 3. 用当前 buildPromptByTemplate 重新生成
const currentPrompt = await buildPromptByTemplate({
  templateKey: firstRecord.template || 'trend',
  name: stockName,
  code: firstRecord.code || '000001',
  klines,
  period: 'monthly',
  provider: 'deepseek',
  decisionMode: false,
});

console.log(`当前 prompt 长度: ${currentPrompt.length} chars`);
console.log(`长度差异: ${currentPrompt.length - v4Prompt.length} chars`);

// 4. 逐段 diff
function extractHARDConstraints(prompt) {
  const match = prompt.match(/## 硬约束\s*\n([\s\S]*?)(?=\n## |\n---\s*\n|$)/);
  return match ? match[1].trim() : '(未找到硬约束段落)';
}

function extractTemplateTasks(prompt) {
  const match = prompt.match(/## 分析任务\s*\n([\s\S]*?)(?=\n## 综合结论|\n## 硬约束|$)/);
  return match ? match[1].trim() : '(未找到分析任务段落)';
}

function extractStructuredOutput(prompt) {
  const match = prompt.match(/## 结构化数据输出\s*\n([\s\S]*?)(?=\n## |\n---|$)/);
  return match ? match[1].trim() : '(未找到结构化输出段落)';
}

function extractPersonalDecision(prompt) {
  const match = prompt.match(/## 个人决策视角[\s\S]*/);
  return match ? match[0].substring(0, 200) + '...' : '(无个人决策段落)';
}

function extractSignalBlock(prompt) {
  const match = prompt.match(/## 已触发的结构化信号[\s\S]*?(?=\n## 综合结论|\n## 极端标签|\n## 硬约束|$)/);
  return match ? match[0].substring(0, 500) : '(无信号段落)';
}

function extractExtremeLabel(prompt) {
  const match = prompt.match(/## 极端标签指引[\s\S]*?(?=\n## |\n---|$)/);
  return match ? match[0].substring(0, 500) : '(无极端标签段落)';
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

// 打印关键段落对比
console.log('\n' + '='.repeat(70));
console.log('=== 段落级 diff ===');
console.log('='.repeat(70));

// HARD_CONSTRAINTS
console.log('\n--- HARD_CONSTRAINTS ---');
const v4HCLines = v4HC.split('\n');
const curHCLines = curHC.split('\n');
console.log(`v4: ${v4HCLines.length} 行`);
console.log(`cur: ${curHCLines.length} 行`);

// 检查特定约束是否存在
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

console.log('\nHARD_CONSTRAINTS 存在性检查:');
for (const check of checks) {
  const inV4 = check.pattern.test(v4HC);
  const inCur = check.pattern.test(curHC);
  const status = (inV4 && inCur) ? '✓ 两边都有' : (!inV4 && !inCur) ? '- 两边都无' : inCur ? '▶ cur新增' : '◀ v4有但cur无';
  if (inV4 !== inCur) {
    console.log(`  **${check.label}**: ${status} **`);
  } else {
    console.log(`  ${check.label}: ${status}`);
  }
}

// 分析任务段落
console.log('\n--- 分析任务段落 ---');
const v4TaskLen = v4Tasks.length;
const curTaskLen = curTasks.length;
console.log(`v4: ${v4TaskLen} chars`);
console.log(`cur: ${curTaskLen} chars`);
console.log(`Δ: ${curTaskLen - v4TaskLen} chars`);

// 结构化输出
console.log('\n--- 结构化输出段落 ---');
console.log(`v4 有: ${v4Structured !== '(未找到结构化输出段落)'}`);
console.log(`cur 有: ${curStructured !== '(未找到结构化输出段落)'}`);
if (v4Structured !== curStructured) {
  console.log('v4 前200字:', v4Structured.substring(0, 200));
  console.log('cur 前200字:', curStructured.substring(0, 200));
}

// 个人决策
console.log('\n--- 个人决策段落 ---');
console.log(`v4 有: ${v4Decision !== '(无个人决策段落)'}`);
console.log(`cur 有: ${curDecision !== '(无个人决策段落)'}`);

// 信号块
console.log('\n--- 结构化信号段落 ---');
console.log(`v4 有: ${v4Signals !== '(无信号段落)'}`);
console.log(`cur 有: ${curSignals !== '(无信号段落)'}`);

// 极端标签
console.log('\n--- 极端标签段落 ---');
console.log(`v4 有: ${v4Extreme !== '(无极端标签段落)'}`);
console.log(`cur 有: ${curExtreme !== '(无极端标签段落)'}`);

// 5. 逐行文本 diff（前 30 处差异）
console.log('\n' + '='.repeat(70));
console.log('=== 文本级 diff (前 30 处差异) ===');
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
  console.log('完全一致! (可能是提取问题,跳过)');
} else {
  console.log(`共 ${diffs.length} 处差异 (显示前 30):`);
  for (const d of diffs.slice(0, 30)) {
    console.log(`\n[行${d.line}]`);
    if (d.old) console.log(`  - ${d.old}`);
    if (d.new) console.log(`  + ${d.new}`);
  }
}

// 最终判定
console.log('\n' + '='.repeat(70));
console.log('=== 最终判定 ===');
console.log('='.repeat(70));

const hcChanges = checks.filter(c => c.pattern.test(v4HC) !== c.pattern.test(curHC));
if (hcChanges.length === 0) {
  console.log('HARD_CONSTRAINTS: 完全一致');
} else {
  hcChanges.forEach(c => {
    const inV4 = c.pattern.test(v4HC);
    const inCur = c.pattern.test(curHC);
    console.log(`HARD_CONSTRAINTS 变化: ${c.label} — v4=${inV4} cur=${inCur}`);
  });
}

console.log(`\n总文本差异: ${diffs.length} 行`);
console.log(`v4 长度: ${v4Prompt.length} cur 长度: ${currentPrompt.length} Δ: ${currentPrompt.length - v4Prompt.length} chars`);
