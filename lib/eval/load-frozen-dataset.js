// Frozen Eval Dataset 加载器
// 单一入口，所有 eval 脚本必须从此加载，禁止自行构造数据集
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..', '..');
const DATA_DIR = path.join(PROJECT_DIR, 'data');

const DATASETS = {
  v1: 'frozen-eval-dataset-v1.json',
};

// 内置随机种子（可复现）
let subsetSeed = 42;

/**
 * 加载 frozen eval dataset
 * @param {object} options
 * @param {string} [options.version='v1']
 * @param {number|null} [options.subsetStocks=null] — 随机抽 N 只股票（用于快速实验），null=全量
 * @param {number} [options.seed=42] — 随机种子
 * @returns {{ version, baseline, stocks, testPoints, templates, subsetInfo?: {seed, nStocks} }}
 */
export function loadFrozenDataset(options = {}) {
  const { version = 'v1', subsetStocks = null, seed = subsetSeed } = options;

  const filename = DATASETS[version];
  if (!filename) throw new Error(`Unknown frozen dataset version: ${version}`);

  const filePath = path.join(DATA_DIR, filename);
  if (!fs.existsSync(filePath)) {
    throw new Error(`Frozen dataset not found: ${filePath}. Run scripts/build-frozen-dataset.js first.`);
  }

  const dataset = JSON.parse(fs.readFileSync(filePath, 'utf-8'));

  if (subsetStocks != null && subsetStocks < dataset.stocks.length) {
    // 可复现随机抽样
    const rng = mulberry32(seed);
    const shuffled = [...dataset.stocks].sort(() => rng() - 0.5);
    const selected = shuffled.slice(0, subsetStocks);
    const selectedCodes = new Set(selected.map(s => s.code));

    const filteredTP = dataset.testPoints.filter(tp => selectedCodes.has(tp.stockCode));

    console.log(`[frozen-dataset] subset=${subsetStocks}/${dataset.stocks.length} stocks, seed=${seed}, testPoints=${filteredTP.length}`);

    return {
      ...dataset,
      stocks: selected,
      testPoints: filteredTP,
      subsetInfo: { seed, nStocks: subsetStocks, totalStocks: dataset.stocks.length },
    };
  }

  return dataset;
}

/** 列出可用版本 */
export function listFrozenVersions() {
  const versions = [];
  for (const [v, file] of Object.entries(DATASETS)) {
    const p = path.join(DATA_DIR, file);
    if (fs.existsSync(p)) versions.push({ version: v, file, path: p });
  }
  return versions;
}

// 简单可复现 PRNG
function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
