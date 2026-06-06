// Frozen Eval Dataset loader
// Single entry point; all eval scripts MUST load from here, no self-constructed datasets
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..', '..');
const DATA_DIR = path.join(PROJECT_DIR, 'data');

const DATASETS = {
  v1: 'frozen-eval-dataset-v1.json',
  v2: 'frozen-eval-dataset-v2.json',
};

// Built-in random seed (reproducible)
let subsetSeed = 42;

/**
 * Load frozen eval dataset
 * @param {object} options
 * @param {string} [options.version='v1']
 * @param {number|null} [options.subsetStocks=null] — randomly sample N stocks (for quick experiments), null=all
 * @param {number} [options.seed=42] — random seed
 * @param {string} [options.groundTruthKey='groundTruth'] — which field to use as ground truth (v2 can use 'groundTruthDiscounted')
 * @returns {{ version, baseline, stocks, testPoints, templates, subsetInfo?: {seed, nStocks} }}
 */
export function loadFrozenDataset(options = {}) {
  const { version = 'v1', subsetStocks = null, seed = subsetSeed, groundTruthKey = 'groundTruth' } = options;

  const filename = DATASETS[version];
  if (!filename) throw new Error(`Unknown frozen dataset version: ${version}`);

  const filePath = path.join(DATA_DIR, filename);
  if (!fs.existsSync(filePath)) {
    throw new Error(`Frozen dataset not found: ${filePath}. Run scripts/build-frozen-dataset.js first.`);
  }

  const dataset = JSON.parse(fs.readFileSync(filePath, 'utf-8'));

  // If a different ground truth key is specified, remap
  if (groundTruthKey !== 'groundTruth' && dataset.testPoints.length > 0) {
    const hasKey = dataset.testPoints[0][groundTruthKey] !== undefined;
    if (hasKey) {
      dataset.testPoints = dataset.testPoints.map(tp => ({
        ...tp,
        groundTruth: tp[groundTruthKey],
        _groundTruthSource: groundTruthKey,
      }));
    }
    // Otherwise silently fall back to groundTruth field
  }

  if (subsetStocks != null && subsetStocks < dataset.stocks.length) {
    // Reproducible random sampling
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

/** List available versions */
export function listFrozenVersions() {
  const versions = [];
  for (const [v, file] of Object.entries(DATASETS)) {
    const p = path.join(DATA_DIR, file);
    if (fs.existsSync(p)) versions.push({ version: v, file, path: p });
  }
  return versions;
}

// Simple reproducible PRNG
function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
