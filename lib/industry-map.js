// Industry mapping — loads data/industry-map.json with in-memory cache
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MAP_PATH = path.resolve(__dirname, '..', 'data', 'industry-map.json');

let _cache = null;

export function loadMap(refresh = false) {
  if (_cache && !refresh) return _cache;
  try {
    _cache = JSON.parse(fs.readFileSync(MAP_PATH, 'utf-8'));
  } catch (_) {
    _cache = { version: '0', industries: {}, stockToIndustry: {} };
  }
  return _cache;
}

export function getIndustry(code) {
  const map = loadMap();
  return map.stockToIndustry[String(code)] || null;
}

export function getIndustryStocks(industry) {
  const map = loadMap();
  return map.industries[industry] || [];
}

export function getAllIndustries() {
  const map = loadMap();
  return Object.keys(map.industries);
}

export function getCoverageStats() {
  const map = loadMap();
  const industries = Object.keys(map.industries);
  let totalStocks = 0;
  for (const ind of industries) {
    totalStocks += map.industries[ind].length;
  }
  return {
    version: map.version,
    industryCount: industries.length,
    totalStocks,
    industries: industries.map(name => ({ name, count: map.industries[name].length })).sort((a, b) => b.count - a.count),
  };
}
