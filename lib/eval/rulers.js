// 双尺子 + block bootstrap + 反转因子基线
// 退役 scorePrediction 后的新主指标层
import { readFileSync } from 'fs';

// ====== Block Bootstrap ======
export function blockBootstrap(records, keyFn, statFn, nBootstrap = 10000) {
  const blocks = new Map();
  for (const r of records) {
    const bk = keyFn(r);
    if (!blocks.has(bk)) blocks.set(bk, []);
    blocks.get(bk).push(r);
  }
  const blockList = [...blocks.values()];
  const values = [];
  for (let b = 0; b < nBootstrap; b++) {
    const sample = [];
    for (let i = 0; i < blockList.length; i++)
      sample.push(...blockList[Math.floor(Math.random() * blockList.length)]);
    const v = statFn(sample);
    if (v !== null && !isNaN(v)) values.push(v);
  }
  return { values, nBlocks: blockList.length };
}

export function bootstrapCI(values, alpha = 0.05) {
  const s = [...values].sort((a, b) => a - b);
  const lo = s[Math.floor(s.length * alpha / 2)];
  const hi = s[Math.floor(s.length * (1 - alpha / 2))];
  return { lo, hi, mean: s.reduce((a, v) => a + v, 0) / s.length, n: s.length };
}

// ====== Winsorize ======
export function winsorize(arr, lower = 0.01, upper = 0.99) {
  const s = [...arr].sort((a, b) => a - b);
  const lo = s[Math.floor(s.length * lower)];
  const hi = s[Math.floor(s.length * upper)];
  return arr.map(v => v < lo ? lo : v > hi ? hi : v);
}

// ====== 尺子 1: 稳健中心 ======
export function rulerRobustCenter(records, predKey = 'signal', alphaKey = 'alpha') {
  const valid = records.filter(r => r.alpha != null);
  const getSig = r => r[predKey] || r.signal || 'neutral';
  const result = {};
  for (const b of ['strong_bull', 'bull', 'neutral', 'bear', 'strong_bear']) {
    const subset = valid.filter(r => getSig(r) === b);
    if (subset.length === 0) { result[b] = null; continue; }
    const alphas = subset.map(r => r[alphaKey]);
    const wAlpha = winsorize(alphas);
    result[b] = { n: subset.length, median: percentile(alphas, 0.5), mean: alphas.reduce((a,b2)=>a+b2,0)/alphas.length, winsorizedMean: wAlpha.reduce((a,b2)=>a+b2,0)/wAlpha.length };
  }
  const bullish = valid.filter(r => { const s = getSig(r); return s === 'strong_bull' || s === 'bull'; });
  const bearish = valid.filter(r => { const s = getSig(r); return s === 'strong_bear' || s === 'bear'; });
  const bW = bullish.length > 0 ? winsorize(bullish.map(r=>r[alphaKey])).reduce((a,v)=>a+v,0)/bullish.length : null;
  const beW = bearish.length > 0 ? winsorize(bearish.map(r=>r[alphaKey])).reduce((a,v)=>a+v,0)/bearish.length : null;
  return { buckets: result, bullish: { n: bullish.length, winsorizedMean: bW }, bearish: { n: bearish.length, winsorizedMean: beW }, spread: bW != null && beW != null ? bW - beW : null };
}

// ====== 尺子 2: 尾部捕获 ======
export function rulerTailCapture(records, predKey = 'signal', alphaKey = 'alpha') {
  const valid = records.filter(r => r.alpha != null);
  const alphas = valid.map(r => r.alpha);
  const top20Cut = percentile(alphas, 0.80);
  const pairKey = r => r.id || `${r.stockCode||r.code}|${r.cutoffDate}`;
  const getSig = r => r[predKey] || r.signal || 'neutral';
  const bullish = valid.filter(r => { const s = getSig(r); return s === 'strong_bull' || s === 'bull'; });
  const bigBull = valid.filter(r => r.alpha > 20);
  const top20Set = new Set(valid.filter(r => r.alpha >= top20Cut).map(pairKey));
  return {
    top20CapturePct: top20Set.size > 0 ? bullish.filter(r => top20Set.has(pairKey(r))).length / top20Set.size * 100 : null,
    bigBullHitPct: bigBull.length > 0 ? bigBull.filter(r => { const s = getSig(r); return s === 'strong_bull' || s === 'bull'; }).length / bigBull.length * 100 : null,
    bullishMeanReturn: bullish.length > 0 ? bullish.reduce((s,r)=>s+r.alpha,0)/bullish.length : null,
    nTop20: top20Set.size, nBigBull: bigBull.length,
  };
}

// ====== 复合评估: 同一 bootstrap 块上算两侧 spread 差值 CI ======
export function compareSpreads(recordsA, recordsB, predKeyA, predKeyB, alphaKey = 'alpha', nBoot = 10000) {
  const validA = recordsA.filter(r => r.alpha != null);
  const validB = recordsB.filter(r => r.alpha != null);
  const blockKey = r => `${r.stockCode||r.code}|${r.cutoffDate}`;
  const blocksA = new Map(), blocksB = new Map();
  for (const r of validA) { const bk = blockKey(r); if (!blocksA.has(bk)) blocksA.set(bk, []); blocksA.get(bk).push(r); }
  for (const r of validB) { const bk = blockKey(r); if (!blocksB.has(bk)) blocksB.set(bk, []); blocksB.get(bk).push(r); }
  const allBlocks = [...new Set([...blocksA.keys(), ...blocksB.keys()])];
  const diffs = [];
  for (let b = 0; b < nBoot; b++) {
    const sA = [], sB = [];
    for (let i = 0; i < allBlocks.length; i++) {
      const bk = allBlocks[Math.floor(Math.random() * allBlocks.length)];
      if (blocksA.has(bk)) sA.push(...blocksA.get(bk));
      if (blocksB.has(bk)) sB.push(...blocksB.get(bk));
    }
    if (sA.length === 0 || sB.length === 0) continue;
    const rA = rulerRobustCenter(sA, predKeyA, alphaKey);
    const rB = rulerRobustCenter(sB, predKeyB, alphaKey);
    if (rA.spread != null && rB.spread != null) diffs.push(rA.spread - rB.spread);
  }
  const ci = bootstrapCI(diffs);
  return { values: diffs, ci: { lo: ci.lo, hi: ci.hi, mean: ci.mean }, nBlocks: allBlocks.length };
}

// ====== 反转因子 (纯 cutoff 前数据) ======
export function reversalFactor(klines, cutoffIdx) {
  if (cutoffIdx < 14) return null;
  const close = klines[cutoffIdx].close;
  if (!close || close <= 0.01) return null;
  const rev1m = cutoffIdx >= 1 && klines[cutoffIdx-1].close>0.01 ? (close-klines[cutoffIdx-1].close)/klines[cutoffIdx-1].close : null;
  const rev3m = cutoffIdx >= 3 && klines[cutoffIdx-3].close>0.01 ? (close-klines[cutoffIdx-3].close)/klines[cutoffIdx-3].close : null;
  let maSum = 0, maN = 0;
  for (let i = cutoffIdx; i >= 0 && maN < 60; i--) { if (klines[i].close > 0.01) { maSum += klines[i].close; maN++; } }
  const disc = maN >= 60 ? (close - maSum/maN) / (maSum/maN) : null;
  let gain = 0, loss = 0;
  for (let i = cutoffIdx - 13; i <= cutoffIdx; i++) {
    const d = klines[i].close - klines[i-1].close;
    if (d > 0) gain += d; else loss -= d;
  }
  const rsi = loss > 0 ? 100 - 100 / (1 + gain / loss) : (gain > 0 ? 100 : 50);
  return { rev1m, rev3m, ma60Discount: disc, rsi };
}

export function reversalCompositeZ(factors, refMeans, refStds) {
  if (!factors || !refMeans) return null;
  const keys = ['rev1m', 'rev3m', 'ma60Discount'];
  let z = 0, n = 0;
  for (const k of keys) {
    if (factors[k] != null && refStds[k] > 1e-8) { z += (factors[k] - refMeans[k]) / refStds[k]; n++; }
  }
  return n > 0 ? z / n : null;
}

function percentile(arr, p) {
  const s = [...arr].sort((a,b)=>a-b);
  const idx = p*(s.length-1);
  const lo=Math.floor(idx), hi=Math.ceil(idx);
  return lo===hi ? s[lo] : s[lo]+(s[hi]-s[lo])*(idx-lo);
}
