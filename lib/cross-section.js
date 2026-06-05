// Cross-section analysis engine — relative ranking within industry
// Upgrade single-stock absolute prediction to intra-industry relative ranking
import { loadMap, getIndustry, getIndustryStocks } from './industry-map.js';

/**
 * Single-industry analysis
 * @param {Array} scores — [{ code, score, name, ... }] scoreData for the same industry
 * @param {string} industryName
 * @returns {object|null} returns null if sample < 5
 */
export function analyzeIndustry(scores, industryName) {
  if (!scores || scores.length < 5) return null;

  const sorted = [...scores].sort((a, b) => (b.score || 0) - (a.score || 0));
  const values = sorted.map(s => s.score).filter(v => v != null);
  if (values.length === 0) return null;

  // Statistics
  const total = sorted.length;
  const mean = +(values.reduce((a, b) => a + b, 0) / values.length).toFixed(1);
  const sortedVals = [...values].sort((a, b) => a - b);
  const median = sortedVals.length % 2 === 0
    ? +((sortedVals[sortedVals.length / 2 - 1] + sortedVals[sortedVals.length / 2]) / 2).toFixed(1)
    : +sortedVals[Math.floor(sortedVals.length / 2)].toFixed(1);

  // Standard deviation
  const variance = values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length;
  const std = +Math.sqrt(variance).toFixed(1);

  // Strength label (relative to neutral 50)
  const strength_tag = mean >= 60 ? 'strong' : mean >= 45 ? 'neutral' : 'weak';

  // Enrich each stock with industry ranking info
  const items = sorted.map((s, i) => {
    const rank = i + 1;
    return {
      ...s,
      industry_rank: rank,
      industry_total: total,
      industry_percentile: +(rank / total * 100).toFixed(1),
      industry_relative_score: +(s.score - median).toFixed(1),
      industry_median: median,
    };
  });

  return {
    industry: industryName,
    total,
    median_score: median,
    mean_score: mean,
    std_score: std,
    strength_tag,
    top3: sorted.slice(0, 3).map(s => ({ code: s.code, score: s.score, name: s.name })),
    bottom3: sorted.slice(-3).reverse().map(s => ({ code: s.code, score: s.score, name: s.name })),
    items,
  };
}

/**
 * All-industry analysis
 * @param {Array} scores — [{ code, score, name, ... }]
 * @returns {object}
 */
export function analyzeAll(scores) {
  const map = loadMap();
  const industryMap = new Map();

  // Group by industry
  for (const s of scores) {
    const ind = getIndustry(s.code);
    if (!ind) continue;
    if (!industryMap.has(ind)) industryMap.set(ind, []);
    industryMap.get(ind).push(s);
  }

  const industries = {};
  for (const [ind, items] of industryMap) {
    const analysis = analyzeIndustry(items, ind);
    if (analysis) industries[ind] = analysis;
  }

  // Industry ranking
  const ranking = Object.values(industries)
    .map(a => ({ industry: a.industry, mean_score: a.mean_score, strength_tag: a.strength_tag, count: a.total }))
    .sort((a, b) => b.mean_score - a.mean_score);

  // Rotation signals (top-5 pairwise gap between strongest and weakest)
  const rotation_signals = [];
  if (ranking.length >= 2) {
    for (let i = 0; i < Math.min(5, Math.floor(ranking.length / 2)); i++) {
      const strong = ranking[i];
      const weak = ranking[ranking.length - 1 - i];
      rotation_signals.push({
        strong: strong.industry,
        weak: weak.industry,
        strong_score: strong.mean_score,
        weak_score: weak.mean_score,
        gap: +(strong.mean_score - weak.mean_score).toFixed(1),
      });
    }
  }

  return {
    industries,
    ranking,
    rotation_signals,
    coverage: { totalStocks: scores.length, coveredStocks: [...industryMap.values()].reduce((s, a) => s + a.length, 0), industryCount: Object.keys(industries).length },
  };
}

/**
 * Enrich a single scoreData with cross_section field
 */
export function enrichWithCrossSection(scoreData, industryAnalysis) {
  if (!scoreData || !industryAnalysis) return scoreData;

  const items = industryAnalysis.items;
  if (!items) return { ...scoreData, cross_section: null };

  const found = items.find(i => i.code === scoreData.code);
  if (!found) return { ...scoreData, cross_section: null };

  return {
    ...scoreData,
    cross_section: {
      industry: industryAnalysis.industry,
      industry_rank: found.industry_rank,
      industry_total: industryAnalysis.total,
      industry_percentile: found.industry_percentile,
      industry_relative_score: found.industry_relative_score,
      industry_median: industryAnalysis.median_score,
      industry_strength: industryAnalysis.strength_tag,
    },
  };
}
