// ema sector init — one-time Shenwan industry mapping + market cap snapshot from legulegu.com
// writes industries + stock_industry_mapping + synthesizes hs300_sector_klines

import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..');

// Shenwan Level-1 31 industries (snapshot verified from legulegu.com 2026-05-17)
const SW_LEVEL1_INDUSTRIES = [
  { code: '801010.SI', name: '农林牧渔' },
  { code: '801030.SI', name: '基础化工' },
  { code: '801040.SI', name: '钢铁' },
  { code: '801050.SI', name: '有色金属' },
  { code: '801080.SI', name: '电子' },
  { code: '801880.SI', name: '汽车' },
  { code: '801110.SI', name: '家用电器' },
  { code: '801120.SI', name: '食品饮料' },
  { code: '801130.SI', name: '纺织服饰' },
  { code: '801140.SI', name: '轻工制造' },
  { code: '801150.SI', name: '医药生物' },
  { code: '801160.SI', name: '公用事业' },
  { code: '801170.SI', name: '交通运输' },
  { code: '801180.SI', name: '房地产' },
  { code: '801200.SI', name: '商贸零售' },
  { code: '801210.SI', name: '社会服务' },
  { code: '801780.SI', name: '银行' },
  { code: '801790.SI', name: '非银金融' },
  { code: '801230.SI', name: '综合' },
  { code: '801710.SI', name: '建筑材料' },
  { code: '801720.SI', name: '建筑装饰' },
  { code: '801730.SI', name: '电力设备' },
  { code: '801890.SI', name: '机械设备' },
  { code: '801740.SI', name: '国防军工' },
  { code: '801750.SI', name: '计算机' },
  { code: '801760.SI', name: '传媒' },
  { code: '801770.SI', name: '通信' },
  { code: '801950.SI', name: '煤炭' },
  { code: '801960.SI', name: '石油石化' },
  { code: '801970.SI', name: '环保' },
  { code: '801980.SI', name: '美容护理' },
];

const LEGULEGU_COMPOSITION = 'https://legulegu.com/stockdata/index-composition?industryCode=';
const FETCH_HEADERS = {
  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
  'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
  'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
};

async function fetchWithRetry(url, maxRetries = 3) {
  let lastErr;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const resp = await fetch(url, { headers: FETCH_HEADERS });
      if (resp.ok) return resp;
      if (resp.status === 429 || resp.status === 502 || resp.status === 504) {
        lastErr = new Error(`HTTP ${resp.status}`);
        if (attempt < maxRetries) {
          await new Promise(r => setTimeout(r, 3000 * (attempt + 1)));
          continue;
        }
        throw lastErr;
      }
      throw new Error(`HTTP ${resp.status}`);
    } catch (err) {
      if (err.message && err.message.startsWith('HTTP')) throw err;
      lastErr = err;
      if (attempt < maxRetries) {
        await new Promise(r => setTimeout(r, 3000 * (attempt + 1)));
      }
    }
  }
  throw lastErr;
}

export async function sectorInit(force = false) {
  const { getDb, closeDb } = await import('../lib/db/connection.js');
  const { initSchema } = await import('../lib/db/schema.js');
  const db = getDb();

  try {
    initSchema(db);

    const existing = db.prepare('SELECT COUNT(*) as n FROM industries').get();
    if (existing.n > 0 && !force) {
      console.log(`Already have ${existing.n} industry mappings. Add --force to rebuild.`);
      return;
    }
    if (force) {
      console.log('--force: clearing existing mappings and rebuilding');
      db.exec('DELETE FROM stock_industry_mapping');
      db.exec('DELETE FROM industries');
    }

    // 1. write industries table
    const snapshotDate = new Date().toISOString().split('T')[0];
    console.log(`Writing 31 Shenwan Level-1 industries (snapshot date: ${snapshotDate})...`);
    const insertIndustry = db.prepare(
      'INSERT OR REPLACE INTO industries (industry_code, industry_name, source, snapshot_date) VALUES (?, ?, ?, ?)'
    );
    db.transaction(() => {
      for (const ind of SW_LEVEL1_INDUSTRIES) {
        insertIndustry.run(ind.code, ind.name, 'sw_l1_legulegu', snapshotDate);
      }
    })();

    // 2. get HS300 stock list
    const hs300Stocks = db.prepare('SELECT code, name, market FROM stocks').all();
    const hs300CodeSet = new Set(hs300Stocks.map(r => r.code));
    const hs300NameMap = {};
    for (const r of hs300Stocks) hs300NameMap[r.code] = r.name;
    console.log(`HS300 constituents: ${hs300CodeSet.size}`);

    // 3. fetch constituents + market cap per industry
    const insertMapping = db.prepare(
      'INSERT OR REPLACE INTO stock_industry_mapping (stock_code, industry_code, stock_name, market_cap, snapshot_date) VALUES (?, ?, ?, ?, ?)'
    );

    let totalMapped = 0;
    const industriesWithData = new Set();
    const failed = [];

    for (const ind of SW_LEVEL1_INDUSTRIES) {
      // resume from checkpoint
      const alreadyDone = db.prepare(
        'SELECT COUNT(*) as n FROM stock_industry_mapping WHERE industry_code = ?'
      ).get(ind.code).n;
      if (alreadyDone > 0 && !force) {
        console.log(`  Skip ${ind.name}(${ind.code}) — already has ${alreadyDone}`);
        industriesWithData.add(ind.code);
        totalMapped += alreadyDone;
        continue;
      }

      console.log(`  Fetching ${ind.name}(${ind.code}) constituents...`);
      try {
        const stocks = await fetchIndustryStocks(ind.code);
        let mapped = 0;
        for (const s of stocks) {
          if (!hs300CodeSet.has(s.code)) continue;
          const stockName = hs300NameMap[s.code] || s.name;
          insertMapping.run(s.code, ind.code, stockName, s.marketCap, snapshotDate);
          mapped++;
        }
        console.log(`    → ${stocks.length} total constituents, ${mapped} in HS300`);
        if (mapped > 0) industriesWithData.add(ind.code);
        totalMapped += mapped;
      } catch (err) {
        console.error(`    ⚠ ${ind.name} fetch failed: ${err.message}`);
        failed.push(ind.name);
      }
      // 3s interval to avoid anti-scraping
      await new Promise(r => setTimeout(r, 3000));
    }

    // 4. completeness report
    const coveragePct = (totalMapped / hs300CodeSet.size * 100).toFixed(1);
    const uncovered = SW_LEVEL1_INDUSTRIES.filter(ind => !industriesWithData.has(ind.code)).map(ind => ind.name);
    console.log(`\nMapping written: ${industriesWithData.size}/${SW_LEVEL1_INDUSTRIES.length} industries, ${totalMapped} constituents`);
    console.log(`HS300 coverage: ${totalMapped}/${hs300CodeSet.size} (${coveragePct}%)`);
    if (uncovered.length > 0) {
      console.log(`Industries with no HS300 constituents (${uncovered.length}): ${uncovered.join(', ')}`);
    }
    if (failed.length > 0) {
      console.log(`Fetch failures (${failed.length}): ${failed.join(', ')}`);
      console.log('Re-run "ema sector init" to resume.');
    }

    // 5. synthesize sector klines
    console.log('\nSynthesizing sector klines (market-cap weighted, current snapshot, lookback <= 12 months)...');
    const { buildAllSectorKlines } = await import('../lib/sector/build-sector-klines.js');
    for (const period of ['monthly', 'weekly', 'daily']) {
      const result = buildAllSectorKlines(db, period, { force });
      console.log(`  ${period}: built ${result.built}, skipped ${result.skipped}${result.errors.length > 0 ? ', failed ' + result.errors.length : ''}`);
    }

    const totalKlines = db.prepare(
      'SELECT period, COUNT(*) as n FROM hs300_sector_klines GROUP BY period'
    ).all();
    console.log('\nSector klines summary:');
    for (const r of totalKlines) console.log(`  ${r.period}: ${r.n} rows`);

  } finally {
    closeDb();
  }
}

async function fetchIndustryStocks(industryCode) {
  const resp = await fetchWithRetry(LEGULEGU_COMPOSITION + industryCode);
  const html = await resp.text();

  const stocks = [];
  // match each row: <tr class="index-basic-composition-item"> ... </tr>
  const rowRe = /<tr class="index-basic-composition-item">([\s\S]*?)<\/tr>/gi;
  let rowMatch;
  while ((rowMatch = rowRe.exec(html)) !== null) {
    const row = rowMatch[1];
    // stock code: 6-digit number followed by .SH or .SZ in the first <a>
    const codeMatch = row.match(/>(\d{6})\.(?:SH|SZ)</);
    if (!codeMatch) continue;
    const code = codeMatch[1];
    // stock name: text of the second <a>
    const nameMatch = row.match(/<a[^>]*>([^<]+)<\/a>/g);
    const name = nameMatch && nameMatch.length >= 2
      ? nameMatch[1].replace(/<[^>]*>/g, '').trim()
      : '';
    // market cap (in 100M CNY): <td class="marketCapItem"> ... digits ... </td>
    const mcMatch = row.match(/marketCapItem">\s*([\d.]+)/);
    const marketCap = mcMatch ? parseFloat(mcMatch[1]) : null;
    if (code && name) {
      stocks.push({ code, name, marketCap });
    }
  }
  return stocks;
}
