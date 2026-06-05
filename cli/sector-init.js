// ema sector init — 从 legulegu.com 一次性抓取申万行业映射 + 市值快照
// 写入 industries + stock_industry_mapping + 合成 hs300_sector_klines

import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..');

// 申万一级行业 31 个（2026-05-17 从 legulegu.com 快照验证）
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
      console.log(`已有 ${existing.n} 个行业映射。如需重建请加 --force`);
      return;
    }
    if (force) {
      console.log('--force: 清除已有映射并重建');
      db.exec('DELETE FROM stock_industry_mapping');
      db.exec('DELETE FROM industries');
    }

    // 1. 写入 industries 表
    const snapshotDate = new Date().toISOString().split('T')[0];
    console.log(`写入 31 个申万一级行业（快照日期: ${snapshotDate}）...`);
    const insertIndustry = db.prepare(
      'INSERT OR REPLACE INTO industries (industry_code, industry_name, source, snapshot_date) VALUES (?, ?, ?, ?)'
    );
    db.transaction(() => {
      for (const ind of SW_LEVEL1_INDUSTRIES) {
        insertIndustry.run(ind.code, ind.name, 'sw_l1_legulegu', snapshotDate);
      }
    })();

    // 2. 获取 HS300 股票列表
    const hs300Stocks = db.prepare('SELECT code, name, market FROM stocks').all();
    const hs300CodeSet = new Set(hs300Stocks.map(r => r.code));
    const hs300NameMap = {};
    for (const r of hs300Stocks) hs300NameMap[r.code] = r.name;
    console.log(`HS300 成分股 ${hs300CodeSet.size} 只`);

    // 3. 逐个行业抓取成分股 + 市值
    const insertMapping = db.prepare(
      'INSERT OR REPLACE INTO stock_industry_mapping (stock_code, industry_code, stock_name, market_cap, snapshot_date) VALUES (?, ?, ?, ?, ?)'
    );

    let totalMapped = 0;
    const industriesWithData = new Set();
    const failed = [];

    for (const ind of SW_LEVEL1_INDUSTRIES) {
      // 断点续抓
      const alreadyDone = db.prepare(
        'SELECT COUNT(*) as n FROM stock_industry_mapping WHERE industry_code = ?'
      ).get(ind.code).n;
      if (alreadyDone > 0 && !force) {
        console.log(`  跳过 ${ind.name}(${ind.code}) — 已有 ${alreadyDone} 只`);
        industriesWithData.add(ind.code);
        totalMapped += alreadyDone;
        continue;
      }

      console.log(`  抓取 ${ind.name}(${ind.code}) 成分股...`);
      try {
        const stocks = await fetchIndustryStocks(ind.code);
        let mapped = 0;
        for (const s of stocks) {
          if (!hs300CodeSet.has(s.code)) continue;
          const stockName = hs300NameMap[s.code] || s.name;
          insertMapping.run(s.code, ind.code, stockName, s.marketCap, snapshotDate);
          mapped++;
        }
        console.log(`    → 共 ${stocks.length} 只成分股，HS300 内 ${mapped} 只`);
        if (mapped > 0) industriesWithData.add(ind.code);
        totalMapped += mapped;
      } catch (err) {
        console.error(`    ⚠ ${ind.name} 抓取失败: ${err.message}`);
        failed.push(ind.name);
      }
      // 间隔 3s 防反爬
      await new Promise(r => setTimeout(r, 3000));
    }

    // 4. 完整性报告
    const coveragePct = (totalMapped / hs300CodeSet.size * 100).toFixed(1);
    const uncovered = SW_LEVEL1_INDUSTRIES.filter(ind => !industriesWithData.has(ind.code)).map(ind => ind.name);
    console.log(`\n映射写入完成：${industriesWithData.size}/${SW_LEVEL1_INDUSTRIES.length} 个行业，${totalMapped} 只成分股`);
    console.log(`HS300 覆盖率: ${totalMapped}/${hs300CodeSet.size} (${coveragePct}%)`);
    if (uncovered.length > 0) {
      console.log(`无 HS300 成分股的行业 (${uncovered.length}): ${uncovered.join(', ')}`);
    }
    if (failed.length > 0) {
      console.log(`抓取失败 (${failed.length}): ${failed.join(', ')}`);
      console.log('可重新运行 ema sector init 续抓');
    }

    // 5. 合成行业 K 线
    console.log('\n合成行业 K 线（市值加权，当前快照，lookback≤12月）...');
    const { buildAllSectorKlines } = await import('../lib/sector/build-sector-klines.js');
    for (const period of ['monthly', 'weekly', 'daily']) {
      const result = buildAllSectorKlines(db, period, { force });
      console.log(`  ${period}: 新建 ${result.built}, 跳过 ${result.skipped}${result.errors.length > 0 ? ', 失败 ' + result.errors.length : ''}`);
    }

    const totalKlines = db.prepare(
      'SELECT period, COUNT(*) as n FROM hs300_sector_klines GROUP BY period'
    ).all();
    console.log('\n行业 K 线汇总:');
    for (const r of totalKlines) console.log(`  ${r.period}: ${r.n} 行`);

  } finally {
    closeDb();
  }
}

async function fetchIndustryStocks(industryCode) {
  const resp = await fetchWithRetry(LEGULEGU_COMPOSITION + industryCode);
  const html = await resp.text();

  const stocks = [];
  // 匹配每行：<tr class="index-basic-composition-item"> ... </tr>
  const rowRe = /<tr class="index-basic-composition-item">([\s\S]*?)<\/tr>/gi;
  let rowMatch;
  while ((rowMatch = rowRe.exec(html)) !== null) {
    const row = rowMatch[1];
    // 股票代码：第一个 <a> 中的 6 位数字.SH 或 .SZ
    const codeMatch = row.match(/>(\d{6})\.(?:SH|SZ)</);
    if (!codeMatch) continue;
    const code = codeMatch[1];
    // 股票名称：第二个 <a> 的文本
    const nameMatch = row.match(/<a[^>]*>([^<]+)<\/a>/g);
    const name = nameMatch && nameMatch.length >= 2
      ? nameMatch[1].replace(/<[^>]*>/g, '').trim()
      : '';
    // 市值（亿元）：<td class="marketCapItem"> ... 数字 ... </td>
    const mcMatch = row.match(/marketCapItem">\s*([\d.]+)/);
    const marketCap = mcMatch ? parseFloat(mcMatch[1]) : null;
    if (code && name) {
      stocks.push({ code, name, marketCap });
    }
  }
  return stocks;
}
