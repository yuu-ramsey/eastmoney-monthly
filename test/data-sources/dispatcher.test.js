// data-sources 测试 — URL拼接、字段映射、降级逻辑
import { test, afterEach } from 'node:test';
import assert from 'node:assert/strict';

const originalFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = originalFetch; });

// ---- eastmoney URL ----
test('eastmoney: URL join correct', async () => {
  let capturedUrl = '';
  globalThis.fetch = (url) => { capturedUrl = url; return Promise.resolve({ status: 200, ok: true, json: () => Promise.resolve({ data: { name: 'test', klines: [] } }) }); };
  try { await (await import('../../lib/data-sources/eastmoney.js')).fetchKlines({ market: '1', code: '600522', period: 'monthly', limit: 30, adjust: 'qfq' }); } catch (_) {}
  assert.ok(capturedUrl.includes('secid=1.600522') && capturedUrl.includes('klt=103') && capturedUrl.includes('lmt=30'));
});

// ---- sina ----
test('sina: sh/sz prefix', async () => {
  let u = '';
  globalThis.fetch = (url) => { u = url; return Promise.resolve({ status: 200, ok: true, json: () => Promise.resolve([]) }); };
  try { await (await import('../../lib/data-sources/sina.js')).fetchKlines({ market: '1', code: '600522', period: 'monthly' }); } catch (_) {}
  assert.ok(u.includes('sh600522'));
});

test('sina: calc amplitude and change%', async () => {
  const data = [
    { day: '2026-04-01', open: '10.0', close: '10.5', high: '11.0', low: '9.5', volume: '1000' },
    { day: '2026-05-01', open: '10.5', close: '9.0', high: '12.0', low: '8.5', volume: '2000' },
  ];
  globalThis.fetch = () => Promise.resolve({ status: 200, ok: true, json: () => Promise.resolve(data) });
  const { fetchKlines } = await import('../../lib/data-sources/sina.js');
  const result = await fetchKlines({ market: '1', code: '600522', period: 'monthly' });
  assert.ok(Math.abs(result.klines[1].changePercent - (-14.29)) < 0.2);
  assert.ok(Math.abs(result.klines[1].amplitude - 33.33) < 0.5);
  assert.equal(result.klines[0].changePercent, null);
});

// ---- tencent ----
test('tencent: data key lookup', async () => {
  const txData = { data: { sz000001: { qfqmonth: [['2026-05', '10', '11', '12', '9', '1000']] } } };
  globalThis.fetch = () => Promise.resolve({ status: 200, ok: true, json: () => Promise.resolve(txData) });
  const { fetchKlines } = await import('../../lib/data-sources/tencent.js');
  const result = await fetchKlines({ market: '0', code: '000001', period: 'monthly' });
  assert.equal(result.klines.length, 1);
  assert.equal(result.sourceUsed, 'tencent');
  assert.equal(result.klines[0].close, 11);
});

// ---- dispatcher: manual fallback chain (不 import dispatcher 模块,原位测试降级逻辑) ----
test('fallback: eastmoney→sina→tencent chain', async () => {
  const klines = Array.from({ length: 20 }, (_, i) => ({ day: `2024-${String(i + 1).padStart(2, '0')}-01`, open: '10', close: '11', high: '12', low: '9', volume: '1000' }));
  const emData = { data: { name: 'stock', klines: Array.from({ length: 30 }, (_, i) => `2024-${String(i + 1).padStart(2, '0')},10,10,11,9,1000,10000,5,2,0.5,3`) } };
  const txData = { data: { sh600522: { qfqmonth: Array.from({ length: 20 }, (_, i) => [`2024-${String(i + 1).padStart(2, '0')}-01`, '10', '11', '12', '9', '1000']) } } };

  const eastmoney = await import('../../lib/data-sources/eastmoney.js');
  const sina = await import('../../lib/data-sources/sina.js');
  const tencent = await import('../../lib/data-sources/tencent.js');

  // Test 1: all succeed → use eastmoney
  globalThis.fetch = () => Promise.resolve({ status: 200, ok: true, json: () => Promise.resolve(emData) });
  const r1 = await eastmoney.fetchKlines({ market: '1', code: '600522', period: 'monthly' });
  assert.equal(r1.sourceUsed, 'eastmoney');
  assert.ok(r1.klines.length >= 12);

  // Test 2: eastmoney 500 → fallback to sina
  let n = 0;
  globalThis.fetch = () => { n++; return n === 1 ? Promise.resolve({ status: 500, ok: false, json: () => Promise.resolve({}) }) : Promise.resolve({ status: 200, ok: true, json: () => Promise.resolve(klines) }); };
  let errCount = 0;
  for (const src of [{ name: 'eastmoney', f: eastmoney.fetchKlines }, { name: 'sina', f: sina.fetchKlines }]) {
    try {
      const r = await src.f(params);
      if (r && Array.isArray(r.klines) && r.klines.length >= 12) {
        assert.equal(src.name, 'sina');
        break;
      }
    } catch (e) { errCount++; continue; }
  }
  assert.equal(errCount, 1);
  globalThis.fetch = originalFetch;

  // Test 3: first two 500 → fallback to tencent
  n = 0;
  globalThis.fetch = () => { n++; return n <= 2 ? Promise.resolve({ status: 500, ok: false, json: () => Promise.resolve({}) }) : Promise.resolve({ status: 200, ok: true, json: () => Promise.resolve(txData) }); };
  let tencentHit = false;
  errCount = 0;
  for (const src of [{ name: 'eastmoney', f: eastmoney.fetchKlines }, { name: 'sina', f: sina.fetchKlines }, { name: 'tencent', f: tencent.fetchKlines }]) {
    try {
      const r = await src.f({ market: '1', code: '600522', period: 'monthly' });
      if (r && Array.isArray(r.klines) && r.klines.length > 0) {
        tencentHit = true;
        break;
      }
    } catch (e) { errCount++; continue; }
  }
  assert.ok(tencentHit);
  assert.equal(errCount, 2);

  // Test 4: all three fail
  n = 0;
  globalThis.fetch = () => { n++; return Promise.resolve({ status: 500, ok: false, json: () => Promise.resolve({}) }); };
  let allFailed = true;
  for (const src of [{ name: 'eastmoney', f: eastmoney.fetchKlines }, { name: 'sina', f: sina.fetchKlines }, { name: 'tencent', f: tencent.fetchKlines }]) {
    try { await src.f({ market: '1', code: '600522', period: 'monthly' }); allFailed = false; break; } catch (e) { continue; }
  }
  assert.ok(allFailed);
});

const params = { market: '1', code: '600522', period: 'monthly' };
