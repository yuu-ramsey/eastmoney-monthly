// get_financials handler test
import { test, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { getFinancialsTool } from '../../lib/tools/get-financials.js';

const originalFetch = globalThis.fetch;

function mockFetch(status, body) {
  globalThis.fetch = () => Promise.resolve({
    status,
    ok: status >= 200 && status < 300,
    json: () => Promise.resolve(body),
  });
}

function mockFetchAbort() {
  globalThis.fetch = () => new Promise((_, reject) => {
    const err = new Error('The operation was aborted');
    err.name = 'AbortError';
    reject(err);
  });
}

function mockFetchNetworkError() {
  globalThis.fetch = () => Promise.reject(new Error('fetch failed'));
}

afterEach(() => {
  globalThis.fetch = originalFetch;
});

// ---- Normal response ----

test('handler: 正常解析完整财务指标', async () => {
  mockFetch(200, {
    data: {
      // Eastmoney raw units: f43/f162/f167 need ÷100
      f43: 4374, f57: '600522', f58: '中天科技',
      f162: 2530, f167: 310,
      f116: 150000000000, f117: 120000000000,
      f127: '通信设备',
    },
  });
  const result = await getFinancialsTool.handler({ secid: '1.600522' });
  assert.ok(result.includes('中天科技(600522)'));
  assert.ok(result.includes('43.74'));
  assert.ok(result.includes('25.30'));
  assert.ok(result.includes('3.10'));
  assert.ok(result.includes('1500 亿'));
  assert.ok(result.includes('通信设备'));
});

// ---- Field missing fallback ----

test('handler: 部分字段缺失时标注"该数据暂不可用"', async () => {
  mockFetch(200, {
    data: {
      f43: 1050, f57: '000001', f58: '平安银行',
      // f162 PE missing
      // f167 PB missing
      f116: null,
      f117: 50000000000,
      f127: '',
    },
  });
  const result = await getFinancialsTool.handler({ secid: '0.000001' });
  assert.ok(result.includes('市盈率(TTM): 该数据暂不可用'));
  assert.ok(result.includes('市净率: 该数据暂不可用'));
  assert.ok(result.includes('总市值: 该数据暂不可用'));
  assert.ok(result.includes('行业: 该数据暂不可用'));
  // Some fields are normal
  assert.ok(result.includes('10.50'));
  assert.ok(result.includes('500 亿'));
});

// ---- API error ----

test('handler: data 为 null 返回错误信息', async () => {
  mockFetch(200, { data: null });
  const result = await getFinancialsTool.handler({ secid: '99.999999' });
  assert.ok(result.includes('未找到该股票'));
});

test('handler: HTTP 错误返回状态码', async () => {
  mockFetch(500, {});
  const result = await getFinancialsTool.handler({ secid: '1.600519' });
  assert.ok(result.includes('HTTP 500'));
});

// ---- Network error ----

test('handler: 超时返回超时文本', async () => {
  mockFetchAbort();
  const result = await getFinancialsTool.handler({ secid: '1.600519' });
  assert.ok(result.includes('超时'));
});

test('handler: 网络错误返回网络错误文本', async () => {
  mockFetchNetworkError();
  const result = await getFinancialsTool.handler({ secid: '1.600519' });
  assert.ok(result.includes('网络请求失败'));
});

// ---- Input validation ----

test('handler: 缺少 secid 返回错误', async () => {
  const result = await getFinancialsTool.handler({});
  assert.ok(result.includes('缺少 secid'));
});

test('handler: 空对象返回错误', async () => {
  const result = await getFinancialsTool.handler(null);
  assert.ok(result.includes('缺少 secid'));
});

// ---- Tool definition ----

test('工具定义: 有 name/description/input_schema/handler', () => {
  assert.equal(getFinancialsTool.name, 'get_financials');
  assert.equal(typeof getFinancialsTool.description, 'string');
  assert.ok(getFinancialsTool.description.length > 10);
  assert.equal(getFinancialsTool.input_schema.type, 'object');
  assert.ok(getFinancialsTool.input_schema.required.includes('secid'));
  assert.equal(typeof getFinancialsTool.handler, 'function');
});
