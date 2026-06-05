// get_money_flow handler test
import { test, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { getMoneyFlowTool } from '../../lib/tools/get-money-flow.js';

const originalFetch = globalThis.fetch;

function mockFetch(status, body) {
  globalThis.fetch = () => Promise.resolve({
    status,
    ok: status >= 200 && status < 300,
    json: () => Promise.resolve(body),
  });
}

afterEach(() => {
  globalThis.fetch = originalFetch;
});

// ---- 正常响应 ----

test('handler: 正常解析资金流数据', async () => {
  mockFetch(200, {
    data: {
      klines: [
        '2025-06-30,15000.5,2000.3,5000.1,10000.4',
        '2025-07-31,-8000.2,3000.1,2000.5,-11000.6',
        '2025-08-31,25000.0,5000.0,8000.0,17000.0',
      ],
    },
  });
  const result = await getMoneyFlowTool.handler({ secid: '1.600522', limit: 12 });
  // 表格存在
  assert.ok(result.includes('主力净流入'));
  assert.ok(result.includes('2025-06'));
  assert.ok(result.includes('2025-07'));
  assert.ok(result.includes('2025-08'));
  // 方向标注
  assert.ok(result.includes('流入'));
  assert.ok(result.includes('流出'));
  // 统计
  assert.ok(result.includes('累计净流入'));
  assert.ok(result.includes('主力净流入为正的月份'));
});

// ---- 空数据 ----

test('handler: klines 为空数组返回提示', async () => {
  mockFetch(200, { data: { klines: [] } });
  const result = await getMoneyFlowTool.handler({ secid: '1.600522' });
  assert.ok(result.includes('暂无资金流数据'));
});

test('handler: data 为 null 返回提示', async () => {
  mockFetch(200, { data: null });
  const result = await getMoneyFlowTool.handler({ secid: '1.600522' });
  assert.ok(result.includes('暂无资金流数据'));
});

// ---- HTTP 错误 ----

test('handler: HTTP 错误返回状态码', async () => {
  mockFetch(500, {});
  const result = await getMoneyFlowTool.handler({ secid: '1.600522' });
  assert.ok(result.includes('HTTP 500'));
});

// ---- limit 参数 ----

test('handler: 默认 limit=12', async () => {
  let capturedUrl = '';
  globalThis.fetch = (url) => {
    capturedUrl = String(url);
    return Promise.resolve({ status: 200, ok: true, json: () => Promise.resolve({ data: { klines: [] } }) });
  };
  await getMoneyFlowTool.handler({ secid: '1.600522' });
  // URL 应包含 lmt=12
  assert.ok(capturedUrl.includes('lmt=12'));
});

test('handler: limit 超 60 截断为 60', async () => {
  let capturedUrl = '';
  globalThis.fetch = (url) => {
    capturedUrl = String(url);
    return Promise.resolve({ status: 200, ok: true, json: () => Promise.resolve({ data: { klines: [] } }) });
  };
  await getMoneyFlowTool.handler({ secid: '1.600522', limit: 100 });
  assert.ok(capturedUrl.includes('lmt=60'));
});

// ---- 输入校验 ----

test('handler: 缺少 secid 返回错误', async () => {
  const result = await getMoneyFlowTool.handler({});
  assert.ok(result.includes('缺少 secid'));
});

// ---- 全是流入 ----

test('handler: 全为正数月份', async () => {
  mockFetch(200, {
    data: {
      klines: [
        '2025-06-30,5000,100,200,4800',
        '2025-07-31,3000,500,300,2800',
      ],
    },
  });
  const result = await getMoneyFlowTool.handler({ secid: '1.600522' });
  assert.ok(result.includes('2/2'));
  assert.ok(result.includes('资金面偏多'));
});

// ---- 全是流出 ----

test('handler: 全为负数月份', async () => {
  mockFetch(200, {
    data: {
      klines: [
        '2025-06-30,-5000,100,-200,-4800',
        '2025-07-31,-3000,500,-300,-2800',
      ],
    },
  });
  const result = await getMoneyFlowTool.handler({ secid: '1.600522' });
  assert.ok(result.includes('资金面偏空'));
});

// ---- 工具定义 ----

test('工具定义: 有 name/description/input_schema/handler', () => {
  assert.equal(getMoneyFlowTool.name, 'get_money_flow');
  assert.equal(typeof getMoneyFlowTool.description, 'string');
  assert.ok(getMoneyFlowTool.description.length > 10);
  assert.equal(getMoneyFlowTool.input_schema.type, 'object');
  assert.ok(getMoneyFlowTool.input_schema.required.includes('secid'));
  assert.equal(typeof getMoneyFlowTool.handler, 'function');
});
