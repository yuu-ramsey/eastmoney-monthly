import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseStockUrl } from '../lib/parse-url.js';

test('parseStockUrl: 沪市标准 URL', () => {
  assert.deepEqual(
    parseStockUrl('https://quote.eastmoney.com/sh600519.html'),
    { market: '1', code: '600519' },
  );
});

test('parseStockUrl: 深市标准 URL', () => {
  assert.deepEqual(
    parseStockUrl('https://quote.eastmoney.com/sz000001.html'),
    { market: '0', code: '000001' },
  );
});

test('parseStockUrl: 大写也能识别', () => {
  assert.deepEqual(
    parseStockUrl('https://quote.eastmoney.com/SH600519.html'),
    { market: '1', code: '600519' },
  );
});

test('parseStockUrl: sh.xxxxxx 点号格式', () => {
  assert.deepEqual(
    parseStockUrl('https://quote.eastmoney.com/sh.600519'),
    { market: '1', code: '600519' },
  );
});

test('parseStockUrl: 无效输入返回 null', () => {
  assert.equal(parseStockUrl('https://www.baidu.com'), null);
  assert.equal(parseStockUrl(''), null);
  assert.equal(parseStockUrl(null), null);
  assert.equal(parseStockUrl(undefined), null);
});

test('parseStockUrl: 代码位数不对', () => {
  // 5 位
  assert.equal(parseStockUrl('https://quote.eastmoney.com/sh60019.html'), null);
});
