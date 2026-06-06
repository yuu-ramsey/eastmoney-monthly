import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseStockUrl } from '../lib/parse-url.js';

test('parseStockUrl: Shanghai standard URL', () => {
  assert.deepEqual(
    parseStockUrl('https://quote.eastmoney.com/sh600519.html'),
    { market: '1', code: '600519' },
  );
});

test('parseStockUrl: Shenzhen standard URL', () => {
  assert.deepEqual(
    parseStockUrl('https://quote.eastmoney.com/sz000001.html'),
    { market: '0', code: '000001' },
  );
});

test('parseStockUrl: uppercase also recognized', () => {
  assert.deepEqual(
    parseStockUrl('https://quote.eastmoney.com/SH600519.html'),
    { market: '1', code: '600519' },
  );
});

test('parseStockUrl: sh.xxxxxx dot format', () => {
  assert.deepEqual(
    parseStockUrl('https://quote.eastmoney.com/sh.600519'),
    { market: '1', code: '600519' },
  );
});

test('parseStockUrl: invalid input returns null', () => {
  assert.equal(parseStockUrl('https://www.baidu.com'), null);
  assert.equal(parseStockUrl(''), null);
  assert.equal(parseStockUrl(null), null);
  assert.equal(parseStockUrl(undefined), null);
});

test('parseStockUrl: incorrect code digit count', () => {
  // 5 digits
  assert.equal(parseStockUrl('https://quote.eastmoney.com/sh60019.html'), null);
});
