// refine 测试 — parseUserReview
import { test, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dir = path.dirname(fileURLToPath(import.meta.url));
const REVIEW_DIR = path.resolve(__dir, '..', '..', '.eastmoney-ai', 'reviews');
const testFile = path.join(REVIEW_DIR, 'draft-test-parse.md');

function writeTestDraft(content) {
  if (!fs.existsSync(REVIEW_DIR)) fs.mkdirSync(REVIEW_DIR, { recursive: true });
  fs.writeFileSync(testFile, content, 'utf-8');
}

afterEach(() => {
  try { fs.unlinkSync(testFile); } catch (_) {}
});

test('parseUserReview: 提取勾选通过的建议', async () => {
  const { parseUserReview } = await import('../../lib/evaluation/refine.js');

  writeTestDraft(`# 草稿复盘 2026-05-14

## 用户审核区
- [x] 建议 1：增加均线偏离度计算 [ 通过 ]
- [ ] 建议 2：优化估值模板 [ 拒绝 ]
- [x] 建议 3：增加资金流分析 [ 通过 ]
`);

  const { approved, rejected } = parseUserReview(testFile);
  assert.equal(approved.length, 2);
  assert.ok(approved[0].includes('均线'));
  assert.ok(approved[1].includes('资金流'));
});

test('parseUserReview: 无勾选返回空', async () => {
  const { parseUserReview } = await import('../../lib/evaluation/refine.js');

  writeTestDraft(`# 草稿

## 用户审核区
- [ ] 建议 1：未审核
- [ ] 建议 2：也待审核
`);

  const { approved } = parseUserReview(testFile);
  assert.equal(approved.length, 0);
});

test('parseUserReview: 修改建议', async () => {
  const { parseUserReview } = await import('../../lib/evaluation/refine.js');

  writeTestDraft(`# 草稿

## 用户审核区
- [x] 建议 1：原建议内容 [ 修改：改为xxx ]
`);

  const { modified } = parseUserReview(testFile);
  assert.equal(modified.length, 1);
});

test('refineWithClaude: 无建议抛错', async () => {
  const { refineWithClaude } = await import('../../lib/evaluation/refine.js');

  await assert.rejects(
    () => refineWithClaude({ approvedSuggestions: [], draftContent: '', callClaude: async () => {}, apiKey: 'x' }),
    /没有审核通过的建议/,
  );
});
