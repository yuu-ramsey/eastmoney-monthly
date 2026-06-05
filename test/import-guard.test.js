// Runtime import boundary guard test
// 锁住"Service Worker 静态 import 闭包不含 Node 原生模块"不变量
//
// 规则：
// - 只跟踪【静态】import/export...from 声明
// - 【排除】动态 import() —— 本项目刻意用动态 import() 隔离 better-sqlite3,
//   例如 lib/multi-period/resonance.js:12 的
//   await import('../db/klines-repo.js') 和 native-host/server.js:197 的
//   await import('../lib/db/connection.js')。若跟踪动态导入会误报。
// - crypto 裸名不禁 —— Chrome Service Worker 提供 Web Crypto API
//   (crypto.subtle / crypto.randomUUID) 是合法依赖
//
// 局限性（正则方案, 对守卫用途可接受）：
// - 多行 import 不会被提取（本项目所有 import 均为单行）
// - 注释/字符串内的 import 字面不会被排除（本项目无此习惯）
// - 裸 specifier（npm 包名）不跟踪 —— 本项目不依赖任何浏览器端 npm 包

import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');

// ---- 禁止的 Node 原生模块 (service worker 不可用) ----
const BANNED_MODULES = new Set([
  // npm C++ 原生模块
  'better-sqlite3', 'sqlite3',
  // node: 前缀的核心模块
  'node:fs', 'node:path', 'node:url', 'node:crypto',
  'node:child_process', 'node:os', 'node:net', 'node:http', 'node:https',
  'node:worker_threads',
  // 裸名核心模块（不含 crypto — Chrome Service Worker 提供 Web Crypto）
  'fs', 'path', 'child_process', 'os', 'net', 'http', 'https',
  'worker_threads', 'stream', 'util',
]);

// ---- 单行静态 import/export 提取 ----
// 只匹配 "import ... from '...'" 和 "export ... from '...'" 单行声明
// import() 动态调用不含 from 关键字, 天然被排除

const IMPORT_LINE_RE = /^\s*import\s+.*\s+from\s+['"]([^'"]+)['"]/;
const EXPORT_LINE_RE = /^\s*export\s+.*\s+from\s+['"]([^'"]+)['"]/;

function extractStaticSpecifiers(filePath) {
  if (!fs.existsSync(filePath)) return [];

  const content = fs.readFileSync(filePath, 'utf-8');
  const specifiers = [];

  for (const line of content.split('\n')) {
    const importMatch = line.match(IMPORT_LINE_RE);
    if (importMatch) { specifiers.push(importMatch[1]); continue; }

    const exportMatch = line.match(EXPORT_LINE_RE);
    if (exportMatch) { specifiers.push(exportMatch[1]); }
  }

  return specifiers;
}

// ---- 相对路径解析 (.js 补全 / index.js 解析) ----
function resolveSpecifier(specifier, fromFile) {
  // 只处理相对路径, npm 包名不跟踪
  if (!specifier.startsWith('./') && !specifier.startsWith('../')) return null;

  const dir = path.dirname(fromFile);
  let resolved = path.resolve(dir, specifier);

  // .js 补全
  if (!resolved.endsWith('.js') && !resolved.endsWith('.mjs')) {
    if (fs.existsSync(resolved + '.js')) {
      resolved += '.js';
    } else if (fs.existsSync(resolved + '.mjs')) {
      resolved += '.mjs';
    } else if (fs.existsSync(resolved) && fs.statSync(resolved).isDirectory()) {
      // 目录 → 尝试 index.js
      const idx = path.join(resolved, 'index.js');
      if (fs.existsSync(idx)) resolved = idx;
      else return null;
    } else {
      return null;
    }
  }

  if (!fs.existsSync(resolved)) return null;
  return resolved;
}

// ---- BFS 构建静态 import 闭包 ----
function buildReachableSet(entryFile) {
  const visited = new Set();
  const queue = [path.resolve(ROOT, entryFile)];

  while (queue.length > 0) {
    const file = queue.shift();

    if (visited.has(file)) continue;
    if (!fs.existsSync(file)) continue;
    visited.add(file);

    for (const spec of extractStaticSpecifiers(file)) {
      const resolved = resolveSpecifier(spec, file);
      if (resolved && !visited.has(resolved)) {
        queue.push(resolved);
      }
    }
  }

  return visited;
}

// ---- 检查闭包内是否存在禁止模块的静态 import 声明 ----
function findBannedStaticImports(reachableSet) {
  const violations = [];
  for (const file of reachableSet) {
    const content = fs.readFileSync(file, 'utf-8');
    const lines = content.split('\n');

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const importMatch = line.match(IMPORT_LINE_RE);
      const exportMatch = line.match(EXPORT_LINE_RE);
      const spec = importMatch ? importMatch[1] : exportMatch ? exportMatch[1] : null;

      if (spec && BANNED_MODULES.has(spec)) {
        violations.push({
          file: path.relative(ROOT, file),
          line: i + 1,
          module: spec,
        });
      }
    }
  }
  return violations;
}

// ============ 测试用例 ============

let reachableSet = null;

test('import-guard: 构建 Service Worker 静态 import 闭包', () => {
  reachableSet = buildReachableSet('background.js');

  // 统一用 / 比较, 避免 Windows \ 分隔符误判
  const relPaths = [...reachableSet].map((f) => path.relative(ROOT, f).replaceAll('\\', '/'));

  // 核心入口必须在
  assert.ok(relPaths.includes('background.js'), 'background.js 应在可达集中');

  // 关键 lib 模块必须在（验证递归解析正确）
  assert.ok(relPaths.some((f) => f.endsWith('build-prompt.js')), 'lib/build-prompt.js 应在可达集中');
  assert.ok(relPaths.some((f) => f.endsWith('agents/runner.js')), 'lib/agents/runner.js 应在可达集中');
  assert.ok(relPaths.some((f) => f.endsWith('llm/index.js')), 'lib/llm/index.js 应在可达集中');
  assert.ok(relPaths.some((f) => f.endsWith('indicators/calculate.js')), 'lib/indicators/calculate.js 应在可达集中');
  assert.ok(relPaths.some((f) => f.endsWith('signals/summary.js')), 'lib/signals/summary.js 应在可达集中');
  assert.ok(relPaths.some((f) => f.endsWith('prompt-templates.js')), 'lib/prompt-templates.js 应在可达集中');
  assert.ok(relPaths.some((f) => f.endsWith('self-backtest.js')), 'lib/self-backtest.js 应在可达集中');
});

test('import-guard: lib/db/ 不在静态 import 闭包中', () => {
  const relPaths = [...reachableSet].map((f) => path.relative(ROOT, f).replaceAll('\\', '/'));
  const dbFiles = relPaths.filter((f) => f.includes('lib/db/'));
  assert.equal(dbFiles.length, 0,
    `lib/db/ 不应在静态闭包中, 发现: ${dbFiles.join(', ')}`);
});

test('import-guard: lib/lstm/ 不在静态 import 闭包中', () => {
  const relPaths = [...reachableSet].map((f) => path.relative(ROOT, f).replaceAll('\\', '/'));
  const lstmFiles = relPaths.filter((f) => f.includes('lib/lstm/'));
  assert.equal(lstmFiles.length, 0,
    `lib/lstm/ 不应在静态闭包中, 发现: ${lstmFiles.join(', ')}`);
});

test('import-guard: 静态 import 闭包不含 Node 原生模块', () => {
  const violations = findBannedStaticImports(reachableSet);

  if (violations.length > 0) {
    const lines = violations.map((v) => `  ${v.file}:${v.line} → '${v.module}'`);
    assert.fail(`发现 ${violations.length} 个违规静态 import:\n${lines.join('\n')}`);
  }

  // 不变量成立
  assert.ok(true);
});

test('import-guard: crypto 不在禁止列表中 (Web Crypto API 合法)', () => {
  assert.ok(!BANNED_MODULES.has('crypto'),
    'crypto 不应在 BANNED_MODULES 中 — Chrome Service Worker 的 crypto.subtle 是合法依赖');
});

test('import-guard: 动态 import 未被跟踪 (设计决策验证)', () => {
  // resonance.js 使用了 await import('../db/klines-repo.js')
  // 如果动态导入被误跟踪, lib/db/ 就会出现在闭包中
  const relPaths = [...reachableSet].map((f) => path.relative(ROOT, f).replaceAll('\\', '/'));
  const dbFiles = relPaths.filter((f) => f.includes('lib/db'));
  assert.equal(dbFiles.length, 0,
    '动态 import() 被正确排除 — lib/db/ 不在闭包中');
});
