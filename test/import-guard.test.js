// Runtime import boundary guard test
// Enforces the invariant: "Service Worker static import closure must not contain Node native modules"
//
// Rules:
// - Only track [static] import/export...from declarations
// - [Exclude] dynamic import() -- this project intentionally uses dynamic import()
//   to isolate better-sqlite3, e.g. lib/multi-period/resonance.js:12's
//   await import('../db/klines-repo.js') and native-host/server.js:197's
//   await import('../lib/db/connection.js'). Tracking dynamic imports would false-positive.
// - Bare 'crypto' is not banned -- Chrome Service Worker provides Web Crypto API
//   (crypto.subtle / crypto.randomUUID) which is a legitimate dependency
//
// Limitations (regex approach, acceptable for guard purposes):
// - Multi-line imports are not extracted (all imports in this project are single-line)
// - Import literals inside comments/strings are not excluded (not a practice in this project)
// - Bare specifiers (npm package names) are not tracked -- this project has no browser-side npm deps

import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');

// ---- Banned Node native modules (unavailable in service worker) ----
const BANNED_MODULES = new Set([
  // npm C++ native modules
  'better-sqlite3', 'sqlite3',
  // node: prefixed core modules
  'node:fs', 'node:path', 'node:url', 'node:crypto',
  'node:child_process', 'node:os', 'node:net', 'node:http', 'node:https',
  'node:worker_threads',
  // Bare name core modules (excluding crypto -- Chrome Service Worker provides Web Crypto)
  'fs', 'path', 'child_process', 'os', 'net', 'http', 'https',
  'worker_threads', 'stream', 'util',
]);

// ---- Single-line static import/export extraction ----
// Only matches "import ... from '...'" and "export ... from '...'" single-line declarations
// Dynamic import() calls do not contain the 'from' keyword and are naturally excluded

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

// ---- Relative path resolution (.js completion / index.js resolution) ----
function resolveSpecifier(specifier, fromFile) {
  // Only handle relative paths, npm package names not tracked
  if (!specifier.startsWith('./') && !specifier.startsWith('../')) return null;

  const dir = path.dirname(fromFile);
  let resolved = path.resolve(dir, specifier);

  // .js completion
  if (!resolved.endsWith('.js') && !resolved.endsWith('.mjs')) {
    if (fs.existsSync(resolved + '.js')) {
      resolved += '.js';
    } else if (fs.existsSync(resolved + '.mjs')) {
      resolved += '.mjs';
    } else if (fs.existsSync(resolved) && fs.statSync(resolved).isDirectory()) {
      // Directory -> try index.js
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

// ---- BFS builds static import closure ----
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

// ---- Check if closure contains static import declarations of banned modules ----
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

// ============ Test cases ============

let reachableSet = null;

test('import-guard: build Service Worker static import closure', () => {
  reachableSet = buildReachableSet('background.js');

  // Normalize to / for comparison, avoid Windows \ separator false positives
  const relPaths = [...reachableSet].map((f) => path.relative(ROOT, f).replaceAll('\\', '/'));

  // Core entry must be present
  assert.ok(relPaths.includes('background.js'), 'background.js should be in reachable set');

  // Key lib modules must be present (verify recursive resolution is correct)
  assert.ok(relPaths.some((f) => f.endsWith('build-prompt.js')), 'lib/build-prompt.js should be in reachable set');
  assert.ok(relPaths.some((f) => f.endsWith('agents/runner.js')), 'lib/agents/runner.js should be in reachable set');
  assert.ok(relPaths.some((f) => f.endsWith('llm/index.js')), 'lib/llm/index.js should be in reachable set');
  assert.ok(relPaths.some((f) => f.endsWith('indicators/calculate.js')), 'lib/indicators/calculate.js should be in reachable set');
  assert.ok(relPaths.some((f) => f.endsWith('signals/summary.js')), 'lib/signals/summary.js should be in reachable set');
  assert.ok(relPaths.some((f) => f.endsWith('prompt-templates.js')), 'lib/prompt-templates.js should be in reachable set');
  assert.ok(relPaths.some((f) => f.endsWith('self-backtest.js')), 'lib/self-backtest.js should be in reachable set');
});

test('import-guard: lib/db/ is not in static import closure', () => {
  const relPaths = [...reachableSet].map((f) => path.relative(ROOT, f).replaceAll('\\', '/'));
  const dbFiles = relPaths.filter((f) => f.includes('lib/db/'));
  assert.equal(dbFiles.length, 0,
    `lib/db/ should not be in static closure, found: ${dbFiles.join(', ')}`);
});

test('import-guard: lib/lstm/ is not in static import closure', () => {
  const relPaths = [...reachableSet].map((f) => path.relative(ROOT, f).replaceAll('\\', '/'));
  const lstmFiles = relPaths.filter((f) => f.includes('lib/lstm/'));
  assert.equal(lstmFiles.length, 0,
    `lib/lstm/ should not be in static closure, found: ${lstmFiles.join(', ')}`);
});

test('import-guard: static import closure contains no Node native modules', () => {
  const violations = findBannedStaticImports(reachableSet);

  if (violations.length > 0) {
    const lines = violations.map((v) => `  ${v.file}:${v.line} -> '${v.module}'`);
    assert.fail(`found ${violations.length} violation static import(s):\n${lines.join('\n')}`);
  }

  // Invariant holds
  assert.ok(true);
});

test('import-guard: crypto is not in banned list (Web Crypto API is legitimate)', () => {
  assert.ok(!BANNED_MODULES.has('crypto'),
    'crypto should not be in BANNED_MODULES -- Chrome Service Worker crypto.subtle is a legitimate dependency');
});

test('import-guard: dynamic import is not tracked (design decision verification)', () => {
  // resonance.js uses await import('../db/klines-repo.js')
  // If dynamic import were incorrectly tracked, lib/db/ would appear in closure
  const relPaths = [...reachableSet].map((f) => path.relative(ROOT, f).replaceAll('\\', '/'));
  const dbFiles = relPaths.filter((f) => f.includes('lib/db'));
  assert.equal(dbFiles.length, 0,
    'dynamic import() correctly excluded -- lib/db/ is not in closure');
});
