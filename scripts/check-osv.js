// OSV.dev automatic security scan — 对 package-lock.json 每个依赖查已知 CVE
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const LOCK_PATH = path.resolve(__dirname, '..', 'package-lock.json');

if (!fs.existsSync(LOCK_PATH)) {
  console.log('[osv] package-lock.json 不存在，跳过');
  process.exit(0);
}

const lock = JSON.parse(fs.readFileSync(LOCK_PATH, 'utf-8'));
const packages = lock.packages || {};

// 提取依赖名和版本
const deps = Object.entries(packages)
  .filter(([name]) => name !== '')
  .map(([name, info]) => ({ name: name.replace('node_modules/', ''), version: info.version }));

console.log(`[osv] 检查 ${deps.length} 个依赖...`);

let cveCount = 0;
const batchSize = 10; // OSV API 速率限制友好

for (let i = 0; i < deps.length; i += batchSize) {
  const batch = deps.slice(i, i + batchSize);
  const results = await Promise.allSettled(
    batch.map(async (dep) => {
      const resp = await fetch('https://api.osv.dev/v1/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          package: { name: dep.name, ecosystem: 'npm' },
          version: dep.version,
        }),
        signal: AbortSignal.timeout(5000),
      });
      if (!resp.ok) return null;
      const data = await resp.json();
      return { name: dep.name, version: dep.version, vulns: data.vulns || [] };
    }),
  );

  for (const r of results) {
    if (r.status === 'fulfilled' && r.value && r.value.vulns.length > 0) {
      cveCount++;
      console.log(`\n⚠ ${r.value.name}@${r.value.version}:`);
      for (const v of r.value.vulns) {
        console.log(`  - ${v.id}: ${v.summary || v.details?.slice(0, 100)}`);
        if (v.severity) console.log(`    severity: ${JSON.stringify(v.severity)}`);
      }
    }
  }

  // 速率限制
  if (i + batchSize < deps.length) {
    await new Promise(r => setTimeout(r, 200));
  }
}

console.log(`\n[osv] 完成。${cveCount} 个依赖有已知 CVE。`);
if (cveCount > 0) process.exit(1);
