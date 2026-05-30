// Phase C v2: LLM eval on Baostock lowpos pool (with real kline data)
import { readFileSync, appendFileSync, existsSync, mkdirSync } from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..');
const ds = JSON.parse(readFileSync(path.join(PROJECT_DIR, 'data', 'frozen-eval-lowpos-v2-baostock.json'), 'utf-8'));
const kcache = JSON.parse(readFileSync(path.join(PROJECT_DIR, 'data', 'baostock-klines-cache.json'), 'utf-8'));
const TRAIN = new Set(['2018-06','2018-12','2019-06','2020-09','2021-06','2022-06']);
const uniquePairs = new Map();
for (const tp of ds.testPoints) { if (tp.alpha == null) continue; const k = `${tp.stockCode}|${tp.cutoffDate}`; if (!uniquePairs.has(k)) uniquePairs.set(k, tp); }
const pairs = [...uniquePairs.values()];
console.log(`Pairs: ${pairs.length} (Train: ${pairs.filter(p=>TRAIN.has(p.cutoffDate)).length}, Test: ${pairs.filter(p=>!TRAIN.has(p.cutoffDate)).length})`);
const envPath = path.join(PROJECT_DIR, '.env'); const env = {};
if (existsSync(envPath)) for (const line of readFileSync(envPath, 'utf-8').split('\n')) { const t = line.trim(); if (!t || t.startsWith('#')) continue; const eq = t.indexOf('='); if (eq > 0) env[t.slice(0, eq).trim()] = t.slice(eq + 1).trim(); }
const API_KEY = env.DEEPSEEK_API_KEY; if (!API_KEY) { console.error('DEEPSEEK_API_KEY missing'); process.exit(1); }
const RUNS_DIR = path.join(PROJECT_DIR, '.eastmoney-ai', 'eval', 'runs');
if (!existsSync(RUNS_DIR)) mkdirSync(RUNS_DIR, { recursive: true });
const RUN_ID = `lowpos-v2-${new Date().toISOString().slice(0,19).replace(/[T:]/g,'-')}`;
const OUT_PATH = path.join(RUNS_DIR, `${RUN_ID}.jsonl`);
const completed = new Set();
if (existsSync(OUT_PATH)) for (const line of readFileSync(OUT_PATH, 'utf-8').trim().split('\n').filter(Boolean)) { try { const r = JSON.parse(line); completed.add(`${r.stockCode}|${r.cutoffDate}`); } catch (_) {} }
console.log(`Completed: ${completed.size}`);
const pending = pairs.filter(p => !completed.has(`${p.stockCode}|${p.cutoffDate}`));
console.log(`Pending: ${pending.length}`);
if (pending.length === 0) { console.log('All done.'); process.exit(0); }

function getKlines(code, cutoffDate) {
  const fullKey = Object.keys(kcache).find(k=>k.endsWith('.'+code));
  const kl = kcache[code]||(fullKey?kcache[fullKey]:null);
  if(!kl||kl.length<60) return null;
  const ci=kl.findIndex(r=>r[0]===cutoffDate); if(ci<0) return null;
  // Last 12 months of data up to cutoff
  const rows=[];
  for(let i=Math.max(0,ci-12); i<=ci; i++){
    const close=kl[i][1].toFixed(2);
    rows.push(kl[i][0]+' '+close);
  }
  return 'Date Close\n'+rows.join('\n');
}

function buildPrompt(tp) {
  const kl=getKlines(tp.stockCode, tp.cutoffDate);
  if(!kl) return null;
  return '你是A股技术分析师。以下是'+tp.stockCode+'近12+个月月线(前复权):\n'+kl+'\n该股处于低位——过去12月底部20%且<MA60。判断未来6个月方向。\n\n输出JSON:\n```json\n{"signal":"strong_bull|bull|neutral|bear|strong_bear"}\n```\nsignal必五选一。';
}

async function callLLM(prompt) {
  const resp = await fetch('https://api.deepseek.com/chat/completions', { method: 'POST', headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${API_KEY}` }, body: JSON.stringify({ model: 'deepseek-chat', messages: [{ role: 'user', content: prompt }], max_tokens: 4000, temperature: 0.0 }) });
  if (!resp.ok) { const e = await resp.text().catch(()=>''); throw new Error(`HTTP ${resp.status}: ${e.slice(0,200)}`); }
  const d = await resp.json(); return { text: d.choices?.[0]?.message?.content || '', usage: { inputTokens: d.usage?.prompt_tokens || 0, outputTokens: d.usage?.completion_tokens || 0 } };
}
async function retry(p, n=3) { let e; for(let i=0;i<=n;i++){try{return await callLLM(p);}catch(e2){e=e2;if(i<n)await new Promise(r=>setTimeout(r,[1000,4000,16000][i]));}} throw e; }
function parseSignal(t) { try { const m = t.match(/```json\s*([\s\S]*?)```/); if (m) return JSON.parse(m[1].trim()).signal || 'parse_failed'; } catch (_) {} return 'parse_failed'; }

let ok=completed.size, fail=0, totalCost=0; const start=Date.now();
for (let i=0;i<pending.length;i+=2) {
  const batch=pending.slice(i,i+2);
  const results=await Promise.allSettled(batch.map(async tp=>{
    const prompt=buildPrompt(tp);
    if(!prompt) return {stockCode:tp.stockCode,cutoffDate:tp.cutoffDate,signal:'no_kline_data',alpha:tp.alpha,cost:0};
    const {text,usage}=await retry(prompt);
    return {stockCode:tp.stockCode,cutoffDate:tp.cutoffDate,signal:parseSignal(text),alpha:tp.alpha,cost:usage.inputTokens/1e6*1+usage.outputTokens/1e6*4};
  }));
  for(const r of results){ if(r.status==='fulfilled'){appendFileSync(OUT_PATH,JSON.stringify(r.value)+'\n');totalCost+=r.value.cost;ok++;}else{fail++;} }
  if((ok+fail)%40===0||ok+fail===pairs.length) console.log(` ${ok+fail}/${pairs.length} ok=${ok} ¥${totalCost.toFixed(2)} ${((Date.now()-start)/60000).toFixed(1)}min`);
  if(i+2<pending.length) await new Promise(r=>setTimeout(r,300));
}
console.log(`\nDone ¥${totalCost.toFixed(2)} → ${OUT_PATH}`);
