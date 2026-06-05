// Low-position sample pool builder (survival-only)
// Usage: node lib/eval/dataset-builder-lowpos.js
import Database from 'better-sqlite3';
import { writeFileSync, mkdirSync, existsSync } from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..', '..');
const DB_PATH = path.join(PROJECT_DIR, '.eastmoney-ai', 'db', 'klines-v2.sqlite');
const DATA_DIR = path.join(PROJECT_DIR, 'data');
if (!existsSync(DATA_DIR)) mkdirSync(DATA_DIR, { recursive: true });

const db = new Database(DB_PATH, { readonly: true });

const TIMEPOINTS = ['2018-06','2018-12','2019-06','2020-03','2020-09','2021-06','2021-12','2022-06','2022-10','2023-06','2024-02','2024-10'];
const N_MONTHS = 12, MA_PERIOD = 60, HORIZON = 6, MAX_PER_TP = 140, SEED = 42;

let rng = SEED;
function rand() { rng|=0; rng=rng+0x6D2B79F5|0; let t=Math.imul(rng^rng>>>15,1|rng); t=t+Math.imul(t^t>>>7,61|t)^t; return((t^t>>>14)>>>0)/4294967296; }
function shuffle(a) { for(let i=a.length-1;i>0;i--){const j=Math.floor(rand()*(i+1));[a[i],a[j]]=[a[j],a[i]];} return a; }

const getHist = db.prepare('SELECT date,close FROM monthly_klines WHERE code=? AND date<=? ORDER BY date DESC LIMIT 70');
const getFwd = db.prepare('SELECT date,close,volume FROM monthly_klines WHERE code=? AND date>? ORDER BY date LIMIT 7');

function isLowPos(klines, targetDate, xPct) {
  const ci = klines.findIndex(k=>String(k.date).startsWith(targetDate));
  if(ci<0) return null;
  const close = klines[ci].close;
  if(!close||close<=0.01) return null;
  const past=[];
  for(let i=ci+1;i<klines.length&&past.length<N_MONTHS;i++)
    if(klines[i].close>0.01) past.push(klines[i].close);
  if(past.length<N_MONTHS) return null;
  const mn=Math.min(...past), mx=Math.max(...past);
  const rp = mx>mn ? (close-mn)/(mx-mn) : 0.5;
  let maSum=close, maN=1;
  for(let i=ci+1;i<klines.length&&maN<MA_PERIOD;i++)
    if(klines[i].close>0.01){maSum+=klines[i].close;maN++;}
  return {close,rp,ma60:maSum/maN,cutoffIdx:ci,nMonths:klines.filter(k=>k.close>0.01).length};
}

function forwardAlpha(code, cutoffDate, cutoffClose) {
  const rows = getFwd.all(code, cutoffDate);
  if(rows.length<HORIZON) return {alpha:null,horizonOk:false,suspended:0};
  let end=null,endDate=null,susp=0;
  for(let i=0;i<HORIZON&&i<rows.length;i++){
    if(rows[i].close>0.01){end=rows[i].close;endDate=rows[i].date;}else susp++;
  }
  if(!end) return {alpha:null,horizonOk:false,suspended:susp};
  const ret=(end-cutoffClose)/cutoffClose*100;
  let thin=false;
  for(let i=0;i<Math.min(HORIZON,rows.length);i++)
    if(!rows[i].volume||rows[i].volume<1000){thin=true;break;}
  return {alpha:ret,actualReturn:ret,horizonOk:true,suspendedMonths:susp,endDate,thinTrading:thin};
}

function build(xPct) {
  console.log('\n--- X='+(xPct*100).toFixed(0)+'% ---');
  const tps=[], stocks=new Map(); let id=0; const counts={};
  for(const t of TIMEPOINTS){
    const all=db.prepare('SELECT code,close FROM monthly_klines WHERE date=? AND close>0.01 AND volume>0').all(t);
    const cand=[];
    for(const s of all){
      const kl=getHist.all(s.code,t);
      if(kl.length<12) continue;
      const lp=isLowPos(kl,t,xPct);
      if(!lp||lp.rp>xPct||lp.close>=lp.ma60) continue;
      cand.push({code:s.code,close:lp.close,rp:lp.rp,ma60:lp.ma60,nMonths:lp.nMonths});
    }
    if(cand.length>MAX_PER_TP){shuffle(cand);cand.length=MAX_PER_TP;}
    counts[t]=cand.length;
    for(const c of cand){
      const fwd=forwardAlpha(c.code,t,c.close);
      if(!stocks.has(c.code)) stocks.set(c.code,{code:c.code,market:c.code.startsWith('6')?'1':'0',name:c.code,category:'lowpos'});
      tps.push({id:'lp_'+id++,stockCode:c.code,cutoffDate:t,horizonMonths:HORIZON,
        closeAtCutoff:c.close,rangePosition:+c.rp.toFixed(4),ma60:+c.ma60.toFixed(2),
        alpha:fwd.alpha!=null?+fwd.alpha.toFixed(2):null,
        actualReturn:fwd.actualReturn!=null?+fwd.actualReturn.toFixed(2):null,
        horizonReached:fwd.horizonOk,suspendedMonths:fwd.suspendedMonths,
        thinTrading:fwd.thinTrading,endDate:fwd.endDate||null});
    }
    console.log('  '+t+': '+cand.length);
  }
  const v=tps.filter(tp=>tp.alpha!=null);
  console.log('total:'+tps.length+' with alpha:'+v.length+' stocks:'+stocks.size);
  return {version:'lowpos-v1',createdAt:new Date().toISOString(),universe:'low_position',
    config:{nMonths:N_MONTHS,xPct,maPeriod:MA_PERIOD,horizonMonths:HORIZON,maxPerTimepoint:MAX_PER_TP,seed:SEED,
      note:'survival-only. Alpha biased upward. See docs/p1-survivorship-stress.md.'},
    perTimepointCounts:counts,stocks:[...stocks.values()],testPoints:tps};
}

console.log('=== Low-position pool builder (survival-only) ===');
const main = build(0.20);
writeFileSync(path.join(DATA_DIR,'frozen-eval-lowpos-v1.json'),JSON.stringify(main,null,2),'utf-8');
console.log('\nSaved: frozen-eval-lowpos-v1.json ('+main.testPoints.length+' tp)');

for(const x of[0.10,0.20,0.30]){
  const d=build(x);
  writeFileSync(path.join(DATA_DIR,'frozen-eval-lowpos-x'+(x*100).toFixed(0)+'.json'),JSON.stringify(d,null,2),'utf-8');
}
db.close();
console.log('\nDone.');
