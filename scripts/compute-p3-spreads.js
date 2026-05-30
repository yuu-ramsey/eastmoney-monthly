// Compute kronos + LSTM spreads from saved signals
import { readFileSync } from 'fs';
import { rulerRobustCenter, blockBootstrap, bootstrapCI } from '../lib/eval/rulers.js';

const sig = JSON.parse(readFileSync('data/p3-kronos-lstm-signals.json', 'utf-8'));
const v2 = JSON.parse(readFileSync('data/frozen-eval-lowpos-v2-baostock.json', 'utf-8'));
const TRAIN = new Set(['2018-06', '2018-12', '2019-06', '2020-09', '2021-06', '2022-06']);

function evalSignal(obj, name) {
  const ap = [];
  for (const tp of v2.testPoints) {
    if (tp.alpha == null) continue;
    const k = tp.stockCode + '|' + tp.cutoffDate;
    const v = obj[k];
    if (v == null || isNaN(v)) continue;
    ap.push({ stockCode: tp.stockCode, cutoffDate: tp.cutoffDate, alpha: tp.alpha, sv: v, isTrain: TRAIN.has(tp.cutoffDate) });
  }
  if (ap.length === 0) { console.log(name + ': 0 pairs!'); return; }
  const s = [...ap].sort((a, b) => b.sv - a.sv);
  const n20 = Math.floor(ap.length * 0.2);
  for (let i = 0; i < ap.length; i++) s[i].sig = i < n20 ? 'bull' : (i >= ap.length - n20 ? 'bear' : 'neutral');
  const r = rulerRobustCenter(ap, 'sig', 'alpha');
  const boot = blockBootstrap(ap, p => p.stockCode + '|' + p.cutoffDate, s2 => rulerRobustCenter(s2, 'sig', 'alpha').spread);
  const ci = bootstrapCI(boot.values);
  console.log(name + ': n=' + ap.length + ' spread=' + r.spread?.toFixed(2) + '% CI[' + ci.lo.toFixed(1) + ',' + ci.hi.toFixed(1) + ']');
  for (const [label, sub] of [['Train', ap.filter(p => p.isTrain)], ['Test', ap.filter(p => !p.isTrain)]]) {
    const s2 = [...sub].sort((a, b) => b.sv - a.sv);
    const n20_2 = Math.floor(sub.length * 0.2);
    for (let i = 0; i < sub.length; i++) s2[i].sig = i < n20_2 ? 'bull' : (i >= sub.length - n20_2 ? 'bear' : 'neutral');
    const r2 = rulerRobustCenter(sub, 'sig', 'alpha');
    console.log('  ' + label + ': n=' + sub.length + ' spread=' + (r2.spread?.toFixed(2) || 'N/A') + '%');
  }
}

console.log('Kronos:', Object.keys(sig.kronos || {}).length);
console.log('LSTM-old:', Object.keys(sig.lstm || {}).length);
console.log('LSTM-WF:', Object.keys(sig.lstm_wf || {}).length);
console.log('');
evalSignal(sig.kronos || {}, 'Kronos');
console.log('');
evalSignal(sig.lstm || {}, 'LSTM-old');
if (sig.lstm_wf && Object.keys(sig.lstm_wf).length > 0) {
  console.log('');
  evalSignal(sig.lstm_wf, 'LSTM-WF');
}
console.log('\nRef: LLM +8.9% CI[3.7,14.1] | Rev +6.6% CI[1.3,11.2]');
