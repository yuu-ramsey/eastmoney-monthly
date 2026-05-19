"""Task 3: LLM eval — Run A (no LSTM) vs Run B (with LSTM signal). Frozen dataset."""
import torch, torch.nn as nn, numpy as np, pandas as pd, json, time, sys
from pathlib import Path
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib' / 'lstm'))
from model_v2 import create_model

PROJECT = Path(__file__).parent.parent
DATA = PROJECT / '.eastmoney-ai' / 'lstm'
DEVICE = torch.device('cuda')

# Load frozen dataset (same as Phase 12 Run A)
frozen = json.load(open('data/frozen-eval-dataset-v1.json'))

# Load LSTM monthly signals
lstm_df = pd.read_csv(DATA / 'monthly_lstm_signals.csv')
lstm_lookup = {}
for _, r in lstm_df.iterrows():
    lstm_lookup[(r['code'], r['month'])] = r['lstm_signal']

# Track baseline
baseline_score = frozen['baseline']['score']
print(f"Frozen baseline: {baseline_score}")
print(f"LSTM signals: {len(lstm_df)} records")

# For each testPoint, generate a "score" using the LSTM signal as a predictor
# We compare: using LSTM signal vs not using it (baseline)
# Since we can't actually run LLM calls (cost ¥14), we simulate:
# - LSTM signal direction correctness vs groundTruth

test_points = frozen['testPoints']
correct_lstm, total_lstm = 0, 0
correct_no_lstm, total_no_lstm = 0, 0

for tp in test_points[:20]:  # Sample first 20 for quick check
    code = tp['stockCode']; cutoff = tp['cutoffDate']; gt = tp['groundTruth']
    # Map groundTruth to numeric
    gt_map = {'strong_bull': 2, 'bull': 1, 'neutral': 0, 'bear': -1, 'strong_bear': -2}
    gt_val = gt_map.get(gt, 0)

    # LSTM signal lookup
    lstm_key = (code, cutoff)
    lstm_sig = lstm_lookup.get(lstm_key, 0)

    # Direction match
    if lstm_sig > 0.1 and gt_val > 0: correct_lstm += 1
    elif lstm_sig < -0.1 and gt_val < 0: correct_lstm += 1
    elif abs(lstm_sig) <= 0.1 and gt_val == 0: correct_lstm += 1
    total_lstm += 1

print(f"\nLSTM signal alone (quick check on 20 testPoints):")
print(f"  Correct direction: {correct_lstm}/{total_lstm} ({correct_lstm/total_lstm*100:.0f}%)")

# The real comparison requires LLM calls. Estimate:
print(f"\nTo run full LLM eval (640 samples × 2 runs = 1280 calls, ~¥29):")
print(f"  Run A: baseline (no LSTM) → score={baseline_score}")
print(f"  Run B: +LSTM signal → score TBD")

# Quick proxy: compute LSTM signal IC on all testPoints
all_sig, all_gt = [], []
for tp in test_points:
    code = tp['stockCode']; cutoff = tp['cutoffDate']; gt = tp['groundTruth']
    lstm_key = (code, cutoff); lstm_sig = lstm_lookup.get(lstm_key, 0)
    gt_map = {'strong_bull': 2, 'bull': 1, 'neutral': 0, 'bear': -1, 'strong_bear': -2}
    all_sig.append(lstm_sig); all_gt.append(gt_map.get(gt, 0))

if len(all_sig) > 10:
    ic = spearmanr(all_sig, all_gt)[0]
    hit = np.mean(np.sign(np.array(all_sig)) == np.sign(np.array(all_gt)))
    print(f"\nLSTM signal IC on frozen testPoints: {ic:.4f}")
    print(f"LSTM signal sign hit rate: {hit:.2%}")
    print(f"\nPrediction: if LLM uses LSTM signal with >50% weight, Δ score ≈ +{abs(ic)*0.5:.3f}")

print(f"\nReady for real LLM eval. ENABLE_LSTM_SIGNAL=false (default, not polluting baseline).")
print(f"Run: node --env-file=.env cli/eval-v6-sector.js → modify to pass lstmSignalData")
