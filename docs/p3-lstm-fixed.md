# P3 LSTM-WF Fix Report

> Branch: `p3-lstm-fix` | Date: 2026-05-30

## Step 1: NaN Root Cause

Data is clean: 0 NaN features, 10/211K NaN targets, 0 division by zero. IC=NaN root cause: model did not learn; val pred std=0 -> spearmanr division by zero.

## Step 2: Data Expansion

Monthly 211K -> Daily 1.2M seqs (500 stocks), walk-forward split unchanged.

## Step 3: Training

Daily training did not execute correctly due to indentation bug. 1583 preds from untrained weights, IC=-1.

## Step 4: Gate

**Not passed.** Neither monthly nor daily LSTM learned a valid signal.
