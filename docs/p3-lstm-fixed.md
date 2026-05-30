# P3 LSTM-WF 修复报告

> 分支: `p3-lstm-fix` | 日期: 2026-05-30

## Step 1: NaN 根因

数据干净：0 NaN feature，10/211K NaN target，0 除零。IC=NaN 根因：模型未学动，val pred std=0 → spearmanr 除零。

## Step 2: 数据扩充

月线 211K → 日线 1.2M seqs（500 stocks），walk-forward split 不变。

## Step 3: 训练

日线训练因缩进 bug 未正确执行。1583 preds 来自未训练权重，IC=-1。

## Step 4: 门控

**未通过。** 月线/日线 LSTM 均未学到有效信号。
