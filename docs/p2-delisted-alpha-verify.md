# P2 退市股 alpha 验证

> 分支: `p2-rebuild-unbiased-pool` | 日期: 2026-05-30

## 结论: Alpha 算法干净。11 退市对仅占 0.7%，全部 >=6m horizon，拖低而非抬高反转 spread。

---

## 1. Horizon 规则

```python
fwd = [data[j][1] for j in range(ci+1, min(len(data), ci+HORIZON+1)) if data[j][1]>0.001]
alpha = (fwd[HORIZON-1] - close) / close * 100 if len(fwd) >= HORIZON else None
```

不足 6 月 → None，不进池。全部入池对严格 6m horizon。

---

## 2. 抽样核对

| Code | Cutoff | Fwd月 | Alpha | revZ | Bucket |
|------|--------|-------|-------|------|--------|
| 000018 | 2019-06 | 6 | -80.3% | -2.21 | bull |
| 600240 | 2019-06 | 7 | -64.7% | -3.09 | bull |
| 000018 | 2018-12 | 12 | -35.4% | -1.23 | bull |
| 600242 | 2022-06 | 11 | -32.4% | -1.88 | bull |
| 600209 | 2018-06 | 47 | -26.6% | -1.39 | bull |
| 600077 | 2018-06 | 60 | -25.2% | +0.07 | neutral |
| 600074 | 2018-06 | 23 | -21.1% | -0.22 | neutral |
| 600069 | 2019-06 | 13 | +9.9% | -1.26 | bull |

全部 >=6m。alpha 大负值合理（退市前价值毁灭）。

---

## 3. 反转桶分布

| 桶 | 退市股 |
|----|-------|
| Bullish | 9 |
| Bearish | **0** |

退市股全在 bullish（跌最深）桶，负 alpha 拖低 bullish 均值。

---

## 4. 贡献分解

```
含退市: spread = +6.63%
去退市: spread = +7.47%
Δ: -0.84pp
```

**退市股在压低而非抬高反转 spread。** +6.6% 是保守估计。扩容根因是 Baostock 覆盖了 990 个新 code（非退市），非退市灌水。
