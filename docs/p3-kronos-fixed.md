# P3 Kronos 修复 — 最终结果

> 分支: `p3-kronos-fix` | 日期: 2026-05-30 | GPU: RTX 5070

## 根因

`KronosPredictor` 构造参数顺序反了。`__init__(tokenizer, model)` 但调用时传 `(model, tok)`，`self.tokenizer` 指向 model，`.encode()` 报 `'Kronos' object has no attribute 'encode'`。

## 修复

```python
# 错误: pred = KronosPredictor(model, tok, device=DEVICE)
# 正确:
pred = KronosPredictor(tok, model, device=DEVICE)
```

一字未改 kronos 源码。

## 结果

| 信号 | n | Spread | Train | Test |
|------|---|--------|-------|------|
| LLM | 1608 | +8.94% CI[3.7,14.1] | +10.1% | +6.4% |
| **Kronos** | 1574 | **+7.59%** | +11.1% | +4.7% |
| 反转 | 1608 | +6.63% CI[1.3,11.2] | +0.9% | +13.0% |
| LSTM-old | 1583 | +0.91% | -2.7% | +1.5% |

## 判读

**通过门控。** Kronos +7.59% 与 LLM/反转同级。Train>>Test (11% vs 5%) 提示 regime 依赖。
