# P2 Z-score 方差验证

> 分支: `p2-verify-and-data-probe` | 日期: 2026-05-29

## 结论: 已提交代码全部正确。无需修改。

---

## B4 bug 根因

内联脚本（未存盘）: `Math.pow(b - ref?.mean || 0, 2)`
- `ref?.mean` 对 `{mean:{}, std:{}}` → `undefined`
- `undefined || 0` → 0 → `(b-0)^2` → RMS 非方差
- Z-score 压缩 → 信号退化

## 全仓审计

`Math.pow(.*-.*\?\\.)` → 0 命中。已无存留。

## 逐一验证

| 文件 | 行 | 判定 |
|------|-----|------|
| `rulers.js` | 127 | ✓ 调用方预计算 ref |
| `eval-momentum-validate.js` | 43 | ✓ `Math.pow(b - ref.mean[k], 2)` |
| `eval-strongbull-vs-momentum.js` | 124-126 | ✓ mean 内联重算 |
| `eval-power-analysis.js` | 21 | ✓ `Math.pow(a - alphaMean, 2)` |

反转 -32.5%、动量 -13.1% 均基于正确方差，可信。
