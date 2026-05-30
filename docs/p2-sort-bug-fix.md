# P2 sort 参考扩散 bug 排查与修复

> 分支: `p2-fix-sort-ref-bug` | 日期: 2026-05-29

## 结论

**无 sort 信号覆盖 bug。B4 +29.5% 的真正根因：内联 Z-score 方差计算退化。**

`Math.pow(b - ref?.mean || 0, 2)` 中 `ref?.mean` 对 `{mean:{}, std:{}}` 返回 `undefined`，`(b-0)^2` = RMS，非方差。所有 Z-score 压缩至零附近，巧合显示相同 spread。该脚本未存文件。

---

## 全仓审计（70+ 处 .sort()）

| 文件 | 行 | 模式 | 判定 |
|------|-----|------|------|
| `rulers.js` | 26/34/133 | `[...arr].sort()` | ✓ |
| `eval-momentum-validate.js` | 52/114 | `[...pairs].sort()` | ✓ |
| `eval-strongbull-vs-momentum.js` | 22/48/150 | `[...arr].sort()` | ✓ |
| `sector/alpha.js` | 107 | `items.sort()` | ✓ 新数组 |
| `sector/alpha.js` | 115 | `results.sort()` | ✓ 函数内创建 |
| 其余 ~65 处 | — | filter/spread 后 sort | ✓ |

## 修正值

| 指标 | B4 错误 | 修正 |
|------|---------|------|
| 动量 spread | +29.5% | -13.1% |
| 动量 CI | 全正 | 含 0 |
