# P2 LLM 预注册 v2 — Phase C0

> 分支: `p2-rebuild-unbiased-pool` | 日期: 2026-05-30 | **跑前 commit，跑后不得改**

## 池子

v2 Baostock 无偏: 1608 pairs, MDE 8.1%, 反转 +6.6% CI[1.4,11.3], 动量归零

## 主指标

**(LLM Wins spread − 反转 Wins spread) 分块差值 CI**

## 判读

| 差值 CI | 结论 |
|---------|------|
| 下界 > 0 | LLM > 反转, 增量 edge |
| 含 0 | inconclusive (MDE 8.1%, 标功效边界) |
| 上界 < 0 | LLM < 反转 |

## 参照系

反转 +6.6% CI[1.4,11.3] | 动量 ~0% | LLM 待测

## 预期

上轮 LLM 67% bull/1% bear/spread 负, 预期仍偏多、可能跑不赢反转
