// Cross-level consistency check: compare current period vs parent period structured output, detect nesting anomalies and trend conflicts

/**
 * @param {object} current - 当前分析的结构化数据
 * @param {object|null} parent - 上级周期的结构化数据（如 daily→weekly, weekly→monthly），null 则跳过
 * @returns {{ warnings: string[] }}
 */
export function checkCrossLevelConsistency(current, parent) {
  const warnings = [];
  if (!parent) return { warnings };

  // Rule 1: lower-level central zone should be within higher-level central zone range, with 15% tolerance
  const curZone = current.centralZone;
  const parZone = parent.centralZone;
  if (curZone?.exists && parZone?.exists) {
    const czLower = Number(curZone.lower);
    const czUpper = Number(curZone.upper);
    const pzLower = Number(parZone.lower);
    const pzUpper = Number(parZone.upper);
    if (!Number.isNaN(czLower) && !Number.isNaN(czUpper) && !Number.isNaN(pzLower) && !Number.isNaN(pzUpper)) {
      const tolerance = (pzUpper - pzLower) * 0.15;
      if (czLower < pzLower - tolerance || czUpper > pzUpper + tolerance) {
        warnings.push(
          `中枢嵌套异常：当前周期中枢 [${czLower.toFixed(2)}, ${czUpper.toFixed(2)}] 超出上级周期中枢 [${pzLower.toFixed(2)}, ${pzUpper.toFixed(2)}] 范围（已超出 15% 容差）`
        );
      }
    }
  }

  // Rule 2: trend direction contradiction
  if (current.trend && parent.trend) {
    if (current.trend === 'reversal_top' && parent.trend === 'up') {
      warnings.push('趋势冲突：上级周期偏多但当前周期出现顶部反转信号');
    }
    if (current.trend === 'reversal_bottom' && parent.trend === 'down') {
      warnings.push('趋势冲突：上级周期偏空但当前周期出现底部反转信号');
    }
  }

  return { warnings };
}
