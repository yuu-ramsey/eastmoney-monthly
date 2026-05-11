// 从 LLM 输出文本中提取结构化 JSON 块，供跨级别一致性校验使用

/**
 * @param {string} analysisText - LLM 分析输出全文
 * @returns {{ data: object | null, rawJsonText: string | null, error: string | null }}
 */
export function extractStructuredOutput(analysisText) {
  const match = analysisText.match(/```json\s*([\s\S]*?)\s*```/);
  if (!match) return { data: null, rawJsonText: null, error: 'JSON 块未找到' };

  let data;
  try {
    data = JSON.parse(match[1]);
  } catch (e) {
    return { data: null, rawJsonText: match[1], error: 'JSON 解析失败: ' + e.message };
  }

  // key 归一化：兼容 LLM 输出的字段名大小写差异
  data = normalizeKeys(data);

  // 字段校验
  if (!data.period || !data.centralZone) {
    return { data: null, rawJsonText: match[1], error: 'JSON 结构不完整: 缺少 period 或 centralZone' };
  }

  return { data, rawJsonText: match[1], error: null };
}

function normalizeKeys(obj) {
  if (!obj || typeof obj !== 'object' || Array.isArray(obj)) return obj;
  // 全小写 → camelCase 映射
  const camelMap = { centralzone: 'centralZone', keysupport: 'keySupport', keyresistance: 'keyResistance' };
  const out = {};
  for (const [k, v] of Object.entries(obj)) {
    const key = k.toLowerCase();
    const mappedKey = camelMap[key] || key;
    const val = (mappedKey === 'centralZone' && v && typeof v === 'object' && !Array.isArray(v))
      ? normalizeKeys(v)
      : v;
    out[mappedKey] = val;
  }
  return out;
}
