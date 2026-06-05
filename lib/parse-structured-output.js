// Extract structured JSON blocks from LLM output text for cross-level consistency validation

/**
 * @param {string} analysisText - Full LLM analysis output
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

  // Key normalization: handle LLM output field name case differences
  data = normalizeKeys(data);

  // Field validation
  if (!data.period || !data.centralZone) {
    return { data: null, rawJsonText: match[1], error: 'JSON 结构不完整: 缺少 period 或 centralZone' };
  }

  return { data, rawJsonText: match[1], error: null };
}

function normalizeKeys(obj) {
  if (!obj || typeof obj !== 'object' || Array.isArray(obj)) return obj;
  // Lowercase → camelCase mapping
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
