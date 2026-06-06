// Structured output JSON extraction test
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { extractStructuredOutput } from '../lib/parse-structured-output.js';

test('extractStructuredOutput: normal JSON block extraction passes', () => {
  const text = `# Analysis report
...skipped...
\`\`\`json
{
  "period": "monthly",
  "centralZone": {
    "lower": 1450.00,
    "upper": 1650.00,
    "exists": true
  },
  "keySupport": [1500.00, 1400.00, 1350.00],
  "keyResistance": [1700.00, 1750.00, 1800.00],
  "trend": "up"
}
\`\`\``;

  const result = extractStructuredOutput(text);
  assert.equal(result.error, null);
  assert.ok(result.data);
  assert.equal(result.data.period, 'monthly');
  assert.equal(result.data.centralZone.lower, 1450.00);
  assert.equal(result.data.centralZone.upper, 1650.00);
  assert.equal(result.data.centralZone.exists, true);
  assert.deepEqual(result.data.keySupport, [1500.00, 1400.00, 1350.00]);
  assert.deepEqual(result.data.keyResistance, [1700.00, 1750.00, 1800.00]);
  assert.equal(result.data.trend, 'up');
});

test('extractStructuredOutput: no JSON block returns error', () => {
  const text = '# Analysis report\n\nNo JSON block in analysis\n\nPlain text output';
  const result = extractStructuredOutput(text);
  assert.equal(result.data, null);
  assert.equal(result.error, 'JSON 块未找到');
});

test('extractStructuredOutput: JSON parse failure returns error', () => {
  const text = `\`\`\`json
{ broken json, missing quotes }
\`\`\``;
  const result = extractStructuredOutput(text);
  assert.equal(result.data, null);
  assert.match(String(result.error), /JSON 解析失败/);
  assert.ok(result.rawJsonText);
});

test('extractStructuredOutput: incomplete JSON structure returns error (missing period)', () => {
  const text = `\`\`\`json
{
  "centralZone": { "lower": 10, "upper": 20, "exists": true },
  "keySupport": [8],
  "keyResistance": [22],
  "trend": "sideways"
}
\`\`\``;
  const result = extractStructuredOutput(text);
  assert.equal(result.data, null);
  assert.match(String(result.error), /JSON 结构不完整/);
  assert.match(String(result.error), /period/);
});

test('extractStructuredOutput: missing centralZone also reports incomplete', () => {
  const text = `\`\`\`json
{
  "period": "daily",
  "keySupport": [50],
  "keyResistance": [60],
  "trend": "down"
}
\`\`\``;
  const result = extractStructuredOutput(text);
  assert.equal(result.data, null);
  assert.match(String(result.error), /JSON 结构不完整/);
});

test('extractStructuredOutput: centralZone.exists=false with lower/upper=null is valid', () => {
  const text = `\`\`\`json
{
  "period": "weekly",
  "centralZone": { "lower": null, "upper": null, "exists": false },
  "keySupport": [30.00, 25.00],
  "keyResistance": [40.00],
  "trend": "sideways"
}
\`\`\``;
  const result = extractStructuredOutput(text);
  assert.equal(result.error, null);
  assert.ok(result.data);
  assert.equal(result.data.centralZone.exists, false);
  assert.equal(result.data.centralZone.lower, null);
  assert.equal(result.data.centralZone.upper, null);
});

test('extractStructuredOutput: only first json block matched when multiple blocks present', () => {
  const text = `\`\`\`json
{ "period": "monthly", "centralZone": { "lower": 10, "upper": 20, "exists": true }, "keySupport": [], "keyResistance": [], "trend": "up" }
\`\`\`
Followed by another code block
\`\`\`json
{ "period": "daily" }
\`\`\``;
  const result = extractStructuredOutput(text);
  assert.equal(result.error, null);
  assert.equal(result.data.period, 'monthly');
});

test('extractStructuredOutput: field name case normalization handled', () => {
  const text = `\`\`\`json
{
  "Period": "monthly",
  "CentralZone": {
    "Lower": 50.00,
    "Upper": 60.00,
    "Exists": true
  },
  "KeySupport": [45.00],
  "KeyResistance": [65.00],
  "Trend": "up"
}
\`\`\``;
  const result = extractStructuredOutput(text);
  assert.equal(result.error, null);
  assert.ok(result.data);
  assert.equal(result.data.period, 'monthly');
  assert.equal(result.data.centralZone.lower, 50.00);
  assert.equal(result.data.centralZone.upper, 60.00);
  assert.equal(result.data.centralZone.exists, true);
  assert.deepEqual(result.data.keySupport, [45.00]);
  assert.equal(result.data.trend, 'up');
});

test('extractStructuredOutput: extra whitespace around JSON handled', () => {
  const text = `

\`\`\`json


{
  "period": "monthly",
  "centralZone": { "lower": 100.00, "upper": 200.00, "exists": true },
  "keySupport": [90.00],
  "keyResistance": [210.00],
  "trend": "sideways"
}


\`\`\`

`;
  const result = extractStructuredOutput(text);
  assert.equal(result.error, null);
  assert.ok(result.data);
  assert.equal(result.data.period, 'monthly');
});
