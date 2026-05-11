// Anthropic Claude API adapter

export const KNOWN_MODELS = [
  'claude-opus-4-7',
  'claude-sonnet-4-6',
  'claude-haiku-4-5',
  'claude-haiku-4-5-20251001',
];

const ANTHROPIC_ENDPOINT = 'https://api.anthropic.com/v1/messages';

/** @type {import('./provider.js').LLMProvider} */
export const anthropicProvider = {
  id: 'anthropic',
  displayName: 'Anthropic Claude',
  defaultModel: 'claude-sonnet-4-6',
  apiKeyPattern: 'sk-ant-...',
  apiKeyHelp: '从 console.anthropic.com/settings/keys 获取（需海外支付）',
  docUrl: 'https://docs.claude.com/en/docs/about-claude/models/overview',
  currency: 'USD',

  async call(promptOrMessages, opts) {
    if (typeof promptOrMessages !== 'string' && !Array.isArray(promptOrMessages)) {
      throw new TypeError('anthropic.call() 期望 string 或 messages 数组');
    }
    const messages = Array.isArray(promptOrMessages)
      ? promptOrMessages
      : [{ role: 'user', content: promptOrMessages }];

    let resp;
    try {
      resp = await fetch(ANTHROPIC_ENDPOINT, {
        method: 'POST',
        headers: {
          'x-api-key': opts.apiKey,
          'anthropic-version': '2023-06-01',
          'content-type': 'application/json',
          'anthropic-dangerous-direct-browser-access': 'true',
        },
        body: JSON.stringify({
          model: opts.model,
          max_tokens: opts.maxTokens,
          messages,
        }),
      });
    } catch (_) {
      throw new Error('网络错误，请检查连接');
    }

    if (resp.status === 401) throw new Error('API key 无效，请检查弹窗里填的值');
    if (resp.status === 429) throw new Error('触发 Anthropic Claude 限流，稍后重试');
    if (resp.status >= 500) throw new Error('Anthropic Claude 服务过载，稍后重试');

    if (!resp.ok) {
      let detail = '';
      try {
        const text = await resp.text();
        try {
          const errJson = JSON.parse(text);
          detail = errJson?.error?.message || text;
        } catch (_) { detail = text; }
      } catch (_) { /* ignore */ }
      if (detail.length > 400) detail = detail.slice(0, 400) + '...';
      throw new Error(`Anthropic Claude API 错误：HTTP ${resp.status} — ${detail}`);
    }

    const json = await resp.json();
    const text = json?.content?.[0]?.text;
    if (typeof text !== 'string' || text.length === 0) {
      throw new Error('Claude 响应格式异常');
    }

    return {
      text,
      usage: {
        inputTokens: json.usage?.input_tokens || 0,
        outputTokens: json.usage?.output_tokens || 0,
      },
    };
  },
};
