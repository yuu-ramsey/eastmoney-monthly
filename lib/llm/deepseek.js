// DeepSeek API adapter (OpenAI-compatible interface)

export const KNOWN_MODELS = [
  'deepseek-chat',
  'deepseek-reasoner',
];

const DEEPSEEK_ENDPOINT = 'https://api.deepseek.com/chat/completions';

/** @type {import('./provider.js').LLMProvider} */
export const deepseekProvider = {
  id: 'deepseek',
  displayName: 'DeepSeek',
  defaultModel: 'deepseek-chat',
  apiKeyPattern: 'sk-...',
  apiKeyHelp: 'Get from platform.deepseek.com/api_keys (Alipay/WeChat supported)',
  docUrl: 'https://api-docs.deepseek.com/',
  currency: 'CNY',

  async call(promptOrMessages, opts) {
    if (typeof promptOrMessages !== 'string' && !Array.isArray(promptOrMessages)) {
      throw new TypeError('deepseek.call() 期望 string 或 messages 数组');
    }
    const messages = Array.isArray(promptOrMessages)
      ? promptOrMessages
      : [{ role: 'user', content: promptOrMessages }];

    let resp;
    try {
      resp = await fetch(DEEPSEEK_ENDPOINT, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${opts.apiKey}`,
          'Content-Type': 'application/json',
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
    if (resp.status === 429) throw new Error('触发 DeepSeek 限流，稍后重试');
    if (resp.status >= 500) throw new Error('DeepSeek 服务过载，稍后重试');

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
      throw new Error(`DeepSeek API 错误：HTTP ${resp.status} — ${detail}`);
    }

    const json = await resp.json();
    const text = json?.choices?.[0]?.message?.content;
    if (typeof text !== 'string' || text.length === 0) {
      throw new Error('DeepSeek 响应格式异常');
    }

    return {
      text,
      usage: {
        inputTokens: json.usage?.prompt_tokens || 0,
        outputTokens: json.usage?.completion_tokens || 0,
      },
    };
  },
};
