// Anthropic Claude API adapter (includes Tool Use + SSE streaming + extended thinking)
// Only Anthropic provider enables tool_use / thinking; DeepSeek skips

export const KNOWN_MODELS = [
  'claude-opus-4-7',
  'claude-sonnet-4-6',
  'claude-haiku-4-5',
  'claude-haiku-4-5-20251001',
];

const ANTHROPIC_ENDPOINT = 'https://api.anthropic.com/v1/messages';
const MAX_TOOL_ROUNDS = 5;

// Convert tool definitions to Anthropic API format (strip handler)
function toolsToAPI(tools) {
  return tools.map((t) => ({
    name: t.name,
    description: t.description,
    input_schema: t.input_schema,
  }));
}

/** @type {import('./provider.js').LLMProvider} */
export const anthropicProvider = {
  id: 'anthropic',
  displayName: 'Anthropic Claude',
  defaultModel: 'claude-sonnet-4-6',
  apiKeyPattern: 'sk-ant-...',
  apiKeyHelp: 'Get from console.anthropic.com/settings/keys (requires international payment)',
  docUrl: 'https://docs.claude.com/en/docs/about-claude/models/overview',
  currency: 'USD',

  async call(promptOrMessages, opts) {
    if (typeof promptOrMessages !== 'string' && !Array.isArray(promptOrMessages)) {
      throw new TypeError('anthropic.call() expects string or array of messages');
    }

    const messages = Array.isArray(promptOrMessages)
      ? promptOrMessages
      : [{ role: 'user', content: promptOrMessages }];

    const tools = Array.isArray(opts.tools) && opts.tools.length > 0 ? opts.tools : null;
    const onProgress = typeof opts.onProgress === 'function' ? opts.onProgress : null;

    // Use streaming path when onProgress is provided
    if (onProgress) {
      return streamCall(messages, tools, opts, onProgress);
    }

    // ---- Non-streaming path (backward compatible) ----
    return nonStreamCall(messages, tools, opts);
  },
};

// ---- Non-streaming call (tool_use while loop) ----

async function nonStreamCall(messages, tools, opts) {
  const totalUsage = { inputTokens: 0, outputTokens: 0 };
  let currentMessages = messages;
  let round = 0;

  while (true) {
    round++;
    const body = {
      model: opts.model,
      max_tokens: opts.maxTokens,
      messages: currentMessages,
    };
    if (tools) {
      body.tools = toolsToAPI(tools);
    }

    const json = await fetchJSON(ANTHROPIC_ENDPOINT, body, opts.apiKey);

    totalUsage.inputTokens += json.usage?.input_tokens || 0;
    totalUsage.outputTokens += json.usage?.output_tokens || 0;

    const stopReason = json.stop_reason;
    if (stopReason === 'tool_use' && tools) {
      if (round > MAX_TOOL_ROUNDS) {
        throw new Error(`Tool call rounds exceeded max (${MAX_TOOL_ROUNDS}), aborted to prevent infinite loop`);
      }

      const content = json.content || [];
      const toolUseBlocks = content.filter((b) => b.type === 'tool_use');

      if (toolUseBlocks.length === 0) {
        return { text: extractText(content), usage: totalUsage };
      }

      currentMessages.push({ role: 'assistant', content });

      const toolResults = [];
      for (const block of toolUseBlocks) {
        console.log(`[tool_use] round=${round} tool=${block.name} input=${JSON.stringify(block.input)}`);

        let toolResultText;
        try {
          const toolDef = tools.find((t) => t.name === block.name);
          toolResultText = toolDef
            ? await toolDef.handler(block.input)
            : `Error: unknown tool "${block.name}"`;
        } catch (err) {
          toolResultText = `Tool call exception: ${err.message || String(err)}`;
        }

        toolResults.push({
          type: 'tool_result',
          tool_use_id: block.id,
          content: toolResultText,
        });
      }

      currentMessages.push({ role: 'user', content: toolResults });
      continue;
    }

    const text = extractText(json.content);
    if (typeof text !== 'string' || text.length === 0) {
      throw new Error('Claude response format abnormal');
    }

    return { text, usage: totalUsage };
  }
}

// ---- Streaming call (SSE parsing) ----

async function streamCall(messages, tools, opts, onProgress) {
  const totalUsage = { inputTokens: 0, outputTokens: 0 };
  let currentMessages = messages;
  let round = 0;

  while (true) {
    round++;
    const body = {
      model: opts.model,
      max_tokens: opts.maxTokens,
      messages: currentMessages,
      stream: true,
    };
    if (tools) {
      body.tools = toolsToAPI(tools);
    }
    // extended thinking: Opus only + explicitly enabled
    if (opts.enableThinking && opts.model === 'claude-opus-4-7') {
      body.thinking = { type: 'enabled', budget_tokens: 8000 };
    }

    const result = await fetchStream(ANTHROPIC_ENDPOINT, body, opts.apiKey, tools, onProgress);

    totalUsage.inputTokens += result.usage.inputTokens || 0;
    totalUsage.outputTokens += result.usage.outputTokens || 0;

    if (result.stopReason === 'tool_use' && tools) {
      if (round > MAX_TOOL_ROUNDS) {
        throw new Error(`Tool call rounds exceeded max (${MAX_TOOL_ROUNDS}), aborted to prevent infinite loop`);
      }

      // Append assistant + tool_result messages
      currentMessages.push({ role: 'assistant', content: result.content });

      const toolResults = [];
      for (const tu of result.toolUses) {
        console.log(`[tool_use] round=${round} tool=${tu.name} input=${JSON.stringify(tu.input)}`);

        let toolResultText;
        try {
          const toolDef = tools.find((t) => t.name === tu.name);
          toolResultText = toolDef
            ? await toolDef.handler(tu.input)
            : `Error: unknown tool "${tu.name}"`;
        } catch (err) {
          toolResultText = `Tool call exception: ${err.message || String(err)}`;
        }

        toolResults.push({
          type: 'tool_result',
          tool_use_id: tu.id,
          content: toolResultText,
        });

        onProgress({ type: 'tool_result', name: tu.name, result: toolResultText });
      }

      currentMessages.push({ role: 'user', content: toolResults });
      continue;
    }

    // end_turn or max_tokens
    if (result.text.length === 0) {
      throw new Error('Claude response format abnormal');
    }

    return { text: result.text, usage: totalUsage };
  }
}

// ---- SSE stream parsing ----

async function fetchStream(endpoint, body, apiKey, tools, onProgress) {
  const emit = typeof onProgress === 'function' ? onProgress : () => {};
  let resp;
  try {
    resp = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json',
        'anthropic-dangerous-direct-browser-access': 'true',
      },
      body: JSON.stringify(body),
    });
  } catch (_) {
    throw new Error('Network error, check your connection');
  }

  if (resp.status === 401) throw new Error('Invalid API key, check the value in the popup');
  if (resp.status === 429) throw new Error('Anthropic Claude rate limit triggered, retry later');
  if (resp.status >= 500) throw new Error('Anthropic Claude server overloaded, retry later');

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
    throw new Error(`Anthropic Claude API error: HTTP ${resp.status} — ${detail}`);
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  // Accumulated state
  const textParts = [];
  const toolUses = []; // { id, name, input }
  let currentToolUse = null;
  let currentTextBlock = null; // text buffer for current text block
  let inputJsonBuffer = '';
  let stopReason = null;
  const usage = { inputTokens: 0, outputTokens: 0 };
  const content = []; // reassemble messages (text + tool_use in order)

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split('\n\n');
    buffer = events.pop(); // last one may be incomplete

    for (const event of events) {
      if (!event.trim()) continue;
      const parsed = parseSSEEvent(event);
      if (!parsed) continue;

      const { type, data } = parsed;

      switch (type) {
        case 'message_start':
          if (data.message?.usage) {
            usage.inputTokens += data.message.usage.input_tokens || 0;
          }
          break;

        case 'content_block_start': {
          const block = data.content_block;
          if (!block) break;
          if (block.type === 'thinking') {
            // extended thinking start
          } else if (block.type === 'tool_use') {
            currentToolUse = { id: block.id, name: block.name, input: null };
            inputJsonBuffer = '';
          } else if (block.type === 'text') {
            currentTextBlock = '';
          }
          break;
        }

        case 'content_block_delta': {
          const delta = data.delta;
          if (!delta) break;
          if (delta.type === 'thinking_delta') {
            emit({ type: 'thinking', text: delta.thinking || '' });
          } else if (delta.type === 'text_delta') {
            const t = delta.text || '';
            textParts.push(t);
            if (currentTextBlock !== null) currentTextBlock += t;
            emit({ type: 'text', text: t });
          } else if (delta.type === 'input_json_delta') {
            inputJsonBuffer += delta.partial_json || '';
          }
          break;
        }

        case 'content_block_stop':
          if (currentToolUse) {
            try {
              currentToolUse.input = JSON.parse(inputJsonBuffer);
            } catch (_) {
              currentToolUse.input = {};
            }
            toolUses.push(currentToolUse);
            content.push({
              type: 'tool_use',
              id: currentToolUse.id,
              name: currentToolUse.name,
              input: currentToolUse.input,
            });
            emit({
              type: 'tool_start',
              name: currentToolUse.name,
              input: JSON.stringify(currentToolUse.input),
            });
            currentToolUse = null;
            inputJsonBuffer = '';
          } else if (currentTextBlock !== null) {
            content.push({ type: 'text', text: currentTextBlock });
            currentTextBlock = null;
          }
          break;

        case 'message_delta':
          if (data.usage) {
            usage.outputTokens += data.usage.output_tokens || 0;
          }
          if (data.delta?.stop_reason) {
            stopReason = data.delta.stop_reason;
          }
          break;

        case 'message_stop':
          // stream end
          break;

        // ignore ping events
        default:
          break;
      }
    }
  }

  const text = textParts.join('');

  return {
    text,
    content,
    toolUses,
    stopReason: stopReason || 'end_turn',
    usage,
  };
}

// Parse a single SSE event
function parseSSEEvent(raw) {
  const lines = raw.split('\n');
  let eventType = '';
  let dataStr = '';

  for (const line of lines) {
    if (line.startsWith('event: ')) {
      eventType = line.slice(7).trim();
    } else if (line.startsWith('data: ')) {
      // SSE allows multiple data: lines, concatenate to restore full JSON
      dataStr += line.slice(6);
    }
  }

  if (!dataStr) return null;

  let data;
  try {
    data = JSON.parse(dataStr);
  } catch (_) {
    return null;
  }

  // Prefer data.type, fallback to event line
  const type = (data && data.type) || eventType || '';
  return { type, data };
}

// ---- Non-streaming JSON request ----

async function fetchJSON(endpoint, body, apiKey) {
  let resp;
  try {
    resp = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json',
        'anthropic-dangerous-direct-browser-access': 'true',
      },
      body: JSON.stringify(body),
    });
  } catch (_) {
    throw new Error('Network error, check your connection');
  }

  if (resp.status === 401) throw new Error('Invalid API key, check the value in the popup');
  if (resp.status === 429) throw new Error('Anthropic Claude rate limit triggered, retry later');
  if (resp.status >= 500) throw new Error('Anthropic Claude server overloaded, retry later');

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
    throw new Error(`Anthropic Claude API error: HTTP ${resp.status} — ${detail}`);
  }

  return resp.json();
}

// ---- Utility functions ----

function extractText(content) {
  if (!Array.isArray(content)) return '';
  const textBlocks = content.filter((b) => b.type === 'text');
  return textBlocks.map((b) => b.text || '').join('');
}
