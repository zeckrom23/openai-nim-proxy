// worker.js - OpenAI to NVIDIA NIM API Proxy (Cloudflare Workers)

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = false;

// 🔥 THINKING MODE TOGGLE
const ENABLE_THINKING_MODE = false;

// Model mapping
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o': 'deepseek-ai/deepseek-v3.1',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking'
};

function corsHeaders() {
  return {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
  };
}

function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json', ...corsHeaders() }
  });
}

async function resolveModel(model, nimApiBase, nimApiKey) {
  // Check mapping first
  if (MODEL_MAPPING[model]) return MODEL_MAPPING[model];

  // Try the model directly
  try {
    const test = await fetch(`${nimApiBase}/chat/completions`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${nimApiKey}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, messages: [{ role: 'user', content: 'test' }], max_tokens: 1 })
    });
    if (test.ok) return model;
  } catch (_) {}

  // Fallback by name heuristics
  const lower = model.toLowerCase();
  if (lower.includes('gpt-4') || lower.includes('claude-opus') || lower.includes('405b'))
    return 'meta/llama-3.1-405b-instruct';
  if (lower.includes('claude') || lower.includes('gemini') || lower.includes('70b'))
    return 'meta/llama-3.1-70b-instruct';
  return 'meta/llama-3.1-8b-instruct';
}

async function handleChatCompletions(request, env) {
  const NIM_API_BASE = env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
  const NIM_API_KEY = env.NIM_API_KEY;

  const body = await request.json();
  const { model, messages, temperature, max_tokens, stream } = body;

  const nimModel = await resolveModel(model, NIM_API_BASE, NIM_API_KEY);

  const nimRequest = {
    model: nimModel,
    messages,
    temperature: temperature || 0.6,
    max_tokens: max_tokens || 9024,
    stream: stream || false,
    ...(ENABLE_THINKING_MODE && { extra_body: { chat_template_kwargs: { thinking: true } } })
  };

  const nimResponse = await fetch(`${NIM_API_BASE}/chat/completions`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${NIM_API_KEY}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(nimRequest)
  });

  if (!nimResponse.ok && !stream) {
    const err = await nimResponse.text();
    return jsonResponse({ error: { message: err, type: 'invalid_request_error', code: nimResponse.status } }, nimResponse.status);
  }

  // --- STREAMING ---
  if (stream) {
    const { readable, writable } = new TransformStream();
    const writer = writable.getWriter();
    const encoder = new TextEncoder();
    const decoder = new TextDecoder();

    (async () => {
      const reader = nimResponse.body.getReader();
      let buffer = '';
      let reasoningStarted = false;

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;

            if (line.includes('[DONE]')) {
              await writer.write(encoder.encode(line + '\n\n'));
              continue;
            }

            try {
              const data = JSON.parse(line.slice(6));

              if (data.choices?.[0]?.delta) {
                const reasoning = data.choices[0].delta.reasoning_content;
                const content = data.choices[0].delta.content;

                if (SHOW_REASONING) {
                  let combined = '';
                  if (reasoning && !reasoningStarted) { combined = '<think>\n' + reasoning; reasoningStarted = true; }
                  else if (reasoning) { combined = reasoning; }
                  if (content && reasoningStarted) { combined += '</think>\n\n' + content; reasoningStarted = false; }
                  else if (content) { combined += content; }
                  if (combined) data.choices[0].delta.content = combined;
                } else {
                  data.choices[0].delta.content = content || '';
                }
                delete data.choices[0].delta.reasoning_content;
              }

              await writer.write(encoder.encode(`data: ${JSON.stringify(data)}\n\n`));
            } catch (_) {
              await writer.write(encoder.encode(line + '\n'));
            }
          }
        }
      } finally {
        await writer.close();
      }
    })();

    return new Response(readable, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        ...corsHeaders()
      }
    });
  }

  // --- NON-STREAMING ---
  const data = await nimResponse.json();

  const openaiResponse = {
    id: `chatcmpl-${Date.now()}`,
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model,
    choices: data.choices.map(choice => {
      let content = choice.message?.content || '';
      if (SHOW_REASONING && choice.message?.reasoning_content) {
        content = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + content;
      }
      return {
        index: choice.index,
        message: { role: choice.message.role, content },
        finish_reason: choice.finish_reason
      };
    }),
    usage: data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
  };

  return jsonResponse(openaiResponse);
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // Handle CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders() });
    }

    // Routes
    if (url.pathname === '/health' && request.method === 'GET') {
      return jsonResponse({
        status: 'ok',
        service: 'OpenAI to NVIDIA NIM Proxy',
        reasoning_display: SHOW_REASONING,
        thinking_mode: ENABLE_THINKING_MODE
      });
    }

    if (url.pathname === '/v1/models' && request.method === 'GET') {
      return jsonResponse({
        object: 'list',
        data: Object.keys(MODEL_MAPPING).map(id => ({
          id, object: 'model', created: Date.now(), owned_by: 'nvidia-nim-proxy'
        }))
      });
    }

    if (url.pathname === '/v1/chat/completions' && request.method === 'POST') {
      try {
        return await handleChatCompletions(request, env);
      } catch (err) {
        return jsonResponse({ error: { message: err.message || 'Internal server error', type: 'invalid_request_error', code: 500 } }, 500);
      }
    }

    // 404 catch-all
    return jsonResponse({ error: { message: `Endpoint ${url.pathname} not found`, type: 'invalid_request_error', code: 404 } }, 404);
  }
};
