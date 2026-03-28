// worker.js - OpenAI to NVIDIA NIM API Proxy (Cloudflare Workers)

// 🔥 REASONING DISPLAY TOGGLE
const SHOW_REASONING = false;

// 🔥 THINKING MODE TOGGLE
const ENABLE_THINKING_MODE = false;

// 🔥 DEFAULT FALLBACK MODEL
const DEFAULT_MODEL = 'deepseek-ai/deepseek-v3.1';

// 🔥 RETRY CONFIGURATION
const MAX_RETRIES = 3;        // Número de intentos
const RETRY_DELAY_MS = 1500;  // Espera entre intentos (ms)

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

function resolveModel(model) {
  return MODEL_MAPPING[model] || DEFAULT_MODEL;
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function fetchNIM(nimApiBase, nimApiKey, nimRequest) {
  let lastError = null;

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    try {
      const response = await fetch(`${nimApiBase}/chat/completions`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${nimApiKey}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(nimRequest)
      });

      // Si responde OK o es error del cliente (4xx), no reintentar
      if (response.ok || (response.status >= 400 && response.status < 500)) {
        return response;
      }

      // Error 5xx de NVIDIA — reintentar
      lastError = `NVIDIA error ${response.status} on attempt ${attempt}`;
      console.error(lastError);

    } catch (err) {
      lastError = `Network error on attempt ${attempt}: ${err.message}`;
      console.error(lastError);
    }

    if (attempt < MAX_RETRIES) {
      await sleep(RETRY_DELAY_MS);
    }
  }

  throw new Error(`All ${MAX_RETRIES} attempts failed. Last error: ${lastError}`);
}

async function handleChatCompletions(request, env) {
  const NIM_API_BASE = env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
  const NIM_API_KEY = env.NIM_API_KEY;

  const body = await request.json();
  const { model, messages, temperature, max_tokens } = body;

  const nimModel = resolveModel(model);

  const nimRequest = {
    model: nimModel,
    messages,
    temperature: temperature || 0.6,
    max_tokens: max_tokens || 4096,
    stream: true, // 🔥 Forzar streaming siempre
    ...(ENABLE_THINKING_MODE && { extra_body: { chat_template_kwargs: { thinking: true } } })
  };

  const nimResponse = await fetchNIM(NIM_API_BASE, NIM_API_KEY, nimRequest);

  if (!nimResponse.ok) {
    const err = await nimResponse.text();
    return jsonResponse({ error: { message: err, type: 'invalid_request_error', code: nimResponse.status } }, nimResponse.status);
  }

  // --- STREAMING ---
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();
  const nimReader = nimResponse.body.getReader();

  const readable = new ReadableStream({
    async start(controller) {
      let buffer = '';
      let reasoningStarted = false;

      try {
        while (true) {
          const { done, value } = await nimReader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;

            if (line.includes('[DONE]')) {
              controller.enqueue(encoder.encode('data: [DONE]\n\n'));
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

              controller.enqueue(encoder.encode(`data: ${JSON.stringify(data)}\n\n`));
            } catch (_) {
              controller.enqueue(encoder.encode(line + '\n'));
            }
          }
        }
      } catch (err) {
        console.error('Stream error:', err);
      } finally {
        controller.close();
      }
    },
    cancel() {
      nimReader.cancel();
    }
  });

  return new Response(readable, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      ...corsHeaders()
    }
  });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders() });
    }

    if (url.pathname === '/health' && request.method === 'GET') {
      return jsonResponse({
        status: 'ok',
        service: 'OpenAI to NVIDIA NIM Proxy',
        reasoning_display: SHOW_REASONING,
        thinking_mode: ENABLE_THINKING_MODE,
        default_model: DEFAULT_MODEL,
        max_retries: MAX_RETRIES,
        streaming: 'forced'
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

    return jsonResponse({ error: { message: `Endpoint ${url.pathname} not found`, type: 'invalid_request_error', code: 404 } }, 404);
  }
};
