// worker.js - OpenAI to NVIDIA NIM API Proxy (Cloudflare Workers)
// ✅ Anti-524 edition — streaming forzado, timeouts, rotación de keys en 429

// 🔥 REASONING DISPLAY TOGGLE
const SHOW_REASONING = false;

// 🔥 THINKING MODE TOGGLE
const ENABLE_THINKING_MODE = false;

// 🔥 DEFAULT FALLBACK MODEL
const DEFAULT_MODEL = 'deepseek-ai/deepseek-v4-pro';

// ⏱️ TIMEOUT en ms — 90s seguro con usage_model = "unbound" activo
const NIM_TIMEOUT_MS = 90000;

// 🧠 THINKING BUDGET — 0 = sin thinking (más rápido para roleplay)
const THINKING_BUDGET = 0;

// 🧠 Modelos con thinking que se benefician del THINKING_BUDGET
const THINKING_MODELS = [
  'moonshotai/kimi-k2.6',
  'moonshotai/kimi-k2-thinking',
  'qwen/qwen3-next-80b-a3b-thinking',
];

// Model mapping - Updated May 2026
const MODEL_MAPPING = {

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // 🔥 DEEPSEEK V4 - Mejor para roleplay
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  'gpt-4o':             'deepseek-ai/deepseek-v4-pro',
  'gpt-4-turbo':        'deepseek-ai/deepseek-v4-flash',

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // 🔥 DEEPSEEK V3 - Backup confiable
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  'gpt-4':              'deepseek-ai/deepseek-v3.2',
  'gpt-4-5':            'deepseek-ai/deepseek-v3.1-terminus', // 🤙 El chill de la familia

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // 🔥 KIMI - Muy bueno para narrativa
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  'gpt-4o-mini':        'moonshotai/kimi-k2-instruct',
  'claude-3-opus':      'moonshotai/kimi-k2.6',

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // 🔥 GLM - Bueno para NSFW
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  'o3':                 'z-ai/glm-5.1',
  'o4-mini':            'z-ai/glm-4.7',

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // 🔥 OPENAI OSS - Estable y rápido
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  'gpt-3.5-turbo':      'openai/gpt-oss-120b',
  'gpt-3.5-turbo-16k':  'openai/gpt-oss-20b',

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // 🔥 MISTRAL - Grande y capaz
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  'o1':                 'mistralai/devstral-2-123b-instruct-2512',
  'o1-mini':            'mistralai/mistral-large-3-675b-instruct-2512',
  'o1-preview':         'mistralai/mistral-medium-3.5-128b',

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // 🔥 MINIMAX - Razonamiento
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  'o3-mini':            'minimaxai/minimax-m2.7',

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // 🔥 QWEN - Variedad de opciones
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  'claude-3-sonnet':    'qwen/qwen3-next-80b-a3b-instruct',
  'claude-3-haiku':     'qwen/qwen3-coder-480b-a35b-instruct',

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // 🔥 LLAMA 4 - Nuevo de Meta
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  'gemini-ultra':       'meta/llama-4-maverick-17b-128e-instruct',

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // 🔥 NEMOTRON - Backup NVIDIA nativo
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  'gemini-pro':         'nvidia/nemotron-3-super-120b-a12b',
  'gemini-flash':       'nvidia/llama-3.1-nemotron-ultra-253b-v1',
};

// ─────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────

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

// ✅ Obtiene las keys disponibles desde env y las mezcla aleatoriamente
function getApiKeys(env) {
  const keys = [
    env.NIM_API_KEY,
    env.NIM_API_KEY_1,
    env.NIM_API_KEY_2,
    env.NIM_API_KEY_3,
  ].filter(Boolean); // filtra las que no estén definidas

  // Mezcla aleatoria para distribuir la carga
  for (let i = keys.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [keys[i], keys[j]] = [keys[j], keys[i]];
  }

  return keys;
}

// ✅ Fetch a NIM con rotación automática de keys en 429 y timeout
async function fetchNIMWithRotation(url, options, apiKeys) {
  let lastError = null;
  let lastStatus = null;

  for (let i = 0; i < apiKeys.length; i++) {
    const key = apiKeys[i];

    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), NIM_TIMEOUT_MS);

      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
        headers: {
          ...options.headers,
          'Authorization': `Bearer ${key}`,
        }
      });
      clearTimeout(timer);

      // Si no es 429, devuelve la respuesta (ok o error distinto)
      if (response.status !== 429) {
        return response;
      }

      // Es 429 — registra y prueba la siguiente key
      lastStatus = 429;
      console.warn(`Key ${i + 1}/${apiKeys.length} got 429, trying next key...`);

    } catch (err) {
      clearTimeout && clearTimeout();
      lastError = err;

      if (err.name === 'AbortError') {
        // Timeout — no tiene sentido rotar keys por esto, lanza directo
        throw err;
      }

      console.warn(`Key ${i + 1}/${apiKeys.length} network error: ${err.message}, trying next...`);
    }
  }

  // Todas las keys fallaron
  if (lastStatus === 429) {
    // Devuelve un 429 simulado para que JAI lo muestre
    return new Response(JSON.stringify({
      status: 429,
      title: 'Too Many Requests',
      detail: 'All API keys are rate limited. Try again in a moment.'
    }), { status: 429 });
  }

  throw lastError || new Error('All API keys failed');
}

// ✅ Consume el stream internamente y devuelve el contenido completo
async function collectStream(nimResponse) {
  const decoder = new TextDecoder();
  const reader = nimResponse.body.getReader();
  let fullContent = '';
  let fullReasoning = '';
  let lastData = null;
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (!line.startsWith('data: ') || line.includes('[DONE]')) continue;
      try {
        const data = JSON.parse(line.slice(6));
        lastData = data;
        fullContent += data.choices?.[0]?.delta?.content || '';
        fullReasoning += data.choices?.[0]?.delta?.reasoning_content || '';
      } catch (_) {}
    }
  }

  return { fullContent, fullReasoning, lastData };
}

// ─────────────────────────────────────────
// HANDLER PRINCIPAL
// ─────────────────────────────────────────

async function handleChatCompletions(request, env) {
  const NIM_API_BASE = env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';

  const body = await request.json();
  const { model, messages, temperature, max_tokens, stream } = body;
  const clientWantsStream = stream === true;

  const nimModel = resolveModel(model);
  const isThinkingModel = THINKING_MODELS.includes(nimModel);

  const thinkingExtra = isThinkingModel
    ? { chat_template_kwargs: { thinking: THINKING_BUDGET > 0, budget_tokens: THINKING_BUDGET } }
    : undefined;

  const nimRequest = {
    model: nimModel,
    messages,
    temperature: temperature || 0.6,
    max_tokens: max_tokens || 4096,
    stream: true,
    ...(thinkingExtra && { extra_body: thinkingExtra }),
    ...(ENABLE_THINKING_MODE && !isThinkingModel && { extra_body: { chat_template_kwargs: { thinking: true } } })
  };

  const apiKeys = getApiKeys(env);

  if (apiKeys.length === 0) {
    return jsonResponse({
      error: { message: 'No API keys configured', type: 'auth_error', code: 401 }
    }, 401);
  }

  let nimResponse;
  try {
    nimResponse = await fetchNIMWithRotation(
      `${NIM_API_BASE}/chat/completions`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(nimRequest)
      },
      apiKeys
    );
  } catch (err) {
    if (err.name === 'AbortError') {
      return jsonResponse({
        error: {
          message: 'NIM no respondió a tiempo. Intenta con un modelo más ligero o reintenta.',
          type: 'timeout_error',
          code: 524
        }
      }, 524);
    }
    return jsonResponse({
      error: { message: `Error de red: ${err.message}`, type: 'network_error', code: 503 }
    }, 503);
  }

  if (!nimResponse.ok) {
    const err = await nimResponse.text();
    return jsonResponse({
      error: {
        message: `NIM Error (${nimResponse.status}): ${err}`,
        type: 'invalid_request_error',
        code: nimResponse.status
      }
    }, nimResponse.status);
  }

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // STREAMING
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  if (clientWantsStream) {
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

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // NON-STREAMING
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  const { fullContent, fullReasoning, lastData } = await collectStream(nimResponse);

  let content = fullContent;
  if (SHOW_REASONING && fullReasoning) {
    content = '<think>\n' + fullReasoning + '\n</think>\n\n' + content;
  }

  const openaiResponse = {
    id: `chatcmpl-${Date.now()}`,
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [{
      index: 0,
      message: { role: 'assistant', content },
      finish_reason: lastData?.choices?.[0]?.finish_reason || 'stop'
    }],
    usage: lastData?.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
  };

  return jsonResponse(openaiResponse);
}

// ─────────────────────────────────────────
// ENTRY POINT
// ─────────────────────────────────────────

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
        thinking_budget: THINKING_BUDGET,
        default_model: DEFAULT_MODEL,
        total_models: Object.keys(MODEL_MAPPING).length,
        timeout_ms: NIM_TIMEOUT_MS,
        api_keys_configured: ['NIM_API_KEY', 'NIM_API_KEY_1', 'NIM_API_KEY_2', 'NIM_API_KEY_3']
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
        return jsonResponse({
          error: { message: err.message || 'Internal server error', type: 'invalid_request_error', code: 500 }
        }, 500);
      }
    }

    return jsonResponse({
      error: { message: `Endpoint ${url.pathname} not found`, type: 'invalid_request_error', code: 404 }
    }, 404);
  }
};