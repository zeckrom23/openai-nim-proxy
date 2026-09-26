// worker.js - OpenAI to NVIDIA NIM API Proxy (Cloudflare Workers)
// ✅ Anti-524 edition — streaming forzado, timeouts, rotación de keys en 429, keepalive
// ✅ v2 — fallback global para apagar "thinking mode" en modelos no listados (fix de latencia)

// 🔥 REASONING DISPLAY TOGGLE
const SHOW_REASONING = false;

// 🔥 THINKING MODE TOGGLE (default global, para modelos NO listados abajo)
const ENABLE_THINKING_MODE = false;

// 🔥 DEFAULT FALLBACK MODEL
// ✅ Actualizado: deepseek-v4-flash-0731 se deprecó, ahora usa el sucesor v4.1
const DEFAULT_MODEL = 'deepseek-ai/deepseek-v4.1-flash';

// ⏱️ TIMEOUT en ms para esperar headers de NIM (no es timeout total, solo TTFB)
// ✅ FIX v2: 10s resultó DEMASIADO agresivo — modelos grandes (kimi-k3 2.8T,
// glm-5.3) normalmente tardan 10+ segundos en TTFB por su tamaño, nada que
// ver con estar atascados. Con 10s se estaban abortando conexiones sanas,
// rotando las 4 keys y agotándolas todas → 524 en TODOS los modelos grandes.
// 25s es punto medio: suficiente para prefill normal de modelos grandes,
// sin volver a los 60s que comían un minuto entero por key atascada real.
const HEADER_TIMEOUT_MS = 25000;

// 🧠 THINKING BUDGET — 0 = sin thinking (más rápido para roleplay)
const THINKING_BUDGET = 0;

// 💓 KEEPALIVE — manda comentarios SSE invisibles cada N ms para evitar 524
const KEEPALIVE_INTERVAL_MS = 15000;

// 🧠 Modelos genéricos con thinking que aceptan extra_body.chat_template_kwargs
const THINKING_MODELS = [
  'bytedance/seed-oss-36b-instruct',
  'qwen/qwen3-next-80b-a3b-thinking',
];

// 🧠 Modelos Nemotron que necesitan chat_template_kwargs directo en la raíz (NO en extra_body)
const NEMOTRON_MODELS = [
  'nvidia/nemotron-3.5-lightning-30b-a3b',
  'nvidia/nemotron-3-ultra-550b-a55b',
  'nvidia/nemotron-3-super-120b-a12b',
  'nvidia/nvidia-nemotron-nano-9b-v2',
];

// 🧠 Modelos MiniMax / gpt-oss que necesitan chat_template_kwargs directo (no en extra_body)
const MINIMAX_MODELS = [
  'minimaxai/minimax-m3',
  'minimaxai/minimax-m2.7',
  'openai/gpt-oss-20b',
];

// Model mapping - Updated August 2026
// ✅ Solo los modelos que uso activamente. Los demás quedan comentados abajo
// para reactivarlos rápido cuando salga algo nuevo o quiera probar otro.
const MODEL_MAPPING = {
  // 🔥 DEEPSEEK V4 - Mejor para roleplay NSFW
  // ✅ v4-pro-0813 se deprecó y aún no existe v4.1-pro (DeepSeek confirmó que
  // sigue en desarrollo, sin fecha). Mientras tanto DeepSeek está redirigiendo
  // TODAS las peticiones a "Pro" hacia v4.1-flash por detrás — así que apuntamos
  // gpt-4o directo ahí también, en vez de a un Pro que ya no existe.
  'gpt-4o':             'deepseek-ai/deepseek-v4.1-flash',
  'gpt-4':              'deepseek-ai/deepseek-v4.1-flash',
  // 🔥 Writer & Kimi - Bueno para roleplay
  // ❌ minimaxai/minimax-m3 NO existe en tu cuenta de NIM (confirmado con
  // /v1/models) — lo cambié por palmyra-creative, hecho para escritura creativa.
  'gpt-4o-mini':        'writer/palmyra-creative-122b',
  'claude-3-opus':      'moonshotai/kimi-k3',
  // ✅ kimi-k2-instruct-0905 no existe en tu cuenta, pero kimi-k2.6 sí — este
  // es tu Kimi ligero real, confirmado en /v1/models.
  'claude-3-sonnet':    'moonshotai/kimi-k2.6',
  // 🔥 Respaldos
  'o1':                 'z-ai/glm-5.3',
  'o1-mini':            'z-ai/glm-5.3-flash',
  // 🔥 NEMOTRON LIGHTNING - El más rápido del catálogo, buen respaldo si otros se saturan
  'o3-mini':            'nvidia/nemotron-3.5-lightning-30b-a3b',

  // ── Sin usar por ahora, descomenta para activar ──
  // 🔥 MISTRAL - Parcialmente censurado pero estable
  // 'o1-preview':         'mistralai/mistral-large-3-675b-instruct-2512',
  // 🔥 QWEN - Variedad, MoE grandes
  // 'claude-3-haiku':     'qwen/qwen3.5-122b-a10b',
  // 🔥 NEMOTRON ULTRA - El monstruo de 550B
  // 'o3':                 'nvidia/nemotron-3-ultra-550b-a55b',
  // 'o4-mini':            'nvidia/nemotron-3-super-120b-a12b',
  // 🔥 LLAMA 4 + SEED
  // 'gemini-ultra':       'meta/llama-4-maverick-17b-128e-instruct',
  // 'gemini-pro':         'bytedance/seed-oss-36b-instruct',
  // 🔥 GEMMA 4 - Google
  // 'gemini-flash':       'google/gemma-4-31b-it',
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
  ].filter(Boolean);
  for (let i = keys.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [keys[i], keys[j]] = [keys[j], keys[i]];
  }
  return keys;
}

// ✅ Fetch con rotación de keys en 429 (timeout solo cubre espera de headers/TTFB)
async function fetchNIMWithRotation(url, options, apiKeys) {
  let lastStatus = null;
  let lastError = null;
  for (let i = 0; i < apiKeys.length; i++) {
    const key = apiKeys[i];
    let headerTimer = null;
    const attemptStart = Date.now();
    try {
      const controller = new AbortController();
      headerTimer = setTimeout(() => controller.abort(), HEADER_TIMEOUT_MS);
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
        headers: {
          ...options.headers,
          'Authorization': `Bearer ${key}`,
        }
      });
      clearTimeout(headerTimer);
      headerTimer = null;
      console.log(`Key ${i + 1}/${apiKeys.length}: headers en ${Date.now() - attemptStart}ms, status ${response.status}`);
      if (response.status === 429) {
        lastStatus = 429;
        console.warn(`Key ${i + 1}/${apiKeys.length} got 429, trying next...`);
        continue;
      }
      return response;
    } catch (err) {
      if (headerTimer) clearTimeout(headerTimer);
      lastError = err;
      if (err.name === 'AbortError') {
        console.warn(`Key ${i + 1}/${apiKeys.length} headers timed out after ${Date.now() - attemptStart}ms, trying next...`);
        continue;
      }
      console.warn(`Key ${i + 1}/${apiKeys.length} network error after ${Date.now() - attemptStart}ms: ${err.message}, trying next...`);
    }
  }
  if (lastStatus === 429) {
    return new Response(JSON.stringify({
      status: 429,
      title: 'Too Many Requests',
      detail: 'All API keys are rate limited. Try again in a moment.'
    }), { status: 429 });
  }
  throw lastError || new Error('All API keys failed');
}

// ✅ Consume el stream internamente y devuelve el contenido completo (para clientes non-stream)
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
  const { model, messages, temperature, max_tokens, stream, frequency_penalty, presence_penalty, repetition_penalty } = body;
  const clientWantsStream = stream === true;
  const nimModel = resolveModel(model);
  const isThinkingModel = THINKING_MODELS.includes(nimModel);
  const isNemotronModel = NEMOTRON_MODELS.includes(nimModel);
  const isMinimaxModel = MINIMAX_MODELS.includes(nimModel);

  const nimRequest = {
    model: nimModel,
    messages,
    temperature: temperature || 0.6,
    // ✅ Reenviamos los sliders de "repetición" de JanitorAI — sin esto tus
    // valores de Rep./Freq. penalty nunca llegaban a NIM, por eso no hacían
    // nada contra el loop de "!!!!". A propósito NO reenviamos top_p: kimi-k3
    // lo trae fijo en 0.95 y truena (400) si mandas otro valor.
    ...(frequency_penalty !== undefined ? { frequency_penalty } : {}),
    ...(presence_penalty !== undefined ? { presence_penalty } : {}),
    ...(repetition_penalty !== undefined ? { repetition_penalty } : {}),
    // ✅ Si el cliente no manda max_tokens (o manda 0 = "infinito" en JanitorAI),
    // no forzamos ningún límite — dejamos que NIM use su propio default.
    ...(max_tokens ? { max_tokens } : {}),
    stream: true,
  };

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // Configuración de chat_template_kwargs por familia
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  if (isMinimaxModel) {
    nimRequest.chat_template_kwargs = { thinking_mode: 'disabled' };
  } else if (isNemotronModel) {
    // ✅ Nemotron: directo en la raíz para evitar error 400 por extra_body
    nimRequest.chat_template_kwargs = {
      thinking: THINKING_BUDGET > 0,
      budget_tokens: THINKING_BUDGET
    };
  } else if (isThinkingModel) {
    // ✅ FIX: `extra_body` NO es un campo real de la REST API de NIM — es una
    // convención del SDK de Python/Node de OpenAI que el cliente desempaqueta
    // antes de mandar el request. Como aquí armamos el JSON a mano, hay que
    // mandar chat_template_kwargs directo en la raíz (igual que Nemotron/Minimax),
    // si no NIM lo rechaza con 400 "Unsupported parameter(s): extra_body".
    nimRequest.chat_template_kwargs = { thinking: THINKING_BUDGET > 0, budget_tokens: THINKING_BUDGET };
  } else if (ENABLE_THINKING_MODE) {
    nimRequest.chat_template_kwargs = { thinking: true };
  } else {
    // ✅ FIX: antes, cualquier modelo fuera de las 3 listas (deepseek-v4, glm-5.3,
    // qwen3.5, mistral, llama-4, gemma-4...) no recibía NINGÚN chat_template_kwargs,
    // así que corría con el default del servidor — que en varias familias viene
    // con "thinking" prendido de fábrica. Eso hace que el modelo genere tokens de
    // razonamiento completos (lento) que luego se descartan porque SHOW_REASONING
    // es false. Mandamos thinking:false como default seguro, directo en la raíz
    // (no en extra_body — ver nota arriba). Si el modelo no reconoce la clave,
    // el chat template normalmente la ignora sin romper el request.
    nimRequest.chat_template_kwargs = { thinking: false };
  }

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

  // ✅ Auto-retry: algunos modelos (ej. kimi-k3) traen ciertos parámetros
  // (top_p, frequency_penalty, etc.) fijos e inmutables, y varían cuál según
  // el modelo. En vez de mantener una lista a mano por modelo, detectamos el
  // error "X is immutable" de NIM, quitamos ese campo, y reintentamos —hasta
  // 5 veces por si el modelo se queja de varios parámetros uno por uno.
  let immutableRetries = 0;
  while (!nimResponse.ok && immutableRetries < 5) {
    const errText = await nimResponse.clone().text();
    const match = errText.match(/`(\w+)`\s+is immutable/i);
    if (!match) break;
    const badField = match[1];
    console.warn(`NIM dice que '${badField}' es inmutable para este modelo — quitando y reintentando`);
    delete nimRequest[badField];
    immutableRetries++;
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
          error: { message: 'NIM no respondió a tiempo tras reintentar sin parámetros inmutables.', type: 'timeout_error', code: 524 }
        }, 524);
      }
      return jsonResponse({
        error: { message: `Error de red: ${err.message}`, type: 'network_error', code: 503 }
      }, 503);
    }
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
  // STREAMING con keepalive anti-524
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  if (clientWantsStream) {
    const encoder = new TextEncoder();
    const decoder = new TextDecoder();
    const nimReader = nimResponse.body.getReader();
    const readable = new ReadableStream({
      async start(controller) {
        let buffer = '';
        let reasoningStarted = false;
        // 💓 Keepalive — manda comentarios SSE invisibles para evitar 524
        const keepalive = setInterval(() => {
          try {
            controller.enqueue(encoder.encode(': keepalive\n\n'));
          } catch (_) {}
        }, KEEPALIVE_INTERVAL_MS);
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
          clearInterval(keepalive);
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
        thinking_mode_default: ENABLE_THINKING_MODE,
        thinking_budget: THINKING_BUDGET,
        default_model: DEFAULT_MODEL,
        total_models: Object.keys(MODEL_MAPPING).length,
        header_timeout_ms: HEADER_TIMEOUT_MS,
        keepalive_interval_ms: KEEPALIVE_INTERVAL_MS,
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
