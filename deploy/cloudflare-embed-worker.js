/**
 * Cloudflare Worker: OpenAI embeddings proxy (avoids browser CORS to api.openai.com).
 *
 * Setup:
 *   1. Create worker at dash.cloudflare.com, paste this script.
 *   2. Settings → Variables → Secrets: OPENAI_API_KEY = sk-...
 *   3. Deploy; set frontend/config.json "embedApiUrl" to "https://your-worker.workers.dev"
 *
 * Request:  POST { "input": "query text", "model": "text-embedding-3-small" }
 * Response: same shape as OpenAI /v1/embeddings (JSON).
 */
const OPENAI_URL = "https://api.openai.com/v1/embeddings";
const MAX_INPUT_LEN = 12000;

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, { headers: corsHeaders });
    }
    if (request.method !== "POST") {
      return json({ error: "Use POST" }, 405);
    }
    const key = env.OPENAI_API_KEY;
    if (!key) {
      return json({ error: "Worker missing OPENAI_API_KEY secret" }, 500);
    }
    let body;
    try {
      body = await request.json();
    } catch {
      return json({ error: "Invalid JSON" }, 400);
    }
    const input = body.input;
    const model = body.model || "text-embedding-3-small";
    if (typeof input !== "string" || !input.trim()) {
      return json({ error: "Missing input string" }, 400);
    }
    if (input.length > MAX_INPUT_LEN) {
      return json({ error: "Input too long" }, 400);
    }
    const r = await fetch(OPENAI_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${key}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ model, input: input.trim() }),
    });
    const text = await r.text();
    return new Response(text, {
      status: r.status,
      headers: {
        ...corsHeaders,
        "Content-Type": "application/json",
      },
    });
  },
};

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { ...corsHeaders, "Content-Type": "application/json" },
  });
}
