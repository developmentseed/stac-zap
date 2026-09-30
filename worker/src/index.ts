/**
 * The decisions proxy for the deployed site. It does what `/api/decide` does
 * in the dev server (see `vite.config.ts`): it forwards a jev decisions body
 * to OpenRouter with the key, which stays in the Worker as a secret.
 *
 * Only the site can call it, only the decisions API is reachable, and the
 * model is always jev. A request limit per visitor stops one browser from
 * using the budget. The budget limit on the OpenRouter key is the last check.
 */

type Env = {
  /** The OpenRouter key. Set it with `wrangler secret put OPENROUTER_API_KEY`. */
  OPENROUTER_API_KEY: string;
  /** Comma-separated origins that can call the Worker, e.g. the Pages site. */
  ALLOWED_ORIGINS: string;
  /** The jev model id. Requests cannot choose another model. */
  JEV_MODEL: string;
};

const DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions";
/** A zap body with 250 collections is about 60 kB. */
const MAX_BODY = 256 * 1024;
/** A zap is two or three calls, so this is about ten zaps a minute. */
const MAX_REQUESTS = 30;
const WINDOW_MS = 60_000;
const MAX_QUESTIONS = 10;

// Counts per visitor, per Worker instance. Cloudflare runs many instances,
// so this is a soft limit, not an exact one.
const recent = new Map<string, number[]>();

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const origin = request.headers.get("Origin") ?? "";
    const allowed = env.ALLOWED_ORIGINS.split(",").map((o) => o.trim());
    if (!allowed.includes(origin)) {
      return new Response("Origin not allowed", { status: 403 });
    }
    const cors = {
      "Access-Control-Allow-Origin": origin,
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
      "Access-Control-Max-Age": "86400",
      Vary: "Origin",
    };
    if (request.method === "OPTIONS") return new Response(null, { headers: cors });
    if (request.method !== "POST") {
      return new Response("Use POST", { status: 405, headers: cors });
    }

    const visitor = request.headers.get("CF-Connecting-IP") ?? "unknown";
    if (tooMany(visitor)) {
      return new Response("Too many requests. Wait a minute.", {
        status: 429,
        headers: cors,
      });
    }

    const text = await request.text();
    if (text.length > MAX_BODY) {
      return new Response("Request too large", { status: 413, headers: cors });
    }
    let body: { state?: unknown; questions?: Record<string, unknown> };
    try {
      body = JSON.parse(text);
    } catch {
      return new Response("Not JSON", { status: 400, headers: cors });
    }
    const questions = body.questions;
    if (
      typeof body.state !== "string" ||
      !questions ||
      typeof questions !== "object" ||
      Object.keys(questions).length === 0 ||
      Object.keys(questions).length > MAX_QUESTIONS
    ) {
      return new Response("Needs a state and 1 to 10 questions", {
        status: 400,
        headers: cors,
      });
    }

    const response = await fetch(DECISIONS_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${env.OPENROUTER_API_KEY}`,
        "Content-Type": "application/json",
        "X-Title": "stac-zap",
      },
      body: JSON.stringify({ model: env.JEV_MODEL, state: body.state, questions }),
    });
    return new Response(response.body, {
      status: response.status,
      headers: { ...cors, "Content-Type": "application/json" },
    });
  },
};

function tooMany(visitor: string): boolean {
  const now = Date.now();
  const times = (recent.get(visitor) ?? []).filter((t) => now - t < WINDOW_MS);
  times.push(now);
  recent.set(visitor, times);
  return times.length > MAX_REQUESTS;
}
