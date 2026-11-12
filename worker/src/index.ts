/**
 * The model proxy for the deployed site. It does what `/api/decide` and
 * `/api/extract` do in the dev server (see `vite.config.ts`): it forwards a
 * body to OpenRouter with the key, which stays in the Worker as a secret.
 *
 * - `POST /` (or `/decide`): a jev decisions body, to the decisions API.
 * - `POST /extract`: an extraction body, to the extractions API.
 *
 * Only the site can call it, only these two APIs are reachable, and the
 * models are fixed: jev for decisions, Clairic for extractions. A request
 * limit per visitor stops one browser from using the budget. The budget
 * limit on the OpenRouter key is the last check.
 */

type Env = {
  /** The OpenRouter key. Set it with `wrangler secret put OPENROUTER_API_KEY`. */
  OPENROUTER_API_KEY: string;
  /** Comma-separated origins that can call the Worker, e.g. the Pages site. */
  ALLOWED_ORIGINS: string;
  /** The jev model id. Requests cannot choose another model. */
  JEV_MODEL: string;
  /** The extraction model id, e.g. Clairic. Requests cannot choose another model. */
  EXTRACT_MODEL: string;
};

const DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions";
const EXTRACTIONS_URL = "https://openrouter.ai/api/alpha/extractions";
/** A zap body with 250 collections is about 60 kB. */
const MAX_BODY = 256 * 1024;
/** A zap is four or five calls (two extractions, two or three decisions), so this is about ten zaps a minute. */
const MAX_REQUESTS = 50;
const WINDOW_MS = 60_000;
const MAX_QUESTIONS = 10;
/** The app asks at most 4 fields in one extraction call. */
const MAX_FIELDS = 8;
const FIELD_TYPES = new Set(["text", "number", "date"]);
const FIELD_KEYS = new Set(["type", "instructions", "unit"]);
const MAX_INSTRUCTIONS = 1000;

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
    let body: Record<string, unknown>;
    try {
      body = JSON.parse(text);
    } catch {
      return new Response("Not JSON", { status: 400, headers: cors });
    }
    const path = new URL(request.url).pathname.replace(/\/+$/, "");
    let forward: { url: string; body: unknown };
    if (path === "/extract") {
      const problem = checkExtraction(body);
      if (problem) return new Response(problem, { status: 400, headers: cors });
      forward = {
        url: EXTRACTIONS_URL,
        body: {
          model: env.EXTRACT_MODEL,
          state: body.state,
          fields: body.fields,
          ...(body.reference_date ? { reference_date: body.reference_date } : {}),
        },
      };
    } else if (path === "" || path === "/decide") {
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
      forward = { url: DECISIONS_URL, body: { model: env.JEV_MODEL, state: body.state, questions } };
    } else {
      return new Response("Not found", { status: 404, headers: cors });
    }

    const response = await fetch(forward.url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${env.OPENROUTER_API_KEY}`,
        "Content-Type": "application/json",
        "X-Title": "stac-zap",
      },
      body: JSON.stringify(forward.body),
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

/**
 * Why an extraction body is not one the app sends, or null when it is: a
 * state, 1 to 8 text, number or date fields with short instructions, and an
 * optional `YYYY-MM-DD` reference date. Other keys are not forwarded.
 */
function checkExtraction(body: Record<string, unknown>): string | null {
  if (typeof body.state !== "string") return "Needs a state";
  const fields = body.fields;
  if (!fields || typeof fields !== "object" || Array.isArray(fields)) return "Needs fields";
  const entries = Object.entries(fields as Record<string, unknown>);
  if (entries.length === 0 || entries.length > MAX_FIELDS) return `Needs 1 to ${MAX_FIELDS} fields`;
  for (const [name, field] of entries) {
    if (!field || typeof field !== "object") return `Field ${name} is not an object`;
    const f = field as Record<string, unknown>;
    if (Object.keys(f).some((k) => !FIELD_KEYS.has(k))) return `Field ${name} has an unknown key`;
    if (!FIELD_TYPES.has(f.type as string)) return `Field ${name} has an unknown type`;
    if (f.instructions !== undefined && (typeof f.instructions !== "string" || f.instructions.length > MAX_INSTRUCTIONS)) {
      return `Field ${name} has instructions that are not text or too long`;
    }
    if (f.unit !== undefined && typeof f.unit !== "string") return `Field ${name} has a unit that is not text`;
  }
  if (body.reference_date !== undefined && !/^\d{4}-\d{2}-\d{2}$/.test(String(body.reference_date))) {
    return "The reference date is not YYYY-MM-DD";
  }
  return null;
}
