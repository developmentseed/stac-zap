/**
 * Calls a decision model through one of the APIs that serve the jev
 * decisions format: `{ model, state, questions }` in, `{ answers }` out.
 * Also calls an extraction model through the OpenRouter extractions API:
 * `{ model, state, fields }` in, `{ answers }` out.
 * The keys come from `.env`, as for the dev server.
 */
import { readFileSync } from "node:fs";

export type Question = {
  type: "choice";
  instructions: string;
  criteria: Record<string, string>;
};

export type RawAnswer = {
  choice: string;
  confidence?: number;
  probabilities?: Record<string, number>;
};

export type Reply =
  | { answers: Record<string, RawAnswer>; ms: number; cost: number }
  | { error: string };

const PROVIDERS = {
  openrouter: {
    url: "https://openrouter.ai/api/alpha/decisions",
    keyName: "OPENROUTER_API_KEY",
    // OpenRouter's free models allow 20 requests per minute.
    gap: (model: string) => (model.endsWith(":free") ? 3200 : 0),
  },
  codiv: {
    url: "https://api.codiv.ai/v1/systemone",
    keyName: "CODIV_API_KEY",
    // Codiv allows 60 requests per minute.
    gap: () => 1100,
  },
} as const;

export type ProviderName = keyof typeof PROVIDERS;

/** The provider and model of this run, from `PROVIDER` and `MODEL`. */
export function modelOfEnv(): { provider: ProviderName; model: string; tag: string } {
  const provider = (process.env.PROVIDER ?? "openrouter") as ProviderName;
  if (!(provider in PROVIDERS)) throw new Error(`Unknown provider: ${provider}`);
  const model = process.env.MODEL ?? "typesafe/jev-1.14";
  return { provider, model, tag: model.replace(/[/:~]/g, "_") };
}

function keyOf(name: string): string {
  const env = readFileSync(new URL("../.env", import.meta.url), "utf-8");
  const value = new RegExp(`^${name}=(.*)$`, "m").exec(env)?.[1]?.trim();
  if (!value) throw new Error(`${name} is not set in .env`);
  return value.replace(/^"|"$/g, "");
}

let lastCall = 0;

type Result<T> = { json: T; ms: number } | { error: string };

/**
 * Posts one body with the key of `keyName`. Calls are spaced out by `gap` ms
 * to stay under the rate limit of the provider. Rate limits and server
 * errors are tried again; a 4xx error is returned, as it is a limit of the
 * model, e.g. too many options or a field type it does not accept.
 */
async function post<T>(url: string, keyName: string, gap: number, body: unknown): Promise<Result<T>> {
  const key = keyOf(keyName);
  for (let attempt = 0; ; attempt++) {
    const wait = lastCall + gap - Date.now();
    if (wait > 0) await new Promise((r) => setTimeout(r, wait));
    lastCall = Date.now();
    const start = Date.now();
    let status = 0;
    let text: string;
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(120_000),
      });
      status = res.status;
      if (res.ok) return { json: (await res.json()) as T, ms: Date.now() - start };
      text = await res.text();
    } catch (error) {
      text = String(error);
    }
    const retry = status === 0 || status === 429 || status >= 500;
    if (!retry || attempt >= 3) return { error: `${status} ${text.slice(0, 200)}` };
    await new Promise((r) => setTimeout(r, 3000 * (attempt + 1)));
  }
}

/** Asks one set of questions of a decision model. */
export async function ask(
  provider: ProviderName,
  model: string,
  state: string,
  questions: Record<string, Question>,
): Promise<Reply> {
  const { url, keyName, gap } = PROVIDERS[provider];
  const reply = await post<{ answers: Record<string, RawAnswer>; usage?: { cost?: number } }>(
    url,
    keyName,
    gap(model),
    { model, state, questions },
  );
  if ("error" in reply) return reply;
  return { answers: reply.json.answers, ms: reply.ms, cost: reply.json.usage?.cost ?? 0 };
}

/** A field of the OpenRouter extractions API. Only the types the tests use. */
export type ExtractField = {
  type: "text" | "number" | "date";
  instructions?: string;
  unit?: string;
  integer?: boolean;
};

/** One answer of the extractions API, for a text, number or date field. */
export type ExtractAnswer = {
  /** Null when `p_null` is at least the null threshold (0.5). */
  value: string | number | null;
  /** The value the model would give if it had to give one. */
  guess?: string | number | null;
  /** The probability that the request does not contain the value. */
  p_null?: number;
  confidence?: number;
  /** [start, end) code point offsets into the state. */
  span?: [number, number];
};

/** The size of a request. Above a limit, the server splits the fields into more passes. */
export type ExtractUsage = { schema_tokens?: number; state_tokens?: number; passes?: number };

export type ExtractReply =
  | { answers: Record<string, ExtractAnswer>; ms: number; cost: number; usage: ExtractUsage }
  | { error: string };

const EXTRACTIONS_URL = "https://openrouter.ai/api/alpha/extractions";

/**
 * Asks an extraction model on OpenRouter for typed values.
 * @param referenceDate `YYYY-MM-DD`, the date that "last month" counts from.
 */
export async function extract(
  model: string,
  state: string,
  fields: Record<string, ExtractField>,
  referenceDate?: string,
): Promise<ExtractReply> {
  const body: Record<string, unknown> = { model, state, fields };
  // Models without date fields reject a reference date.
  if (referenceDate) body.reference_date = referenceDate;
  const reply = await post<{
    answers: Record<string, ExtractAnswer>;
    usage?: ExtractUsage & { cost?: number };
  }>(EXTRACTIONS_URL, "OPENROUTER_API_KEY", 0, body);
  if ("error" in reply) return reply;
  const { cost, schema_tokens, state_tokens, passes } = reply.json.usage ?? {};
  return {
    answers: reply.json.answers,
    ms: reply.ms,
    cost: cost ?? 0,
    usage: { schema_tokens, state_tokens, passes },
  };
}
