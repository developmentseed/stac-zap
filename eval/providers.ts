/**
 * Calls a decision model through one of the APIs that serve the jev
 * decisions format: `{ model, state, questions }` in, `{ answers }` out.
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
  const model = process.env.MODEL ?? "typesafe/jev-1.13";
  return { provider, model, tag: model.replace(/[/:~]/g, "_") };
}

function keyOf(name: string): string {
  const env = readFileSync(new URL("../.env", import.meta.url), "utf-8");
  const value = new RegExp(`^${name}=(.*)$`, "m").exec(env)?.[1]?.trim();
  if (!value) throw new Error(`${name} is not set in .env`);
  return value.replace(/^"|"$/g, "");
}

let lastCall = 0;

/**
 * Asks one set of questions. Calls are spaced out to stay under the rate
 * limit of the provider. Rate limits and server errors are tried again; a
 * 4xx error is returned, as it is a limit of the model, e.g. too many
 * options.
 */
export async function ask(
  provider: ProviderName,
  model: string,
  state: string,
  questions: Record<string, Question>,
): Promise<Reply> {
  const { url, keyName, gap } = PROVIDERS[provider];
  const key = keyOf(keyName);
  for (let attempt = 0; ; attempt++) {
    const wait = lastCall + gap(model) - Date.now();
    if (wait > 0) await new Promise((r) => setTimeout(r, wait));
    lastCall = Date.now();
    const start = Date.now();
    let status = 0;
    let text: string;
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
        body: JSON.stringify({ model, state, questions }),
        signal: AbortSignal.timeout(120_000),
      });
      status = res.status;
      if (res.ok) {
        const json = (await res.json()) as {
          answers: Record<string, RawAnswer>;
          usage?: { cost?: number };
        };
        return { answers: json.answers, ms: Date.now() - start, cost: json.usage?.cost ?? 0 };
      }
      text = await res.text();
    } catch (error) {
      text = String(error);
    }
    const retry = status === 0 || status === 429 || status >= 500;
    if (!retry || attempt >= 3) return { error: `${status} ${text.slice(0, 200)}` };
    await new Promise((r) => setTimeout(r, 3000 * (attempt + 1)));
  }
}
