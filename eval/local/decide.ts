/**
 * Phase 0 of the local model plan: answers jev questions with a small open
 * encoder that runs on this computer (ONNX, transformers.js), with no
 * fine-tuning. It gives the baseline that the trained models must beat.
 *
 * The model name selects the scorer:
 * - `bi:<repo>`: a bi-encoder. The query and each option are embedded; the
 *   probabilities are a softmax of the cosine similarities.
 * - `rerank:<repo>[#<onnx file>]`: a cross-encoder. Each (query, option) pair
 *   gets a score; the probabilities are a softmax of the scores.
 *
 * QUERY=prompt (default) embeds the request only; QUERY=full puts the
 * question text before it. TEMP sets the softmax temperature.
 *
 * Usage:
 *   PROVIDER=local MODEL=bi:onnx-community/granite-embedding-30m-english-ONNX pnpm tsx eval/fields.ts
 */
import {
  AutoModel,
  AutoModelForSequenceClassification,
  AutoTokenizer,
  type PreTrainedModel,
  type PreTrainedTokenizer,
  type Tensor,
} from "@huggingface/transformers";
import type { Question, RawAnswer, Reply } from "../providers";

type Scorer = (query: string, options: string[]) => Promise<number[]>;

const loaded = new Map<string, Promise<{ score: Scorer; temp: number }>>();

/** The request, from the state that `buildPayload` writes. */
export function requestOf(state: string): string {
  return /User request: (.*)/.exec(state)?.[1]?.trim() ?? state;
}

/** The option as the model reads it: "<id>: <text>", or the text when the id adds nothing. */
export function optionText(id: string, text: string): string {
  if (id === text || text.toLowerCase().startsWith(id.toLowerCase())) return text;
  return `${id.replaceAll(/[_-]/g, " ")}: ${text}`;
}

function softmax(scores: number[], temp: number): number[] {
  const z = scores.map((s) => s / temp);
  const max = Math.max(...z);
  const e = z.map((v) => Math.exp(v - max));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / sum);
}

/** The CLS vector of each text, normalized (granite pools with CLS). */
async function embed(tok: PreTrainedTokenizer, model: PreTrainedModel, texts: string[]): Promise<number[][]> {
  const out: number[][] = [];
  for (let i = 0; i < texts.length; i += 64) {
    const batch = texts.slice(i, i + 64);
    const x = tok(batch, { padding: true, truncation: true, max_length: 256 });
    const h = (await model(x)).last_hidden_state as Tensor;
    const [n, t, d] = h.dims as [number, number, number];
    const data = h.data as Float32Array;
    for (let j = 0; j < n; j++) {
      const v = Array.from(data.subarray(j * t * d, j * t * d + d));
      const norm = Math.hypot(...v) || 1;
      out.push(v.map((a) => a / norm));
    }
  }
  return out;
}

async function load(spec: string): Promise<{ score: Scorer; temp: number }> {
  const [kind, rest] = spec.split(/:(.*)/s) as [string, string];
  const [repo, file] = rest.split("#") as [string, string | undefined];
  const tok = await AutoTokenizer.from_pretrained(repo);
  if (kind === "bi") {
    const model = await AutoModel.from_pretrained(repo, { dtype: "q8" });
    const cache = new Map<string, number[]>();
    return {
      temp: Number(process.env.TEMP ?? 0.02),
      score: async (query, options) => {
        const todo = [...new Set([query, ...options])].filter((t) => !cache.has(t));
        const vecs = await embed(tok, model, todo);
        todo.forEach((t, i) => cache.set(t, vecs[i]!));
        const q = cache.get(query)!;
        return options.map((o) => cache.get(o)!.reduce((s, v, i) => s + v * q[i]!, 0));
      },
    };
  }
  if (kind === "rerank") {
    const model = await AutoModelForSequenceClassification.from_pretrained(repo, {
      dtype: "fp32",
      ...(file ? { model_file_name: file } : {}),
    });
    return {
      temp: Number(process.env.TEMP ?? 1),
      score: async (query, options) => {
        const scores: number[] = [];
        for (let i = 0; i < options.length; i += 32) {
          const batch = options.slice(i, i + 32);
          const x = tok(Array(batch.length).fill(query), {
            text_pair: batch,
            padding: true,
            truncation: true,
            max_length: 256,
          });
          const logits = (await model(x)).logits as Tensor;
          scores.push(...Array.from(logits.data as Float32Array));
        }
        return scores;
      },
    };
  }
  throw new Error(`Unknown local model: ${spec} (use bi:<repo> or rerank:<repo>)`);
}

/** Answers each question with the local model, in the reply format of the decisions API. */
export async function askLocal(
  model: string,
  state: string,
  questions: Record<string, Question>,
): Promise<Reply> {
  if (!loaded.has(model)) loaded.set(model, load(model));
  const { score, temp } = await loaded.get(model)!;
  const request = requestOf(state);
  const start = performance.now();
  const answers: Record<string, RawAnswer> = {};
  for (const [name, q] of Object.entries(questions)) {
    const ids = Object.keys(q.criteria);
    const query = process.env.QUERY === "full" ? `${q.instructions} Request: ${request}` : request;
    const scores = await score(query, ids.map((id) => optionText(id, q.criteria[id]!)));
    const p = softmax(scores, temp);
    const probabilities = Object.fromEntries(ids.map((id, i) => [id, p[i]!]));
    const best = ids[p.indexOf(Math.max(...p))]!;
    answers[name] = { choice: best, confidence: probabilities[best], probabilities };
  }
  return { answers, ms: Math.round(performance.now() - start), cost: 0 };
}
