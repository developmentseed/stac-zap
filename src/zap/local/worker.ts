/**
 * Runs the local models in a Web Worker, so the page stays responsive:
 * - retriever: embeds texts (CLS vector, normalized),
 * - decider: scores (query, option) pairs; a softmax with the calibrated
 *   temperature of the question gives the probabilities,
 * - tagger: marks the place, time and cloud cover words of a request.
 *
 * The model files come from `base` (see `index.ts`), in the layout that
 * transformers.js reads; int8 weights ("q8"), computed in fp32.
 */
import {
  AutoModel,
  AutoModelForSequenceClassification,
  AutoModelForTokenClassification,
  AutoTokenizer,
  env,
  type PreTrainedModel,
  type PreTrainedTokenizer,
  type Tensor,
} from "@huggingface/transformers";

export type Request =
  | { id: number; type: "load"; base: string }
  | { id: number; type: "embed"; texts: string[]; maxLength: number }
  | { id: number; type: "score"; kind: string; query: string; options: string[] }
  | { id: number; type: "tag"; text: string };

export type Response = { id: number; result?: unknown; error?: string; progress?: string };

type Models = {
  rtok: PreTrainedTokenizer;
  retriever: PreTrainedModel;
  dtok: PreTrainedTokenizer;
  decider: PreTrainedModel;
  ttok: PreTrainedTokenizer;
  tagger: PreTrainedModel;
  temperatures: Record<string, number>;
  maxLength: number;
};

let models: Promise<Models> | null = null;
const TAGS = ["O", "B-PLACE", "I-PLACE", "B-TIME", "I-TIME", "B-CLOUD", "I-CLOUD"];

async function load(base: string): Promise<Models> {
  env.allowRemoteModels = false;
  env.allowLocalModels = true;
  env.localModelPath = base.endsWith("/") ? base : `${base}/`;
  const opts = { dtype: "q8" as const, device: "wasm" as const };
  const [rtok, retriever, dtok, decider, ttok, tagger, cfg] = await Promise.all([
    AutoTokenizer.from_pretrained("retriever"),
    AutoModel.from_pretrained("retriever", opts),
    AutoTokenizer.from_pretrained("decider"),
    AutoModelForSequenceClassification.from_pretrained("decider", opts),
    AutoTokenizer.from_pretrained("tagger"),
    AutoModelForTokenClassification.from_pretrained("tagger", opts),
    fetch(`${env.localModelPath}decider/decider.json`).then((r) => r.json()),
  ]);
  return { rtok, retriever, dtok, decider, ttok, tagger, temperatures: cfg.temperatures, maxLength: cfg.max_length };
}

async function embed(m: Models, texts: string[], maxLength: number): Promise<number[][]> {
  const out: number[][] = [];
  for (let i = 0; i < texts.length; i += 32) {
    const batch = texts.slice(i, i + 32);
    const h = (await m.retriever(m.rtok(batch, { padding: true, truncation: true, max_length: maxLength })))
      .last_hidden_state as Tensor;
    const [n, len, d] = h.dims as [number, number, number];
    const data = h.data as Float32Array;
    for (let j = 0; j < n; j++) {
      const v = Array.from(data.subarray(j * len * d, j * len * d + d));
      const norm = Math.hypot(...v) || 1;
      out.push(v.map((a) => a / norm));
    }
  }
  return out;
}

async function score(m: Models, kind: string, q: string, options: string[]): Promise<number[]> {
  const x = m.dtok(Array(options.length).fill(q), {
    text_pair: options,
    padding: true,
    truncation: true,
    max_length: m.maxLength,
  });
  const logits = Array.from(((await m.decider(x)).logits as Tensor).data as Float32Array);
  const t = m.temperatures[kind] ?? 1;
  const max = Math.max(...logits);
  const e = logits.map((v) => Math.exp((v - max) / t));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / sum);
}

/**
 * The words of each kind, with the mean probability of their tags: the first
 * run of tokens with that tag. transformers.js gives no offsets, so the
 * offsets come from finding each token's text in the request.
 */
async function tag(m: Models, text: string): Promise<Record<string, { words: string; p: number }>> {
  const enc = m.ttok(text, { truncation: true, max_length: 64 });
  const logits = (await m.tagger(enc)).logits as Tensor;
  const [, n, k] = logits.dims as [number, number, number];
  const data = logits.data as Float32Array;
  const ids = Array.from(enc.input_ids.data as BigInt64Array, Number);
  const spans: Record<string, { s: number; e: number; open: boolean; ps: number[] }> = {};
  let pos = 0;
  let prev: string | null = null;
  for (let i = 0; i < n; i++) {
    const word = m.ttok.decode([ids[i]!], { skip_special_tokens: true }).trimStart();
    if (!word) continue;
    const start = text.indexOf(word, pos);
    if (start < 0) continue;
    const end = start + word.length;
    pos = end;
    const row = Array.from(data.subarray(i * k, i * k + k));
    const max = Math.max(...row);
    const e = row.map((v) => Math.exp(v - max));
    const sum = e.reduce((a, b) => a + b, 0);
    const best = row.indexOf(max);
    const kind = best === 0 ? null : TAGS[best]!.slice(2);
    if (kind && kind === prev && spans[kind]?.open) {
      spans[kind]!.e = end;
      spans[kind]!.ps.push(e[best]! / sum);
    } else if (kind && !spans[kind]) {
      spans[kind] = { s: start, e: end, open: true, ps: [e[best]! / sum] };
    }
    if (prev && prev !== kind && spans[prev]) spans[prev]!.open = false;
    prev = kind;
  }
  return Object.fromEntries(
    Object.entries(spans).map(([kd, v]) => [
      kd,
      { words: text.slice(v.s, v.e), p: v.ps.reduce((a, b) => a + b, 0) / v.ps.length },
    ]),
  );
}

self.onmessage = async (event: MessageEvent<Request>) => {
  const msg = event.data;
  const reply = (r: Omit<Response, "id">) => (self as unknown as Worker).postMessage({ id: msg.id, ...r });
  try {
    if (msg.type === "load") {
      models ??= load(msg.base);
      models.catch(() => (models = null));
      await models;
      return reply({ result: true });
    }
    if (!models) throw new Error("The local models are not loaded");
    const m = await models;
    if (msg.type === "embed") return reply({ result: await embed(m, msg.texts, msg.maxLength) });
    if (msg.type === "score") return reply({ result: await score(m, msg.kind, msg.query, msg.options) });
    if (msg.type === "tag") return reply({ result: await tag(m, msg.text) });
  } catch (error) {
    reply({ error: String(error) });
  }
};
