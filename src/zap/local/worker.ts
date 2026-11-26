/**
 * Runs the local model in a Web Worker, so the page stays responsive. One
 * encoder with three outputs (`training/train_hf_jobs.py`):
 * - `last_hidden_state`: the retriever embeds a text as its CLS vector,
 *   normalized,
 * - `decider_logits`: one score per (query, option) pair; a softmax with the
 *   calibrated temperature of the question gives the probabilities,
 * - `tag_logits`: the place, time and cloud cover tags of each token.
 *
 * The files come from `base` (see `index.ts`): `model/` in the layout that
 * transformers.js reads (int4 embeddings and int8 weights, "q4", computed in fp32, fetched as gzip) and
 * `stac-zap.json` with the settings.
 */
import {
  AutoModel,
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
  tok: PreTrainedTokenizer;
  model: PreTrainedModel;
  temperatures: Record<string, number>;
  maxLength: number;
  tags: string[];
};

let models: Promise<Models> | null = null;

/** The model file: int4 embedding table, int8 weights (`training/tools/weight_int4.py`), 22.5 MB. */
const DTYPE = "q4" as const;

/**
 * Fetches an ONNX file as its gzip copy (`model_q4.onnx.gz`, 19.3 MB instead of 22.5 MB) when the host has one,
 * and unzips it in the browser. The Hugging Face Hub sends model files without compression, so the page does it.
 * Range requests (transformers.js asks for the size first) and hosts without the copy get the plain file.
 */
type Fetch = typeof env.fetch;

function gzipFetch(plain: Fetch): Fetch {
  return async (input, init) => {
    const url = typeof input === "string" ? input : input.href;
    const range = new Headers(init?.headers).has("Range");
    if (/\.onnx(\?|$)/.test(url) && !range && typeof DecompressionStream !== "undefined") {
      const gz = await plain(url.replace(/\.onnx(\?|$)/, ".onnx.gz$1"), init).catch(() => null);
      if (gz?.ok && gz.body) {
        // A server that sets Content-Encoding: gzip has already unzipped it.
        if (/gzip/.test(gz.headers.get("content-encoding") ?? "")) return gz;
        return new Response(gz.body.pipeThrough(new DecompressionStream("gzip")), {
          status: 200,
          headers: { "content-type": "application/octet-stream" },
        });
      }
    }
    return plain(input, init);
  };
}

async function load(base: string): Promise<Models> {
  base = base.endsWith("/") ? base : `${base}/`;
  if (/^https?:\/\//.test(base)) {
    // A model on another host, e.g. the Hugging Face Hub: transformers.js reads remote files from
    // remoteHost + remotePathTemplate, with {model} replaced by the name ("model" below). It also keeps them in the
    // browser cache.
    const url = new URL(base);
    env.allowLocalModels = false;
    env.allowRemoteModels = true;
    env.remoteHost = `${url.origin}/`;
    env.remotePathTemplate = `${url.pathname.replace(/^\//, "")}{model}/`;
  } else {
    env.allowRemoteModels = false;
    env.allowLocalModels = true;
    env.localModelPath = base;
  }
  if (!(env.fetch as { gzip?: boolean }).gzip) {
    env.fetch = Object.assign(gzipFetch(env.fetch), { gzip: true });
  }
  const [tok, model, cfg] = await Promise.all([
    AutoTokenizer.from_pretrained("model"),
    AutoModel.from_pretrained("model", { dtype: DTYPE, device: "wasm" }),
    fetch(new URL("stac-zap.json", new URL(base, self.location.href))).then((r) => r.json()),
  ]);
  return { tok, model, temperatures: cfg.temperatures, maxLength: cfg.decider_max_length, tags: cfg.tags };
}

async function embed(m: Models, texts: string[], maxLength: number): Promise<number[][]> {
  const out: number[][] = [];
  for (let i = 0; i < texts.length; i += 32) {
    const batch = texts.slice(i, i + 32);
    const h = (await m.model(m.tok(batch, { padding: true, truncation: true, max_length: maxLength })))
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
  const x = m.tok(Array(options.length).fill(q), {
    text_pair: options,
    padding: true,
    truncation: true,
    max_length: m.maxLength,
  });
  const logits = Array.from(((await m.model(x)).decider_logits as Tensor).data as Float32Array);
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
  const enc = m.tok(text, { truncation: true, max_length: 64 });
  const logits = (await m.model(enc)).tag_logits as Tensor;
  const [, n, k] = logits.dims as [number, number, number];
  const data = logits.data as Float32Array;
  const ids = Array.from(enc.input_ids.data as BigInt64Array, Number);
  const spans: Record<string, { s: number; e: number; open: boolean; ps: number[] }> = {};
  let pos = 0;
  let prev: string | null = null;
  for (let i = 0; i < n; i++) {
    const word = m.tok.decode([ids[i]!], { skip_special_tokens: true }).trimStart();
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
    const kind = best === 0 ? null : m.tags[best]!.slice(2);
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
