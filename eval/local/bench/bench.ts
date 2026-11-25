/**
 * Phase 0 browser bench: how long the encoders take in a browser tab (ONNX
 * Runtime Web, WASM, int8). It times the work of one zap:
 * - one query embedding (the retriever),
 * - 16 (request, option) pairs (the decider on a shortlist),
 * - 250 texts (embedding the collections of a STAC Index catalog at load,
 *   or the decider with no shortlist).
 * URL parameters: model=<repo>, threads=<n>.
 */
import { AutoModel, AutoTokenizer, env } from "@huggingface/transformers";
import snapshot from "../../data/stac-snapshot.json";

const params = new URLSearchParams(location.search);
const repo = params.get("model") ?? "onnx-community/granite-embedding-30m-english-ONNX";
const threads = Number(params.get("threads") ?? 1);
const maxLength = Number(params.get("maxlen") ?? 128);
env.backends.onnx.wasm!.numThreads = threads;
env.useBrowserCache = false;

const out = document.getElementById("out")!;
const query = "Sentinel-2 over Lisbon last summer with few clouds";
type C = { id: string; title?: string; description?: string };
const texts = Object.values(snapshot.collections as Record<string, C[]>)
  .flat()
  .map((c) => `${c.id}: ${c.title ?? ""}: ${(c.description ?? "").replace(/\s+/g, " ").slice(0, 400)}`)
  .slice(0, 250);

const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)]!;
async function time(f: () => Promise<unknown>, n: number) {
  const ms: number[] = [];
  for (let i = 0; i < n; i++) {
    const t = performance.now();
    await f();
    ms.push(performance.now() - t);
  }
  return Math.round(median(ms));
}

const result: Record<string, unknown> = { repo, threads, maxLength, isolated: crossOriginIsolated, cores: navigator.hardwareConcurrency };
try {
  let t = performance.now();
  const tok = await AutoTokenizer.from_pretrained(repo);
  const model = await AutoModel.from_pretrained(repo, { dtype: "q8", device: "wasm" });
  result.load_ms = Math.round(performance.now() - t);
  const run = async (a: string[], b?: string[]) =>
    model(tok(a, { ...(b ? { text_pair: b } : {}), padding: true, truncation: true, max_length: maxLength }));
  await run([query]); // warm up
  result.query_ms = await time(() => run([query]), 5);
  const pairs = texts.slice(0, 16);
  result.tokens_per_pair = tok([query], { text_pair: [pairs[0]!], truncation: true, max_length: maxLength }).input_ids.dims[1];
  result.pairs16_ms = await time(() => run(Array(16).fill(query), pairs), 3);
  t = performance.now();
  for (let i = 0; i < texts.length; i += 32) await run(texts.slice(i, i + 32));
  result.texts250_ms = Math.round(performance.now() - t);
} catch (e) {
  result.error = String(e);
}
out.textContent = JSON.stringify(result, null, 1);
(window as any).__result = result;
