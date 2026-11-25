/**
 * Can the retriever find the collection among all collections of all
 * catalogs, with no catalog question first? Then the decider picks a
 * collection, and the catalog follows from it.
 *
 * Ranks all collections (or only those of the curated catalogs) for each
 * request and reports how often an accepted (catalog, collection) is in the
 * top k. The collection embeddings are cached in training/data/.
 *
 * Usage: pnpm tsx eval/local/global.ts
 */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { AutoModel, AutoTokenizer, type Tensor } from "@huggingface/transformers";
import { CASES } from "../cases";

const RETRIEVER = "onnx-community/granite-embedding-small-english-r2-ONNX";
type Collection = { id: string; title?: string; description?: string };
const DATA = new URL("../../training/data/", import.meta.url);
const snapshot = JSON.parse(readFileSync(new URL("stac-full.json", DATA), "utf-8")) as {
  catalogs: { id: string; curated?: boolean }[];
  collections: Record<string, Collection[]>;
};
const curated = new Set(snapshot.catalogs.filter((c) => c.curated).map((c) => c.id));
const short = (t: string | undefined, n: number) => {
  const s = (t ?? "").split(/\s+/).filter(Boolean).join(" ");
  return s.length <= n ? s : `${s.slice(0, n - 1)}…`;
};
const text = (c: Collection) => `${short(c.title ?? c.id, 100)} (${c.id}): ${short(c.description, 160)}`;

const keys = Object.entries(snapshot.collections).flatMap(([cat, cols]) => cols.map((c) => ({ cat, c })));
const tok = await AutoTokenizer.from_pretrained(RETRIEVER);
const model = await AutoModel.from_pretrained(RETRIEVER, { dtype: "q8" });
const D = 384;

async function embed(texts: string[]): Promise<Float32Array> {
  const out = new Float32Array(texts.length * D);
  for (let i = 0; i < texts.length; i += 32) {
    const h = (await model(tok(texts.slice(i, i + 32), { padding: true, truncation: true, max_length: 256 }))).last_hidden_state as Tensor;
    const [n, len] = h.dims as [number, number, number];
    for (let j = 0; j < n; j++) {
      const v = (h.data as Float32Array).subarray(j * len * D, j * len * D + D);
      const norm = Math.hypot(...v) || 1;
      for (let d = 0; d < D; d++) out[(i + j) * D + d] = v[d]! / norm;
    }
  }
  return out;
}

const cache = new URL("collection-embeddings.f32", DATA);
let vecs: Float32Array;
if (existsSync(cache) && readFileSync(cache).byteLength === keys.length * D * 4) {
  vecs = new Float32Array(readFileSync(cache).buffer.slice(0));
} else {
  vecs = await embed(keys.map((k) => text(k.c)));
  writeFileSync(cache, Buffer.from(vecs.buffer));
}

type Item = { request: string; ok: (cat: string, col: string) => boolean; curatedOnly: boolean };
const recs = readFileSync(new URL("test-claude.jsonl", DATA), "utf-8").trim().split("\n").map((l) => JSON.parse(l));
const items: [string, Item[]][] = [
  ["field test (30)", CASES.filter((c) => c.collection).map((c) => ({
    request: c.prompt, ok: (cat, col) => c.catalog.includes(cat) && c.collection!.includes(col), curatedOnly: true,
  }))],
  ["test-claude-v1", recs.filter((r) => r.collection.accepted.length).map((r) => ({
    request: r.request, ok: (cat, col) => r.catalog.accepted.includes(cat) && r.collection.accepted.includes(col), curatedOnly: false,
  }))],
];

const KS = [1, 5, 16, 32, 64];
for (const [name, list] of items) {
  for (const mode of ["all catalogs", "curated catalogs"] as const) {
    const qs = await embed(list.map((x) => x.request));
    const ranks: number[] = [];
    list.forEach((x, qi) => {
      const scored = keys
        .map((k, i) => ({ k, s: (mode === "curated catalogs" && !curated.has(k.cat)) ? -9 : dot(qi, i) }))
        .sort((a, b) => b.s - a.s);
      const r = scored.findIndex(({ k }) => x.ok(k.cat, k.c.id));
      ranks.push(r < 0 ? Infinity : r + 1);
    });
    function dot(qi: number, i: number) {
      let s = 0;
      for (let d = 0; d < D; d++) s += qs[qi * D + d]! * vecs[i * D + d]!;
      return s;
    }
    if (mode === "curated catalogs" && !list[0]!.curatedOnly) continue;
    console.log(`${name}, ${mode}: ` + KS.map((k) => `@${k} ${ranks.filter((r) => r <= k).length}/${ranks.length}`).join("  "));
  }
}
