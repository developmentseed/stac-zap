/**
 * Phase 0: can a small encoder shortlist the collections, so the decider
 * only scores the top k? For each field-test prompt, this ranks all
 * collections of the expected catalog (and all catalogs) and reports where
 * the first accepted answer lands, for the keyword score of the app
 * (`shortlist` in `src/zap/catalogs.ts`) and for bi-encoders with no
 * fine-tuning.
 *
 * Usage: pnpm tsx eval/local/recall.ts [bi:<repo> ...]
 */
import { readFileSync } from "node:fs";
import { AutoModel, AutoTokenizer, type Tensor } from "@huggingface/transformers";
import type { Catalog, Collection } from "../../src/zap/catalogs";
import { STOPWORDS } from "../../src/zap/geocode";
import { CASES } from "../cases";
import { optionText } from "./decide";

type Snapshot = { catalogs: Catalog[]; collections: Record<string, Collection[]> };
const snapshot: Snapshot = JSON.parse(
  readFileSync(new URL("../data/stac-snapshot.json", import.meta.url), "utf-8"),
);
const KS = [1, 5, 16, 32];

const collectionText = (c: Collection) =>
  optionText(c.id, `${c.title ?? c.id}: ${(c.description ?? "").replace(/\s+/g, " ").slice(0, 400)}`);

/** The keyword score of `shortlist`, as a ranking. Ties keep the catalog order. */
function keywordRank(prompt: string, texts: string[]): number[] {
  const stems = prompt
    .toLowerCase()
    .split(/[^\p{L}\p{N}]+/u)
    .filter((w) => w.length >= 3 && !STOPWORDS.has(w))
    .map((w) => w.slice(0, 6));
  return texts.map((t) => stems.filter((s) => t.toLowerCase().includes(s)).length);
}

type Ranker = (query: string, texts: string[]) => Promise<number[]>;

async function biRanker(repo: string): Promise<Ranker> {
  const tok = await AutoTokenizer.from_pretrained(repo);
  const model = await AutoModel.from_pretrained(repo, { dtype: "q8" });
  const cache = new Map<string, Float32Array>();
  const embed = async (texts: string[]) => {
    const todo = [...new Set(texts)].filter((t) => !cache.has(t));
    for (let i = 0; i < todo.length; i += 32) {
      const batch = todo.slice(i, i + 32);
      const h = (await model(tok(batch, { padding: true, truncation: true, max_length: 256 }))).last_hidden_state as Tensor;
      const [n, t, d] = h.dims as [number, number, number];
      for (let j = 0; j < n; j++) {
        const v = (h.data as Float32Array).slice(j * t * d, j * t * d + d);
        const norm = Math.hypot(...v) || 1;
        cache.set(batch[j]!, v.map((a) => a / norm));
      }
    }
  };
  return async (query, texts) => {
    await embed([query, ...texts]);
    const q = cache.get(query)!;
    return texts.map((t) => cache.get(t)!.reduce((s, v, i) => s + v * q[i]!, 0));
  };
}

/** The 1-based rank of the best accepted id; ties count against it. */
function rankOf(ids: string[], scores: number[], accepted: string[]): number {
  const best = Math.max(...ids.map((id, i) => (accepted.includes(id) ? scores[i]! : -Infinity)));
  if (best === -Infinity) return Infinity;
  return scores.filter((s) => s >= best).length;
}

function report(name: string, ranks: number[]) {
  const n = ranks.length;
  const cells = KS.map((k) => `@${k} ${ranks.filter((r) => r <= k).length}/${n}`);
  const sorted = [...ranks].sort((a, b) => a - b);
  console.log(`  ${name.padEnd(62)} ${cells.join("  ")}  worst ${sorted.at(-1)}`);
}

const rankers: [string, Ranker][] = [["keyword (app shortlist)", async (q, t) => keywordRank(q, t)]];
for (const spec of process.argv.slice(2)) rankers.push([spec, await biRanker(spec.replace(/^bi:/, ""))]);

const catalogIds = snapshot.catalogs.map((c) => c.id);
const catalogTexts = snapshot.catalogs.map((c) => optionText(c.id, c.description));
console.log(`Catalog (${catalogIds.length} options), ${CASES.length} prompts:`);
for (const [name, rank] of rankers) {
  const ranks: number[] = [];
  for (const c of CASES) ranks.push(rankOf(catalogIds, await rank(c.prompt, catalogTexts), c.catalog));
  report(name, ranks);
}

const withCollection = CASES.filter((c) => c.collection);
console.log(`Collection (all collections of the catalog, up to ${Math.max(...Object.values(snapshot.collections).map((x) => x.length))}), ${withCollection.length} prompts:`);
for (const [name, rank] of rankers) {
  const ranks: number[] = [];
  for (const c of withCollection) {
    const all = snapshot.collections[c.cat ?? c.catalog[0]!] ?? [];
    ranks.push(rankOf(all.map((x) => x.id), await rank(c.prompt, all.map(collectionText)), c.collection!));
  }
  report(name, ranks);
}
