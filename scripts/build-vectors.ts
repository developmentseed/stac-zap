/**
 * Embeds the collections of the curated catalogs (and of the STAC Index
 * catalogs, with --all) with the local model (its retriever output) and writes
 * `<model dir>/vectors.json`. The page loads it, so it only embeds the
 * request of each zap. Collections that are not in the file are embedded in
 * the page when a zap first needs them.
 *
 * The collections come from the live catalogs, as the app lists them.
 * Vectors are stored as int8 (one scale for all), base64.
 *
 * Usage: pnpm tsx scripts/build-vectors.ts public/models/stac-zap-v3 [--all]
 */
import { writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { AutoModel, AutoTokenizer, env, type Tensor } from "@huggingface/transformers";
import { listCatalogs, listCollections } from "../src/zap/catalogs";
import { docText } from "../src/zap/local/text";

const dir = resolve(process.argv[2] ?? "public/models/stac-zap-v3");
const all = process.argv.includes("--all");
env.localModelPath = `${dir}/`;
env.allowRemoteModels = false;
const tok = await AutoTokenizer.from_pretrained("model");
const model = await AutoModel.from_pretrained("model", { dtype: "q8" });

const catalogs = await listCatalogs(all);
const docs: { key: string; text: string }[] = [];
for (const catalog of catalogs) {
  try {
    const cols = await listCollections(catalog.href);
    for (const c of cols) docs.push({ key: `${catalog.id}/${c.id}`, text: docText(catalog.id, c) });
    console.log(catalog.id, cols.length);
  } catch (error) {
    console.error(`${catalog.id}: ${error}`);
  }
}

const DIM = 384;
const SCALE = 127;
const ints = new Int8Array(docs.length * DIM);
for (let i = 0; i < docs.length; i += 32) {
  const batch = docs.slice(i, i + 32).map((d) => d.text);
  const h = (await model(tok(batch, { padding: true, truncation: true, max_length: 128 }))).last_hidden_state as Tensor;
  const [n, len] = h.dims as [number, number, number];
  for (let j = 0; j < n; j++) {
    const v = (h.data as Float32Array).subarray(j * len * DIM, j * len * DIM + DIM);
    const norm = Math.hypot(...v) || 1;
    for (let d = 0; d < DIM; d++) ints[(i + j) * DIM + d] = Math.max(-127, Math.min(127, Math.round((v[d]! / norm) * SCALE)));
  }
}
writeFileSync(
  `${dir}/vectors.json`,
  JSON.stringify({ keys: docs.map((d) => d.key), dim: DIM, scale: SCALE, data: Buffer.from(ints.buffer).toString("base64") }),
);
console.log(`${docs.length} collections → ${dir}/vectors.json`);
