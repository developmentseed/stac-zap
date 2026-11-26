/**
 * End-to-end test of the trained local models, as the page will run them:
 * 1. catalog: the decider scores the catalogs,
 * 2. collection: the retriever shortlists 16 collections of the catalog, and
 *    the decider picks one,
 * 3. place, time and cloud cover: the tagger marks the words; code turns the
 *    time words into a range (chrono-node and the rules of the app) and the
 *    cloud words into a number.
 *
 * The text formats are the ones of `training/train_hf_jobs.py`.
 *
 * Sets: the field test, the unparsed and dev requests of the extraction test
 * (`cases.ts`, today 2026-10-02), and test-claude-v1 (today 2026-11-25).
 *
 * Usage: pnpm tsx eval/local/pipeline.ts <model dir with decider/ and tagger/>
 * The collection step reads training/data/stac-full.json (`training/snapshot.ts`).
 */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  AutoModel,
  AutoModelForSequenceClassification,
  AutoModelForTokenClassification,
  AutoTokenizer,
  env,
  type Tensor,
} from "@huggingface/transformers";
import { rangeOfAnswers } from "../../src/zap/extract";
import { CASES, DEV_CASES, UNPARSED_CASES } from "../cases";
import { TODAY } from "../questions";
import { cloudInRange, cloudOk, cloudsOf, placeOf, placeOk, rangeOk, rangesOf, type Range } from "../score";
import { readCloud, readTime } from "./extract";
import { docText } from "../../src/zap/local/text";

const dir = resolve(process.argv[2] ?? `${process.env.HOME}/.cache/stac-zap-models/v1`);
/** v6 and later: one encoder with three outputs, in <dir>/model. */
const SHARED = existsSync(`${dir}/model`);
/** v3 and later: a trained retriever in the model folder, and "catalog follows collection". */
const GLOBAL = SHARED || existsSync(`${dir}/retriever`);
const RETRIEVER = GLOBAL ? "retriever" : "onnx-community/granite-embedding-small-english-r2-ONNX";
/** "q8" (the int8 files the page loads) or "fp32". */
const DTYPE = (process.env.DTYPE ?? "q8") as "q8" | "fp32";
const cfg = (SHARED
  ? ((c) => ({ ...c, max_length: c.decider_max_length }))(JSON.parse(readFileSync(`${dir}/stac-zap.json`, "utf-8")))
  : JSON.parse(readFileSync(`${dir}/decider/decider.json`, "utf-8"))) as {
  temperatures: Record<string, number>;
  shortlist: number;
  max_length: number;
};

type Catalog = { id: string; description: string; curated?: boolean };
type Collection = { id: string; title?: string; description?: string };
const snapshot = JSON.parse(readFileSync(new URL("../../training/data/stac-full.json", import.meta.url), "utf-8")) as {
  catalogs: Catalog[];
  collections: Record<string, Collection[]>;
};
const catalogs = new Map(snapshot.catalogs.map((c) => [c.id, c]));
const curated = snapshot.catalogs.filter((c) => c.curated).map((c) => c.id);

// ---------------------------------------------------------------- text formats (as in training)

const short = (text: string | undefined, n: number) => {
  const t = (text ?? "").split(/\s+/).filter(Boolean).join(" ");
  return t.length <= n ? t : `${t.slice(0, n - 1)}…`;
};
const catalogText = (id: string) => `${id}: ${short(catalogs.get(id)!.description, 300)}`;
const collectionText = (c: Collection) => `${short(c.title ?? c.id, 100)} (${c.id}): ${short(c.description, 160)}`;
const query = (question: string, request: string, current: string | null = null) =>
  `${question} | request: ${request} | current: ${current ?? "not set"}`;

// ---------------------------------------------------------------- models

env.localModelPath = `${dir}/`;
// The shared model answers all three calls; each view returns the output its caller reads.
type Call = (x: unknown) => Promise<Record<string, unknown>>;
async function sharedModels(): Promise<[any, Call, any, Call, any, Call]> {
  const [tok, m] = await Promise.all([
    AutoTokenizer.from_pretrained("model", { local_files_only: true }),
    AutoModel.from_pretrained("model", { dtype: DTYPE, local_files_only: true }),
  ]);
  const view = (name: string, as: string): Call => async (x) => ({ [as]: (await m(x as never))[name] });
  return [tok, view("decider_logits", "logits"), tok, view("tag_logits", "logits"), tok, view("last_hidden_state", "last_hidden_state")];
}
const [dtok, decider, ttok, tagger, rtok, retriever] = SHARED ? await sharedModels() : await Promise.all([
  AutoTokenizer.from_pretrained("decider", { local_files_only: true }),
  AutoModelForSequenceClassification.from_pretrained("decider", { dtype: DTYPE, local_files_only: true }),
  AutoTokenizer.from_pretrained("tagger", { local_files_only: true }),
  AutoModelForTokenClassification.from_pretrained("tagger", { dtype: DTYPE, local_files_only: true }),
  AutoTokenizer.from_pretrained(RETRIEVER, { local_files_only: GLOBAL }),
  AutoModel.from_pretrained(RETRIEVER, { dtype: "q8", local_files_only: GLOBAL }),
]);

/** Probabilities of the options of one question. */
async function decide(kind: string, q: string, texts: string[]): Promise<number[]> {
  const x = dtok(Array(texts.length).fill(q), { text_pair: texts, padding: true, truncation: true, max_length: cfg.max_length });
  const logits = Array.from(((await decider(x)).logits as Tensor).data as Float32Array);
  const t = cfg.temperatures[kind] ?? 1;
  const m = Math.max(...logits);
  const e = logits.map((v) => Math.exp((v - m) / t));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

const vecCache = new Map<string, Float32Array>();
async function embed(texts: string[]): Promise<Float32Array[]> {
  const todo = [...new Set(texts)].filter((t) => !vecCache.has(t));
  for (let i = 0; i < todo.length; i += 32) {
    const batch = todo.slice(i, i + 32);
    const h = (await retriever(rtok(batch, { padding: true, truncation: true, max_length: GLOBAL ? 128 : 256 }))).last_hidden_state as Tensor;
    const [n, len, d] = h.dims as [number, number, number];
    for (let j = 0; j < n; j++) {
      const v = (h.data as Float32Array).slice(j * len * d, j * len * d + d);
      const norm = Math.hypot(...v) || 1;
      vecCache.set(batch[j]!, v.map((a) => a / norm));
    }
  }
  return texts.map((t) => vecCache.get(t)!);
}

async function shortlist(request: string, catalog: string): Promise<Collection[]> {
  const cols = snapshot.collections[catalog] ?? [];
  const [q, ...vs] = await embed([request, ...cols.map(collectionText)]);
  const sims = vs.map((v) => v.reduce((s, a, i) => s + a * q![i]!, 0));
  return cols.map((c, i) => ({ c, s: sims[i]! })).sort((a, b) => b.s - a.s).slice(0, cfg.shortlist).map((x) => x.c);
}

const TAGS = ["O", "B-PLACE", "I-PLACE", "B-TIME", "I-TIME", "B-CLOUD", "I-CLOUD"];

/** The words of each kind: the first run of tokens with that tag. */
async function tag(text: string): Promise<Record<string, string>> {
  // transformers.js has no offset mapping: rebuild offsets by finding each token's text in the request.
  const ids = ttok(text, { truncation: true, max_length: 64 }).input_ids as Tensor;
  const logits = (await tagger(ttok(text, { truncation: true, max_length: 64 }))).logits as Tensor;
  const [, n, k] = logits.dims as [number, number, number];
  const data = logits.data as Float32Array;
  const tokenIds = Array.from(ids.data as BigInt64Array, Number);
  const spans: Record<string, [number, number, boolean]> = {};
  let pos = 0;
  let prev: string | null = null;
  for (let i = 0; i < n; i++) {
    const piece = ttok.decode([tokenIds[i]!], { skip_special_tokens: true });
    if (!piece) continue;
    const word = piece.trimStart();
    const start = text.indexOf(word, pos);
    if (start < 0 || !word) continue;
    const end = start + word.length;
    pos = end;
    let best = 0;
    for (let j = 1; j < k; j++) if (data[i * k + j]! > data[i * k + best]!) best = j;
    const kind = best === 0 ? null : TAGS[best]!.slice(2);
    if (kind && kind === prev && spans[kind]?.[2]) spans[kind][1] = end;
    else if (kind && !spans[kind]) spans[kind] = [start, end, true];
    if (prev && prev !== kind && spans[prev]) spans[prev]![2] = false;
    prev = kind;
  }
  return Object.fromEntries(Object.entries(spans).map(([kd, [s, e]]) => [kd, text.slice(s, e)]));
}

/** Place, range and cloud cover of a request, as the app will read them. */
async function extractValues(request: string, today: Date) {
  const words = await tag(request);
  const time = words.TIME ? readTime(words.TIME, today) : { when: null, start: null, end: null };
  const range = words.TIME
    ? (rangeOfAnswers({ when: { value: time.when ?? words.TIME }, start: { value: time.start }, end: { value: time.end } }, today) ?? null)
    : null;
  const cloud = words.CLOUD ? readCloud(/cloud|clear|sun/i.test(words.CLOUD) ? words.CLOUD : `${words.CLOUD} clouds`) : null;
  return { place: words.PLACE ?? null, range: range as Range | null, cloud, words };
}

async function pick(kind: string, q: string, options: [string, string][]): Promise<[string, number]> {
  const p = await decide(kind, q, options.map(([, t]) => t));
  const i = p.indexOf(Math.max(...p));
  return [options[i]![0], p[i]!];
}

// ---------------------------------------------------------------- tests

const results: Record<string, unknown> = { model: dir };
const pct = (ok: number, n: number) => `${ok}/${n}`;
const t0 = performance.now();

/** v3: the shortlist of all collections of the pool, and the decider picks one; the catalog follows. */
async function pickGlobal(request: string, curatedOnly: boolean, ok: (key: string) => boolean) {
  const pool = snapshot.catalogs.filter((c) => !curatedOnly || c.curated).flatMap((c) => (snapshot.collections[c.id] ?? []).map((x) => ({ cat: c.id, c: x })));
  const texts = pool.map(({ cat, c }) => docText(cat, c));
  const [q, ...vs] = await embed([request, ...texts]);
  const top = pool.map((x, i) => ({ x, t: texts[i]!, s: vs[i]!.reduce((a, v, j) => a + v * q![j]!, 0) })).sort((a, b) => b.s - a.s).slice(0, cfg.shortlist);
  const [key] = await pick("collection", query("Which collection?", request), top.map(({ x, t }) => [`${x.cat}/${x.c.id}`, t]));
  return { key, onShortlist: top.some(({ x }) => ok(`${x.cat}/${x.c.id}`)) };
}

if (GLOBAL) {
  let cat = 0, col = 0, ncol = 0, reach = 0;
  const rows = [];
  for (const c of CASES) {
    const ok = (k: string) => c.catalog.includes(k.split("/")[0]!) && (!c.collection || c.collection.includes(k.slice(k.indexOf("/") + 1)));
    const { key, onShortlist } = await pickGlobal(c.prompt, true, ok);
    cat += Number(c.catalog.includes(key.split("/")[0]!));
    if (c.collection) {
      ncol++;
      col += Number(ok(key));
      reach += Number(onShortlist);
    }
    rows.push({ prompt: c.prompt, catalog: key.split("/")[0], collection: key.slice(key.indexOf("/") + 1) });
  }
  results.field_decisions = { catalog: pct(cat, CASES.length), collection: pct(col, ncol), collection_on_shortlist: pct(reach, ncol), rows };
  console.log(`field test (catalog follows collection): catalog ${pct(cat, CASES.length)}, catalog and collection ${pct(col, ncol)} (on shortlist ${pct(reach, ncol)})`);
}

// Field test: catalog (curated list) and collection (shortlist of the expected catalog).
if (!GLOBAL) {
  let cat = 0, col = 0, ncol = 0, reach = 0;
  const rows = [];
  for (const c of CASES) {
    const [gotCat] = await pick("catalog", query("Which STAC catalog?", c.prompt), curated.map((id) => [id, catalogText(id)]));
    cat += Number(c.catalog.includes(gotCat));
    let gotCol: string | undefined;
    if (c.collection) {
      const list = await shortlist(c.prompt, c.cat ?? c.catalog[0]!);
      reach += Number(list.some((x) => c.collection!.includes(x.id)));
      [gotCol] = await pick("collection", query("Which collection?", c.prompt), list.map((x) => [x.id, collectionText(x)]));
      ncol++;
      col += Number(c.collection.includes(gotCol));
    }
    rows.push({ prompt: c.prompt, catalog: gotCat, collection: gotCol });
  }
  results.field_decisions = { catalog: pct(cat, CASES.length), collection: pct(col, ncol), collection_on_shortlist: pct(reach, ncol), rows };
  console.log(`field test: catalog ${pct(cat, CASES.length)}, collection ${pct(col, ncol)} (on shortlist ${pct(reach, ncol)})`);
}

// Extraction, field test: place, time, cloud cover.
{
  let place = 0, time = 0, nt = 0, cloud = 0, nc = 0;
  const rows = [];
  for (const c of CASES) {
    const v = await extractValues(c.prompt, TODAY);
    if (c.place) place += Number(placeOk(v.place, c.place));
    const r = rangesOf(c);
    if (r) {
      nt++;
      time += Number(r.ranges.some((e) => rangeOk(v.range, e, r.relative)));
    }
    const cl = cloudsOf(c);
    if (cl !== undefined) {
      nc++;
      cloud += Number(cloudInRange(v.cloud, cl));
    }
    rows.push({ prompt: c.prompt, ...v, place_as: placeOf(v.place) });
  }
  results.field_extraction = { place: pct(place, CASES.filter((c) => c.place).length), time: pct(time, nt), cloud: pct(cloud, nc), rows };
  console.log(`field test extraction: place ${pct(place, CASES.filter((c) => c.place).length)}, time ${pct(time, nt)}, cloud ${pct(cloud, nc)}`);
}

// Extraction, unparsed and dev requests.
for (const [name, cases] of [["unparsed", UNPARSED_CASES], ["dev", DEV_CASES]] as const) {
  let ok = 0;
  const rows = [];
  for (const c of cases) {
    const v = await extractValues(c.prompt, TODAY);
    const good =
      (c.range === undefined || rangeOk(v.range, c.range as Range, Boolean(c.relative))) &&
      (c.cloud === undefined || cloudOk(v.cloud, c.cloud)) &&
      (c.place === undefined || placeOk(v.place, [c.place]));
    ok += Number(good);
    rows.push({ id: c.id, prompt: c.prompt, ok: good, ...v });
  }
  results[name] = { ok: pct(ok, cases.length), rows };
  console.log(`${name}: ${pct(ok, cases.length)}`);
}

// test-claude-v1, v3: one shortlist of all catalogs (curated only when every correct catalog is curated).
if (GLOBAL) {
  type Rec = { request: string; catalog: { accepted: string[] }; collection: { accepted: string[] } };
  const recs: Rec[] = readFileSync(new URL("../../training/data/test-claude.jsonl", import.meta.url), "utf-8").trim().split("\n").map((l) => JSON.parse(l));
  let ok = 0, catOk = 0, reach = 0, n = 0;
  const curatedIds = new Set(curated);
  const rows = [];
  for (const r of recs) {
    if (!r.collection.accepted.length) continue;
    n++;
    const good = (k: string) => r.catalog.accepted.includes(k.split("/")[0]!) && r.collection.accepted.includes(k.slice(k.indexOf("/") + 1));
    const curatedOnly = r.catalog.accepted.every((c) => curatedIds.has(c));
    const res = await pickGlobal(r.request, curatedOnly, good);
    ok += Number(good(res.key));
    catOk += Number(r.catalog.accepted.includes(res.key.split("/")[0]!));
    reach += Number(res.onShortlist);
    rows.push({ request: r.request, pick: res.key, ok: good(res.key) });
  }
  results.test_claude_global = { catalog: pct(catOk, n), catalog_and_collection: pct(ok, n), on_shortlist: pct(reach, n), rows };
  console.log(`test-claude-v1 (catalog follows collection): catalog ${pct(catOk, n)}, catalog and collection ${pct(ok, n)} (on shortlist ${pct(reach, n)})`);
}

// test-claude-v1: all catalogs, the shortlist of the expected catalog, and the values.
{
  type Rec = {
    request: string; today: string;
    catalog: { accepted: string[] }; collection: { target: string | null; accepted: string[] };
    place: { words: string | null; map_view: boolean }; time: { range: Range | null }; cloud: { value: number | null };
  };
  const recs: Rec[] = readFileSync(new URL("../../training/data/test-claude.jsonl", import.meta.url), "utf-8").trim().split("\n").map((l) => JSON.parse(l));
  let cat = 0, catCur = 0, nCur = 0, col = 0, ncol = 0, reach = 0, place = 0, time = 0, nt = 0, cloud = 0;
  const rows = [];
  for (const r of recs) {
    const today = new Date(`${r.today}T12:00:00Z`);
    const all = snapshot.catalogs.map((c) => c.id);
    const [gotCat] = await pick("catalog", query("Which STAC catalog?", r.request), all.map((id) => [id, catalogText(id)]));
    cat += Number(r.catalog.accepted.includes(gotCat));
    if (r.catalog.accepted.some((c) => curated.includes(c))) {
      nCur++;
      const [g] = await pick("catalog", query("Which STAC catalog?", r.request), curated.map((id) => [id, catalogText(id)]));
      catCur += Number(r.catalog.accepted.includes(g));
    }
    let gotCol: string | undefined;
    const target = r.catalog.accepted[0]!;
    if (r.collection.accepted.length && snapshot.collections[target]) {
      const list = await shortlist(r.request, target);
      reach += Number(list.some((x) => r.collection.accepted.includes(x.id)));
      [gotCol] = await pick("collection", query("Which collection?", r.request), list.map((x) => [x.id, collectionText(x)]));
      ncol++;
      col += Number(r.collection.accepted.includes(gotCol));
    }
    const v = await extractValues(r.request, today);
    const expPlace = r.place.map_view ? ["map-view"] : r.place.words ? [r.place.words] : ["no-place"];
    place += Number(placeOk(v.place, expPlace));
    if (r.time.range) nt++;
    const relative = Boolean(r.time.range && r.time.range[1] === r.today);
    time += Number(r.time.range ? rangeOk(v.range, r.time.range, relative) : v.range === null);
    cloud += Number(cloudOk(v.cloud, r.cloud.value));
    rows.push({ request: r.request, catalog: gotCat, collection: gotCol, ...v });
  }
  const n = recs.length;
  results.test_claude = {
    catalog_all: pct(cat, n), catalog_curated: pct(catCur, nCur), collection: pct(col, ncol), collection_on_shortlist: pct(reach, ncol),
    place: pct(place, n), time: pct(time, n), time_with_a_time: nt, cloud: pct(cloud, n), rows,
  };
  console.log(
    `test-claude-v1: catalog ${pct(cat, n)} (all ${snapshot.catalogs.length}), catalog ${pct(catCur, nCur)} (curated list), ` +
      `collection ${pct(col, ncol)} (on shortlist ${pct(reach, ncol)}), place ${pct(place, n)}, time ${pct(time, n)}, cloud ${pct(cloud, n)}`,
  );
}

results.seconds = Math.round((performance.now() - t0) / 1000);
const out = `${dir}/pipeline-results-${DTYPE}.json`;
writeFileSync(out, JSON.stringify(results, null, 1));
console.log(`→ ${out} (${results.seconds} s)`);
