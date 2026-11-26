/**
 * The local models of stac-zap: open models that run in the browser and
 * replace Jev and Clairic. No key, no server.
 *
 * - Collection, and the catalog with it: the retriever shortlists the
 *   collections of all known catalogs for the request, and the decider picks
 *   one. The catalog follows from the collection.
 * - Place pick: the decider picks one of the places the geocoder finds.
 * - Place, time and cloud cover: the tagger marks the words; the rules of
 *   `rules.ts` make the values.
 *
 * The models and the vectors of the known collections load from
 * `VITE_LOCAL_MODEL_URL` (default `models/stac-zap-v3/` of the site; see
 * `scripts/fetch-local-models.ts`). Collections that are not in the vectors
 * file are embedded in the page, once.
 */
import type { Catalog, Collection } from "../catalogs";
import type { DayRange, ExtractAnswer, Extraction } from "../extract";
import { isMapView, rangeOfAnswers } from "../extract";
import type { Place } from "../geocode";
import { readCloud, readTime } from "./rules";
import { docText, NONE_PLACE, placeText, query } from "./text";
import type { Request, Response } from "./worker";

export const BASE: string =
  import.meta.env.VITE_LOCAL_MODEL_URL || `${import.meta.env.BASE_URL}models/stac-zap-v3/`;
/** The decider scores this many collections. */
const SHORTLIST = 16;
const QUERY_MAX = 64;
const DOC_MAX = 128;

let worker: Worker | null = null;
let nextId = 0;
const pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>();

/** A request without its id, for each request type. */
type Message = Request extends infer R ? (R extends { id: number } ? Omit<R, "id"> : never) : never;

function call<T>(msg: Message): Promise<T> {
  if (!worker) {
    worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    worker.onmessage = (event: MessageEvent<Response>) => {
      const p = pending.get(event.data.id);
      if (!p) return;
      pending.delete(event.data.id);
      if (event.data.error) p.reject(new Error(event.data.error));
      else p.resolve(event.data.result);
    };
  }
  const id = nextId++;
  return new Promise<T>((resolve, reject) => {
    pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
    worker!.postMessage({ ...msg, id });
  });
}

// ---------------------------------------------------------------- loading

type VectorFile = { keys: string[]; dim: number; scale: number; data: string };
/** "catalog/collection" to its vector. */
const vectors = new Map<string, Float32Array>();
let loading: Promise<void> | null = null;

/** Loads the models and the vectors of the known collections. Call it early: the first load takes a while. */
export function loadLocal(): Promise<void> {
  loading ??= (async () => {
    const [, file] = await Promise.all([
      call({ type: "load", base: BASE }),
      fetch(`${BASE}vectors.json`).then((r) => (r.ok ? (r.json() as Promise<VectorFile>) : null)).catch(() => null),
    ]);
    if (file) {
      const bytes = Uint8Array.from(atob(file.data), (c) => c.charCodeAt(0));
      const ints = new Int8Array(bytes.buffer);
      file.keys.forEach((key, i) => {
        const v = new Float32Array(file.dim);
        for (let d = 0; d < file.dim; d++) v[d] = ints[i * file.dim + d]! / file.scale;
        vectors.set(key, v);
      });
    }
  })();
  loading.catch(() => (loading = null));
  return loading;
}

async function embed(texts: string[], maxLength: number): Promise<Float32Array[]> {
  const out = await call<number[][]>({ type: "embed", texts, maxLength });
  return out.map((v) => Float32Array.from(v));
}

/** The vectors of collections; the ones not in the file are embedded now and kept. */
async function vectorsOf(docs: { key: string; text: string }[]): Promise<Float32Array[]> {
  const missing = docs.filter((d) => !vectors.has(d.key));
  if (missing.length) {
    const vs = await embed(missing.map((d) => d.text), DOC_MAX);
    missing.forEach((d, i) => vectors.set(d.key, vs[i]!));
  }
  return docs.map((d) => vectors.get(d.key)!);
}

const dot = (a: Float32Array, b: Float32Array) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return s;
};

// ---------------------------------------------------------------- collection, catalog with it

export type LocalPick = {
  /** "catalog/collection" of the shortlist to the probability; `keep` for the current collection. */
  probabilities: Record<string, number>;
  /** The option texts, for the answers panel. */
  labels: Record<string, string>;
  /** The best option: "catalog/collection", or `keep`. */
  choice: string;
};

export const KEEP_KEY = "keep";

/**
 * Picks a collection among the collections of the catalogs.
 * @param current The open collection, "catalog/collection", or null.
 */
export async function pickCollection(
  prompt: string,
  catalogs: readonly Catalog[],
  collectionsOf: (catalog: Catalog) => Promise<Collection[]>,
  current: string | null,
): Promise<LocalPick> {
  await loadLocal();
  const lists = await Promise.allSettled(catalogs.map(collectionsOf));
  const docs = lists.flatMap((l, i) =>
    l.status === "fulfilled"
      ? l.value.map((c) => ({ key: `${catalogs[i]!.id}/${c.id}`, text: docText(catalogs[i]!.id, c), c }))
      : [],
  );
  if (docs.length === 0) throw new Error("No catalog lists its collections");
  const [[q], vs] = await Promise.all([embed([prompt], QUERY_MAX), vectorsOf(docs)]);
  const top = docs
    .map((d, i) => ({ d, s: dot(q!, vs[i]!) }))
    .sort((a, b) => b.s - a.s)
    .slice(0, SHORTLIST)
    .map((x) => x.d);
  const options = top.map((d) => d.text);
  const keys = top.map((d) => d.key);
  if (current) {
    options.push(`keep: keep the current collection (${current})`);
    keys.push(KEEP_KEY);
  }
  const p = await call<number[]>({
    type: "score",
    kind: "collection",
    query: query("Which collection?", prompt, current),
    options,
  });
  const probabilities = Object.fromEntries(keys.map((k, i) => [k, p[i]!]));
  const labels = Object.fromEntries(
    top.map((d) => [d.key, `${d.c.title ?? d.c.id} (${d.key.split("/")[0]})`]).concat(current ? [[KEEP_KEY, `Keep ${current}`]] : []),
  );
  const choice = keys[p.indexOf(Math.max(...p))]!;
  return { probabilities, labels, choice };
}

// ---------------------------------------------------------------- place pick

/** Picks the place the request means among the geocoder results, or null for none of them. */
export async function pickPlace(
  prompt: string,
  words: string,
  places: readonly Place[],
): Promise<{ choice: string | null; probabilities: Record<string, number> }> {
  await loadLocal();
  // In the order of the geocoder, most important first, as in training.
  const options = [...places.map((p, i) => `[rank ${i + 1}] ${placeText(p.label)}`), NONE_PLACE];
  const p = await call<number[]>({
    type: "score",
    kind: "place",
    query: query(`Which place is '${words}'?`, prompt),
    options,
  });
  const ids = [...places.map((pl) => pl.id), "none"];
  const probabilities = Object.fromEntries(ids.map((id, i) => [id, p[i]!]));
  const best = ids[p.indexOf(Math.max(...p))]!;
  return { choice: best === "none" ? null : best, probabilities };
}

// ---------------------------------------------------------------- place, time, cloud cover

/** Reads the place, the time and the cloud cover, in the shape of the Clairic extraction. */
export async function extractLocal(prompt: string, today: Date = new Date()): Promise<Extraction> {
  await loadLocal();
  const words = await call<Record<string, { words: string; p: number }>>({ type: "tag", text: prompt });
  const answers: Record<string, ExtractAnswer> = {};
  const place = words.PLACE?.words.replace(/[\s,]+$/, "") ?? null;
  answers.place = place ? { value: place, confidence: words.PLACE!.p, p_null: 0 } : { value: null, p_null: 1 };
  let range: DayRange | null = null;
  if (words.TIME) {
    const t = readTime(words.TIME.words, today);
    range =
      rangeOfAnswers({ when: { value: t.when ?? words.TIME.words }, start: { value: t.start }, end: { value: t.end } }, today) ??
      null;
    answers.when = { value: words.TIME.words, confidence: words.TIME.p, p_null: 0 };
  } else {
    answers.when = { value: null, p_null: 1 };
  }
  let cloud: number | null = null;
  if (words.CLOUD) {
    const w = words.CLOUD.words;
    cloud = readCloud(/cloud|clear|sun/i.test(w) ? w : `${w} clouds`);
    answers.cloud = { value: cloud, confidence: words.CLOUD.p, p_null: cloud === null ? 1 : 0 };
  } else {
    answers.cloud = { value: null, p_null: 1 };
  }
  return {
    place: place && (isMapView(place) || place.length > 0) ? place : null,
    when: words.TIME?.words ?? null,
    range,
    cloud: cloud !== null && cloud < 100 ? cloud : null,
    answers,
  };
}
