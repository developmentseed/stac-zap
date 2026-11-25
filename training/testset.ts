/**
 * The test set kit. People write the test requests; no model writes them.
 *
 * `sheet` makes a CSV of cards. Each card names a collection, a place in its
 * extent, and whether to ask for a time and a cloud cover. A person writes a
 * request for the card in their own words, as they would type it into
 * stac-zap, and fills in what the app must read from it:
 * - catalogs_ok, collections_ok: the correct ids, separated by ";" (more
 *   than one when the same data is in several catalogs or collections),
 * - place_words, time_words: the words of the request, exactly as written,
 * - start, end: the first and the last day (YYYY-MM-DD); no end means today,
 * - cloud: the maximum cloud cover in percent, or empty for no filter,
 * - date: the day the request was written; writer: a name or initials. A card is a hint:
 * the writer can change the place, the time or the collection, and must then
 * change the columns too. Some cards are "free": write any request.
 *
 * `import` checks the filled sheet and writes test records in the format of
 * `generate.ts` (split "test-human").
 *
 * Usage:
 *   pnpm tsx training/testset.ts sheet [cards, default 240] > training/data/testset-sheet.csv
 *   pnpm tsx training/testset.ts import training/data/testset-filled.csv > training/data/test-human.jsonl
 */
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import type { RecordOut } from "./generate";

const DATA = new URL("data/", import.meta.url);
type Rec = RecordOut;

const COLUMNS = [
  "card", "catalog", "collection", "collection_title", "hint_place", "ask_time", "ask_cloud",
  // Filled by the writer:
  "request", "catalogs_ok", "collections_ok", "place_words", "time_words", "start", "end", "cloud", "date", "writer", "notes",
] as const;

const csvCell = (v: unknown) => {
  const s = v === null || v === undefined ? "" : String(v);
  return /[",\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
};

function parseCsv(text: string): Record<string, string>[] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]!;
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') (cell += '"'), i++;
      else if (ch === '"') quoted = false;
      else cell += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") row.push(cell), (cell = "");
    else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(cell), rows.push(row), (row = []), (cell = "");
    } else cell += ch;
  }
  if (cell || row.length) row.push(cell), rows.push(row);
  const [head, ...body] = rows.filter((r) => r.some((c) => c.trim()));
  return body.map((r) => Object.fromEntries(head!.map((h, i) => [h.trim(), (r[i] ?? "").trim()])));
}

/** Cards from the generated records: one per collection, spread over catalogs, held-out catalogs included. */
function sheet(n: number) {
  const records: Rec[] = readFileSync(new URL("records.jsonl", DATA), "utf-8")
    .trim().split("\n").map((l) => JSON.parse(l))
    .filter((r: Rec) => r.collection.target && !r.current.catalog);
  const snapshot = JSON.parse(readFileSync(new URL("stac-full.json", DATA), "utf-8"));
  const titles = new Map<string, string>();
  for (const [cat, list] of Object.entries(snapshot.collections) as [string, { id: string; title?: string }[]][]) {
    for (const c of list) titles.set(`${cat}/${c.id}`, c.title ?? c.id);
  }
  // One record per catalog in turn, so small catalogs get cards too; a hash orders them.
  const byCatalog = new Map<string, Rec[]>();
  const h = (s: string) => createHash("sha1").update(s).digest("hex");
  for (const r of [...records].sort((a, b) => h(a.id).localeCompare(h(b.id)))) {
    const k = r.catalog.target!;
    if (!byCatalog.has(k)) byCatalog.set(k, []);
    const list = byCatalog.get(k)!;
    if (!list.some((x) => x.collection.target === r.collection.target)) list.push(r);
  }
  const cards: Rec[] = [];
  for (let i = 0; cards.length < n && i < 1000; i++) {
    for (const list of byCatalog.values()) if (list[i] && cards.length < n) cards.push(list[i]!);
  }
  console.log(COLUMNS.join(","));
  cards.forEach((r, i) => {
    const free = i % 6 === 5;
    const cells: Record<string, unknown> = free
      ? { card: `c${String(i + 1).padStart(3, "0")}`, catalog: "(free: any request)" }
      : {
          card: `c${String(i + 1).padStart(3, "0")}`,
          catalog: r.catalog.target,
          collection: r.collection.target,
          collection_title: titles.get(`${r.catalog.target}/${r.collection.target}`),
          hint_place: r.place.words ?? "(no place)",
          ask_time: i % 3 === 0 ? "yes" : "no",
          ask_cloud: i % 7 === 0 ? "yes" : "no",
        };
    console.log(COLUMNS.map((c) => csvCell(cells[c])).join(","));
  });
}

/** The span of the words in the request, ignoring case. Throws when they are not there. */
function spanOf(request: string, words: string, what: string, card: string): [number, number] {
  const i = request.toLowerCase().indexOf(words.toLowerCase());
  if (i < 0) throw new Error(`${card}: the ${what} words "${words}" are not in the request`);
  return [i, i + words.length];
}

const DAY = /^\d{4}-\d{2}-\d{2}$/;

function importSheet(file: string) {
  const rows = parseCsv(readFileSync(file, "utf-8")).filter((r) => r.request);
  const errors: string[] = [];
  const out: Rec[] = [];
  for (const r of rows) {
    try {
      const card = r.card!;
      const list = (s: string | undefined) => (s ?? "").split(/[;|]/).map((x) => x.trim()).filter(Boolean);
      const catalogs = list(r.catalogs_ok);
      const collections = list(r.collections_ok);
      if (!catalogs.length) throw new Error(`${card}: catalogs_ok is empty`);
      if (r.date && !DAY.test(r.date)) throw new Error(`${card}: date is not YYYY-MM-DD`);
      if (r.start && !DAY.test(r.start)) throw new Error(`${card}: start is not YYYY-MM-DD`);
      if (r.end && !DAY.test(r.end)) throw new Error(`${card}: end is not YYYY-MM-DD`);
      if (r.time_words && !r.start) throw new Error(`${card}: time words with no start`);
      const cloud = r.cloud ? Number(r.cloud) : null;
      if (r.cloud && !Number.isFinite(cloud)) throw new Error(`${card}: cloud is not a number`);
      const mapView = /^(here|this area|this region|the map( view)?|the current view)$/i.test(r.place_words ?? "");
      out.push({
        id: `h${card}`,
        split: "test-human",
        request: r.request!,
        // The day the request was written: "last summer" counts from it.
        today: r.date || new Date().toISOString().slice(0, 10),
        current: { catalog: null, collection: null, place: null },
        catalog: { target: catalogs[0]!, accepted: catalogs },
        collection: { target: collections[0] ?? null, accepted: collections },
        place: r.place_words
          ? { span: spanOf(r.request!, r.place_words, "place", card), words: r.place_words, map_view: mapView, target: null, candidates: [] }
          : { span: null, words: null, map_view: false, target: null, candidates: [] },
        time: r.time_words
          ? { span: spanOf(r.request!, r.time_words, "time", card), range: [r.start!, r.end || r.date || "today"] }
          : { span: null, range: null },
        // The writer gives the number; the cloud words are not marked.
        cloud: { span: null, value: cloud },
        template: "human",
        source: `human:${r.writer || "unknown"}`,
      });
    } catch (e) {
      errors.push(String((e as Error).message));
    }
  }
  for (const e of errors) console.error(e);
  console.error(`${out.length} records, ${errors.length} errors`);
  for (const r of out) console.log(JSON.stringify(r));
  if (errors.length) process.exitCode = 1;
}

const [mode, arg] = process.argv.slice(2);
if (mode === "sheet") sheet(Number(arg ?? 240));
else if (mode === "import" && arg) importSheet(arg);
else console.error("Usage: testset.ts sheet [n] | import <filled.csv>");
