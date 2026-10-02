/**
 * The field test: asks one decision model the questions of stac-zap for each
 * prompt in `cases.ts`, and scores the answers as the app applies them.
 *
 * The questions come from the app code (`buildPayload`, the option lists and
 * the cloud cover question), with the catalogs of `data/stac-snapshot.json`.
 * Each field is asked in its own call, so a model with a limit on the number
 * of options can still answer the other fields. The app asks the second-step
 * fields in one call.
 *
 * Usage:
 *   PROVIDER=openrouter MODEL=typesafe/jev-1.13 pnpm tsx eval/fields.ts
 * Optional: FIELDS=cloud,period asks only these fields and replaces them in
 * the result file. RESULTS=eval/results/<date> sets the output folder.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { shortlist, type Catalog, type Collection } from "../src/zap/catalogs";
import { buildPayload, KEEP, THRESHOLD, type ZapFields } from "../src/zap/decide";
import { placeSpans } from "../src/zap/geocode";
import {
  cloudChoice,
  CLOUD_OPTIONS,
  CLOUD_QUESTION,
  MAP_VIEW,
  NO_PLACE,
  periodOptions,
  SEASON_OPTIONS,
} from "../src/zap/options";
import { CASES, NO_CHANGE } from "./cases";
import { ask, modelOfEnv, type Question } from "./providers";

/** The expected answers assume this date, e.g. "last summer" is 2026. */
const TODAY = new Date("2026-10-02T12:00:00Z");
const FIELDS = ["catalog", "collection", "place", "period", "season", "cloud"] as const;
type Field = (typeof FIELDS)[number];
/** The probability the app needs to change a field, as in run.ts. */
const THRESHOLDS: Record<Exclude<Field, "cloud">, number> = {
  catalog: 0,
  collection: 0,
  place: THRESHOLD,
  period: 0.35,
  season: THRESHOLD,
};
const NO_CHANGE_OF: Record<string, string[]> = {
  period: [KEEP, "all-time"],
  season: [KEEP, "whole-year"],
  cloud: [KEEP, "any"],
};

type Snapshot = { catalogs: Catalog[]; collections: Record<string, Collection[]> };
const snapshot: Snapshot = JSON.parse(
  readFileSync(new URL("data/stac-snapshot.json", import.meta.url), "utf-8"),
);

/** The collection as jev reads it, as `describe` in run.ts. */
function describe(collection: Collection): string {
  const text = (collection.description ?? "")
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/[#*_`>]/g, "")
    .replace(/\s+/g, " ")
    .trim();
  const short = text.length > 120 ? `${text.slice(0, 120)}…` : text;
  const title = collection.title ?? collection.id;
  return short ? `${title}: ${short}` : title;
}

/** One question per field, with its state, as the app asks on a fresh map. */
function questionsOf(prompt: string, catalogId: string) {
  const first: ZapFields = {
    catalog: {
      label: "STAC catalog",
      current: null,
      options: Object.fromEntries(snapshot.catalogs.map((c) => [c.id, c.description])),
      keepable: false,
      threshold: 0,
    },
  };
  const collections = shortlist(snapshot.collections[catalogId] ?? [], prompt);
  // The second call of run.ts, with no collection, place or period yet.
  const second: ZapFields = {
    collection: {
      label: "collection",
      current: null,
      options: Object.fromEntries(collections.map((c) => [c.id, describe(c)])),
      keepable: false,
      threshold: 0,
    },
    place: {
      label: "place to search",
      current: null,
      options: {
        ...Object.fromEntries(placeSpans(prompt).map((span) => [span, span])),
        [MAP_VIEW]: "The area the map shows now, e.g. 'here' or 'this area'",
        [NO_PLACE]:
          "No place: search everywhere. The request names no place, only " +
          "data, an event such as a hurricane, or a time",
      },
      question:
        "Which words of the request name the place to search? A place can be " +
        "a city, a region, a country, a landmark or a natural feature, " +
        "such as a mountain range, a river or a desert. The name of an " +
        "event, e.g. 'Hurricane Idalia', or of a dataset is not a place.",
    },
    period: { label: "time period", current: null, options: periodOptions(TODAY), threshold: 0.35 },
    season: { label: "part of the year", current: "whole-year", options: SEASON_OPTIONS },
    cloud: {
      label: "maximum cloud cover",
      current: "any",
      options: CLOUD_OPTIONS,
      question: CLOUD_QUESTION,
    },
  };
  const p1 = buildPayload(prompt, first, "", TODAY);
  const p2 = buildPayload(prompt, second, "", TODAY);
  const out: Partial<Record<Field, { state: string; question: Question }>> = {
    catalog: { state: p1.state, question: p1.questions.catalog as Question },
  };
  for (const name of ["collection", "place", "period", "season", "cloud"] as const) {
    out[name] = { state: p2.state, question: p2.questions[name] as Question };
  }
  return out;
}

export type Row = {
  prompt: string;
  field: Field;
  expect: string[];
  options: number;
  error?: string;
  /** The option the model liked most. */
  top?: string;
  p?: number;
  /** What the app does with the answer: the option, or `keep`. */
  choice?: string;
  ok?: boolean;
  /** The five most likely options. */
  probabilities?: Record<string, number>;
  ms?: number;
  cost?: number;
};

const { provider, model, tag } = modelOfEnv();
const only = process.env.FIELDS?.split(",") as Field[] | undefined;
const dir = process.env.RESULTS ?? `eval/results/${new Date().toISOString().slice(0, 10)}`;
const file = `${dir}/fields-${tag}.json`;

const rows: Row[] = [];
for (const c of CASES) {
  const catalogId = c.cat ?? c.catalog[0]!;
  const questions = questionsOf(c.prompt, catalogId);
  for (const field of FIELDS) {
    if (only && !only.includes(field)) continue;
    const expected = field === "catalog" ? c.catalog : c[field];
    const asked = questions[field];
    if (!expected || !asked) continue;
    const options = Object.keys(asked.question.criteria).length;
    // A catalog with one collection leaves no choice.
    if (options < 2) continue;
    const expect = expected === NO_CHANGE ? NO_CHANGE_OF[field]! : expected;
    const row: Row = { prompt: c.prompt, field, expect, options };
    const reply = await ask(provider, model, asked.state, { q: asked.question });
    const answer = "error" in reply ? undefined : reply.answers.q;
    if ("error" in reply || !answer) {
      row.error = "error" in reply ? reply.error : "no answer";
    } else {
      const p = answer.probabilities ?? {};
      row.top = answer.choice;
      row.p = p[answer.choice] ?? answer.confidence ?? 0;
      row.choice =
        field === "cloud"
          ? (cloudChoice(p, KEEP) ?? KEEP)
          : row.p >= THRESHOLDS[field]
            ? row.top
            : KEEP;
      row.ok = expect.includes(row.choice);
      row.probabilities = Object.fromEntries(
        Object.entries(p).sort((a, b) => b[1] - a[1]).slice(0, 5),
      );
      row.ms = reply.ms;
      row.cost = reply.cost;
    }
    rows.push(row);
    process.stderr.write(row.error ? "x" : row.ok ? "." : "-");
  }
}
console.error();

// With FIELDS, the other fields of an earlier run stay.
let all = rows;
if (only && existsSync(file)) {
  const before: { rows: Row[] } = JSON.parse(readFileSync(file, "utf-8"));
  all = [...before.rows.filter((r) => !only.includes(r.field)), ...rows];
}
mkdirSync(dir, { recursive: true });
writeFileSync(file, JSON.stringify({ provider, model, today: TODAY.toISOString().slice(0, 10), rows: all }, null, 1));
const score = (field: Field) => {
  const r = all.filter((x) => x.field === field);
  return `${field} ${r.filter((x) => x.ok).length}/${r.length}`;
};
console.log(`${model}: ${FIELDS.map(score).join(", ")} → ${file}`);
