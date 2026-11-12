/**
 * The questions of stac-zap for one prompt, as the app asks them on a fresh
 * map. The field test (`fields.ts`) and the unparsed test (`unparsed.ts`)
 * use them, so both ask the same questions as the app.
 */
import { readFileSync } from "node:fs";
import { shortlist, type Catalog, type Collection } from "../src/zap/catalogs";
import { buildPayload, type ZapFields } from "../src/zap/decide";
import { placeSpans } from "../src/zap/geocode";
import {
  CLOUD_OPTIONS,
  CLOUD_QUESTION,
  MAP_VIEW,
  NO_PLACE,
  periodOptions,
  SEASON_OPTIONS,
} from "../src/zap/options";
import type { Question } from "./providers";

/** The expected answers assume this date, e.g. "last summer" is 2026. */
export const TODAY = new Date("2026-10-02T12:00:00Z");

export const FIELDS = ["catalog", "collection", "place", "period", "season", "cloud"] as const;
export type Field = (typeof FIELDS)[number];

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
export function questionsOf(prompt: string, catalogId: string) {
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
