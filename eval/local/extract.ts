/**
 * Phase 0 of the local model plan: a rules-only extractor, with no model.
 * It reads the place, the time and the cloud cover with chrono-node (dates),
 * regular expressions and the time rules of the app (`src/zap/extract.ts`).
 * It is the floor that the trained tagger must beat: the tagger finds the
 * words, and code like this turns them into values.
 *
 * It answers the fields of the OpenRouter extractions API, so the extraction
 * test runs it as a model:
 *   MODEL=local:rules VARIANT=compact APP_CALLS=1 pnpm tsx eval/extract.ts
 */
import * as chrono from "chrono-node";
import type { ExtractAnswer, ExtractField, ExtractReply } from "../providers";
import { requestOf } from "./decide";

type Span = { start: number; end: number; from: string; to: string | null; closed?: boolean };

const iso = (y: number, m: number, d: number) => new Date(Date.UTC(y, m - 1, d)).toISOString().slice(0, 10);
const lastDay = (y: number, m: number) => new Date(Date.UTC(y, m, 0)).getUTCDate();

/** The first and the last day of a chrono component, by the parts it knows. */
function period(c: chrono.ParsedComponents): [string, string] {
  const y = c.get("year")!;
  const m = c.get("month")!;
  if (c.isCertain("day")) return [iso(y, m, c.get("day")!), iso(y, m, c.get("day")!)];
  if (c.isCertain("month")) return [iso(y, m, 1), iso(y, m, lastDay(y, m))];
  return [iso(y, 1, 1), iso(y, 12, 31)];
}

const SEASON = /\b(?:(?:last|this|next)\s+)?(?:spring|summer|autumn|fall|winter)(?:\s+(?:of\s+)?\d{4})?\b/i;
const QUARTER = /\b(?:q[1-4]|(?:the\s+)?(?:first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter)(?:\s+(?:of\s+)?\d{4})?\b/i;
const RELATIVE = /\b(?:last|past)\s+(?:\d+|a|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)?\s*(?:day|week|month|year)s?\b/i;
const RECENT = /\b(?:recent|latest|newest|most recent)\b/i;
const YEAR = /\b(?:19|20)\d{2}\b/g;

/** The time words and dates of a request. */
export function readTime(text: string, today: Date) {
  const spans: Span[] = [];
  for (const r of chrono.parse(text, today, { forwardDate: false })) {
    // A bare number such as "20" in "20% clouds" is not a time.
    if (!r.start.isCertain("month") && !r.start.isCertain("year")) continue;
    const [from] = period(r.start);
    const to = period(r.end ?? r.start)[1];
    // "April to September 2023": chrono found both ends, so the period is closed.
    spans.push({ start: r.index, end: r.index + r.text.length, from, to, closed: Boolean(r.end) });
  }
  for (const m of text.matchAll(YEAR)) {
    const i = m.index!;
    if (spans.some((s) => i >= s.start && i < s.end)) continue;
    // Not a part of a name or a number, e.g. "Sentinel-2020" or "2020%".
    if (/[-\w]$/.test(text.slice(0, i)) || /^\d*%/.test(text.slice(i + 4))) continue;
    const y = Number(m[0]);
    spans.push({ start: i, end: i + 4, from: iso(y, 1, 1), to: iso(y, 12, 31) });
  }
  // Words the app turns into a range with code (season, quarter, last N, recent).
  for (const re of [SEASON, QUARTER, RELATIVE, RECENT]) {
    const m = re.exec(text);
    if (m) spans.push({ start: m.index, end: m.index + m[0].length, from: "", to: null });
  }
  if (spans.length === 0) return { when: null, start: null, end: null };
  spans.sort((a, b) => a.start - b.start);
  let start = spans[0]!.start;
  const end = Math.max(...spans.map((s) => s.end));
  const lead = /\b(since|from|between|in|on|during|after|before|until)\s+$/i.exec(text.slice(0, start));
  if (lead) start = lead.index;
  const dated = spans.filter((s) => s.from);
  // Open to today: "since 2020", "after May 2021", "from 2018 until now". A plain "from 2022" is the year 2022, as
  // in the field test and the generator.
  const openWords = /^\s*(?:until now|onwards|on|to date|to today|to the present|up to now)\b/i.test(text.slice(end));
  const since =
    dated.length === 1 && !dated[0]!.closed &&
    (/^(since|after)\b/i.test(lead?.[1] ?? "") || (/^from\b/i.test(lead?.[1] ?? "") && openWords));
  return {
    when: text.slice(start, end),
    start: dated[0]?.from ?? null,
    end: since ? null : (dated.at(-1)?.to ?? null),
  };
}

const CLOUD_WORDS: [RegExp, number][] = [
  [/\b(?:cloud[- ]?free|cloudless|no clouds?)\b/i, 5],
  [/\bmostly clear\b/i, 15],
  [/\b(?:clear(?: sky| skies)?|sunny)\b/i, 10],
  [/\bfew clouds\b/i, 20],
  [/\b(?:some clouds|not too cloudy)\b/i, 30],
  [/\bpartly cloudy\b/i, 50],
];

/** The maximum cloud cover in percent, or null for no filter. */
export function readCloud(text: string): number | null {
  if (/\bany clouds?\b|\bany cloud cover\b/i.test(text)) return null;
  const pct = /(\d{1,3})\s*(?:%|percent)/i.exec(text);
  if (pct && /cloud/i.test(text)) return Number(pct[1]);
  for (const [re, value] of CLOUD_WORDS) if (re.test(text)) return value;
  return null;
}

/** Capitalized words that name data, not places. */
const DATA = new Set(
  "sentinel landsat naip modis maxar swiss arcticdem dem lidar level l1c l2a s1 s2 canadian finnish carinthia kentucky global copernicus sar i".split(" "),
);
const MAP_VIEW = /\b(?:here|this (?:area|region|place|view)|the (?:map view|map|current view))\b/i;
const NAME = String.raw`\p{Lu}[\p{L}'.-]*`;
const PLACE = new RegExp(
  String.raw`\b(?:over|of|in|near|around|across|for|at)\s+((?:the\s+)?${NAME}(?:(?:\s+|\s*,\s*|\s+(?:de|del|da|do|la|las|los|of|the|and)\s+)${NAME})*)`,
  "gu",
);

/** The words that name the place, or null. */
export function readPlace(text: string, timeWords: string | null): string | null {
  const view = MAP_VIEW.exec(text);
  if (view) return view[0];
  const rest = timeWords ? text.replace(timeWords, " ") : text;
  const isData = (words: string) =>
    words.split(/[\s,]+/).filter((w) => !/^the$/i.test(w)).every((w) => DATA.has(w.toLowerCase().replace(/-.*$/, "")));
  for (const m of rest.matchAll(PLACE)) {
    const words = m[1]!.replace(/^the\s+/i, "").replace(/[\s,]+$/, "");
    if (!isData(words) && !/^\p{Lu}+-?\d/u.test(words)) return words;
  }
  // No preposition: the last run of capitalized words that is not data.
  const runs = [...rest.matchAll(new RegExp(`${NAME}(?:\\s+${NAME})*`, "gu"))].map((m) => m[0]);
  return runs.reverse().find((w) => !isData(w) && !/\d/.test(w)) ?? null;
}

/** Answers the requested fields with the rules. */
export async function extractLocal(
  _model: string,
  state: string,
  fields: Record<string, ExtractField>,
  referenceDate?: string,
): Promise<ExtractReply> {
  const t = performance.now();
  const text = requestOf(state);
  const today = referenceDate ? new Date(`${referenceDate}T12:00:00Z`) : new Date();
  const time = readTime(text, today);
  const values: Record<string, string | number | null> = {
    ...time,
    cloud: readCloud(text),
    place: readPlace(text, time.when),
  };
  const answers: Record<string, ExtractAnswer> = {};
  for (const name of Object.keys(fields)) {
    const value = values[name] ?? null;
    answers[name] = { value, p_null: value === null ? 1 : 0 };
  }
  return { answers, ms: Math.round(performance.now() - t), cost: 0, usage: {} };
}
