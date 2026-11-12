/**
 * Scores values (a place, a date range, a cloud cover) instead of option ids.
 * The extraction test (`extract.ts`) and the unparsed test (`unparsed.ts`)
 * use the same rules, so the scores of jev and of the extraction models
 * compare.
 */
import { MAP_VIEW, NO_PLACE, toRange } from "../src/zap/options";
import { NO_CHANGE, type Case } from "./cases";
import { TODAY } from "./questions";

export type Range = [string, string];

/** The year of a season with no year, as `latestYearOf` in run.ts. */
export function latestYearOf(season: string): string {
  const year = TODAY.getUTCFullYear();
  const range = toRange(String(year), season, TODAY);
  const started = range && new Date(`${range.startDatetime}Z`) <= TODAY;
  return String(started ? year : year - 1);
}

/**
 * The range of a season in the words of a time, e.g. "winter 2024" or "last
 * summer", with the app's own `toRange`. Undefined when the words name no
 * season. A season with no year is the most recent one, as in the app.
 */
export function seasonRange(words: string): Range | undefined {
  const name = /\b(spring|summer|autumn|fall|winter)\b/i.exec(words)?.[1]?.toLowerCase();
  if (!name) return undefined;
  const season = name === "fall" ? "autumn" : name;
  const year = /\b(\d{4})\b/.exec(words)?.[1] ?? latestYearOf(season);
  const r = toRange(year, season, TODAY);
  return r ? [r.startDatetime.slice(0, 10), r.endDatetime.slice(0, 10)] : undefined;
}

const isoDay = (d: Date) => d.toISOString().slice(0, 10);
const NUMBERS: Record<string, number> = {
  a: 1, one: 1, two: 2, three: 3, four: 4, five: 5, six: 6,
  seven: 7, eight: 8, nine: 9, ten: 10, eleven: 11, twelve: 12,
};

/**
 * The range of "the last 3 months", "past 10 days" or "past week" in the
 * words of a time: back from today. A month or a year needs a number, as
 * "last month" and "last year" can also mean the calendar month or year
 * before. Undefined for other words.
 */
export function relativeRange(words: string): Range | undefined {
  const m = /\b(?:last|past)\s+(?:(\d+|a|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+)?(day|week|month|year)s?\b/i.exec(words);
  if (!m) return undefined;
  const unit = m[2]!.toLowerCase();
  if (!m[1] && (unit === "month" || unit === "year")) return undefined;
  const n = m[1] ? (NUMBERS[m[1].toLowerCase()] ?? Number(m[1])) : 1;
  const start = new Date(TODAY);
  if (unit === "day") start.setUTCDate(start.getUTCDate() - n);
  if (unit === "week") start.setUTCDate(start.getUTCDate() - 7 * n);
  if (unit === "month") start.setUTCMonth(start.getUTCMonth() - n);
  if (unit === "year") start.setUTCFullYear(start.getUTCFullYear() - n);
  return [isoDay(start), isoDay(TODAY)];
}

const QUARTERS: Record<string, number> = { first: 1, "1st": 1, second: 2, "2nd": 2, third: 3, "3rd": 3, fourth: 4, "4th": 4 };

/**
 * The range of a quarter in the words of a time, e.g. "Q3 2021" or "the
 * second quarter of 2022". With no year, the most recent quarter that has
 * started. Undefined for other words.
 */
export function quarterRange(words: string): Range | undefined {
  const m = /\b(?:q([1-4])|(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter)\b/i.exec(words);
  if (!m) return undefined;
  const q = m[1] ? Number(m[1]) : QUARTERS[m[2]!.toLowerCase()]!;
  let year = Number(/\b(\d{4})\b/.exec(words)?.[1] ?? TODAY.getUTCFullYear());
  if (!/\b\d{4}\b/.test(words) && Date.UTC(year, 3 * (q - 1), 1) > TODAY.getTime()) year--;
  const start = new Date(Date.UTC(year, 3 * (q - 1), 1));
  const end = new Date(Date.UTC(year, 3 * q, 0));
  return [isoDay(start), isoDay(end)];
}

/**
 * The range of "recent" or "latest" in the words of a time: the last 30
 * days, as the period option of the app ("Recent: the last 30 days, the
 * latest imagery"). Undefined for other words.
 */
export function recentRange(words: string): Range | undefined {
  if (!/\b(recent|latest)\b/i.test(words)) return undefined;
  const r = toRange("last-30-days", "whole-year", TODAY);
  return r ? [r.startDatetime.slice(0, 10), r.endDatetime.slice(0, 10)] : undefined;
}

const DAY = 86_400_000;
const days = (a: string, b: string) => Math.abs(Date.parse(a) - Date.parse(b)) / DAY;

/** Words that mean the area the map shows now. */
const MAP_WORDS = new Set(["here", "this area", "this region", "this place", "the map", "the map view"]);

/** A place name in a form to compare: lower case, no punctuation, no "the". */
export function normPlace(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^the /, "");
}

/** The place as an id of the field test: a name, {@link MAP_VIEW} or {@link NO_PLACE}. */
export function placeOf(value: string | null): string {
  if (value === null) return NO_PLACE;
  return MAP_WORDS.has(normPlace(value)) ? MAP_VIEW : value;
}

export function placeOk(value: string | null, expect: string[]): boolean {
  const got = placeOf(value);
  if (got === MAP_VIEW || got === NO_PLACE) return expect.includes(got);
  return expect.some((e) => e !== MAP_VIEW && e !== NO_PLACE && normPlace(e) === normPlace(got));
}

/**
 * True when a range matches the expected range. A fixed range must match to
 * the day. A range back from today ("the last 3 months") must end within one
 * day of the expected end and start within a fifth of its length.
 * @param expect Null when the prompt asks for no time filter.
 */
export function rangeOk(got: Range | null, expect: Range | null, relative = false): boolean {
  if (!got || !expect) return got === expect;
  if (!relative) return got[0] === expect[0] && got[1] === expect[1];
  const tolerance = Math.max(1, days(expect[0], expect[1]) / 5);
  return days(got[1], expect[1]) <= 1 && days(got[0], expect[0]) <= tolerance;
}

/** True when a maximum cloud cover in percent is the expected one. */
export function cloudOk(got: number | null, expect: number | null): boolean {
  if (expect === null) return got === null || got >= 100;
  return got === expect;
}

/**
 * The accepted ranges of a field test case, from its period and season ids.
 * Null in the list means "no time filter". Undefined when the case does not
 * score the time.
 */
export function rangesOf(c: Case): { ranges: (Range | null)[]; relative: boolean } | undefined {
  if (!c.period) return undefined;
  const periods = c.period === NO_CHANGE ? ["all-time"] : c.period;
  const seasons = !c.season || c.season === NO_CHANGE ? ["whole-year"] : c.season;
  const ranges = periods.flatMap((p) =>
    seasons.map((s) => {
      const r = toRange(p, s, TODAY);
      return r ? ([r.startDatetime.slice(0, 10), r.endDatetime.slice(0, 10)] as Range) : null;
    }),
  );
  const relative = periods.some((p) => p.startsWith("last-"));
  return { ranges, relative };
}

/**
 * The accepted cloud cover of a field test case, as an inclusive range in
 * percent. Null when the case asks for no filter. Undefined when the case
 * does not score cloud cover. "Cloud-free" (5%) also accepts 0%.
 */
export function cloudsOf(c: Case): [number, number] | null | undefined {
  if (!c.cloud) return undefined;
  const values = c.cloud === NO_CHANGE ? [] : c.cloud.map(Number).filter((n) => !Number.isNaN(n));
  if (values.length === 0) return null;
  const lo = Math.min(...values);
  return [lo === 5 ? 0 : lo, Math.max(...values)];
}

export function cloudInRange(got: number | null, expect: [number, number] | null): boolean {
  if (expect === null) return got === null || got >= 100;
  return got !== null && got >= expect[0] && got <= expect[1];
}
