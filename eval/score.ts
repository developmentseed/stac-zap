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
