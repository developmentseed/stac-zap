/**
 * Reads the place, the time and the maximum cloud cover of a prompt with an
 * extraction model (Clairic on the OpenRouter extractions API).
 *
 * An extraction model does not select options. It gets a text and named,
 * typed fields, and returns a value for each field, or null when the text
 * does not have it. So it can read values that the option lists of jev
 * cannot hold, e.g. "between March 2023 and June 2024" or "under 12% clouds".
 *
 * The model reads words and values, and code does the calendar: a season, a
 * quarter, "the last 3 months", "recent" or "latest" in the time words gets
 * a range from code, not the dates of the model. The design and its test
 * are in docs/decision-models.md ("Test 3: extraction models").
 */
import { buildPayload } from "./decide";
import { CLOUD_OPTIONS, toRange, latestYearOf, type Range } from "./options";

/** A field of the extractions API. Only the types the app uses. */
export type ExtractField = {
  type: "text" | "number" | "date";
  instructions?: string;
  unit?: string;
};

/** One answer of the extractions API, for a text, number or date field. */
export type ExtractAnswer = {
  /** Null when `p_null` is at least the null threshold (0.5). */
  value: string | number | null;
  /** The value the model would give if it had to give one. */
  guess?: string | number | null;
  /** The probability that the request does not contain the value. */
  p_null?: number;
  confidence?: number;
  /** [start, end) code point offsets into the state. */
  span?: [number, number];
};

/** The first and the last day of a period, both included, as `YYYY-MM-DD`. */
export type DayRange = [string, string];

const PLACE =
  "The words of the request that name the place to search. A place can be " +
  "a city, a region, a country, a landmark or a natural feature, such as " +
  "a mountain range, a river or a desert. The name of an event, e.g. " +
  "'Hurricane Idalia', or of a dataset is not a place. If the request says " +
  "'here', 'this area' or 'the map', give those words.";

/** "cloud-free 5, clear sky 10, ...": the cloud words of the app. */
const CLOUD_MAP = Object.entries(CLOUD_OPTIONS)
  .filter(([id, text]) => id !== "any" && text.includes(":"))
  .map(([id, text]) => `${text.split(":")[0]!.toLowerCase()} ${id}`)
  .join(", ");

/**
 * The fields of each group. The app asks all of them in one call; the test
 * (`eval/extract.ts`) can ask each group in its own call.
 */
export const EXTRACT_GROUPS = {
  place: {
    place: { type: "text", instructions: PLACE },
  },
  time: {
    // With today's date in the state, the model gives a date range for
    // requests with no time, e.g. "orthophotos of Bern". The words of the
    // time tell if there is one, as for the place. "Recent" and "latest"
    // come from the period options of the app.
    when: {
      type: "text",
      instructions:
        "The words of the request that name a time, a date or a period, " +
        "e.g. 'in 2021', 'last summer', 'recent', 'latest' or 'since May'. " +
        "A dataset name or a place is not a time.",
    },
    // Full sentences: short phrases gave worse dates. No season text: it
    // also moved months that are not in a season, so code does seasons.
    start: {
      type: "date",
      instructions:
        "The first day of the time period the request asks for. For a day, " +
        "that day; for a month, the 1st of that month; for a year, 1 January.",
    },
    end: {
      type: "date",
      instructions:
        "The last day of the time period the request asks for. For a day, " +
        "that day; for a month, the last day of that month; for a year, " +
        "31 December. If the period runs up to now, the reference date.",
    },
  },
  cloud: {
    cloud: {
      type: "number",
      unit: "percent",
      instructions:
        "The maximum cloud cover, only if the request mentions clouds or a clear sky. " +
        `Words: ${CLOUD_MAP}. Any cloud cover: no maximum.`,
    },
  },
} as const satisfies Record<string, Record<string, ExtractField>>;

/** One extraction call: its fields, and whether its state has the app context. */
export type ExtractCall = { fields: Record<string, ExtractField>; context: boolean };

/**
 * The calls the app makes, at the same time.
 * - The time fields go in their own call: in one call with the cloud cover,
 *   a number such as "12% clouds" moved into the dates, e.g. "June 2024"
 *   ended on 2024-06-02.
 * - The time call reads the app context and today's date (see
 *   {@link extractState}); the dates were worse without them. The place
 *   and cloud call reads the request only: it is cheaper, and with the
 *   context, "mostly clear" became 50%.
 */
export const EXTRACT_CALLS: ExtractCall[] = [
  { fields: { ...EXTRACT_GROUPS.place, ...EXTRACT_GROUPS.cloud }, context: false },
  { fields: EXTRACT_GROUPS.time, context: true },
];

/** The state of one call: the request, or the request with the app context and today's date. */
export function callState(call: ExtractCall, prompt: string, today: Date = new Date()): string {
  return call.context ? extractState(prompt, today) : prompt;
}

/**
 * The text the model reads: the state that jev reads (the app, today's date
 * and the request), with no current map. Without the app context, the dates
 * were worse.
 */
export function extractState(prompt: string, today: Date = new Date()): string {
  return buildPayload(prompt, {}, "", today).state.replace(/\n\nCurrent map:\n$/, "");
}

const isoDay = (d: Date) => d.toISOString().slice(0, 10);
const dayRange = (r: Range | null): DayRange | undefined =>
  r ? [r.startDatetime.slice(0, 10), r.endDatetime.slice(0, 10)] : undefined;

/**
 * The range of a season in the words of a time, e.g. "winter 2024" or "last
 * summer", from `toRange`, as for a jev answer. A season with no year is the
 * most recent one. Undefined when the words name no season.
 */
export function seasonRange(words: string, today: Date = new Date()): DayRange | undefined {
  const name = /\b(spring|summer|autumn|fall|winter)\b/i.exec(words)?.[1]?.toLowerCase();
  if (!name) return undefined;
  const season = name === "fall" ? "autumn" : name;
  const year = /\b(\d{4})\b/.exec(words)?.[1] ?? latestYearOf(season, today);
  return dayRange(toRange(year, season, today));
}

const QUARTERS: Record<string, number> = {
  first: 1, "1st": 1, second: 2, "2nd": 2, third: 3, "3rd": 3, fourth: 4, "4th": 4,
};

/**
 * The range of a quarter in the words of a time, e.g. "Q3 2021" or "the
 * second quarter of 2022". With no year, the most recent quarter that has
 * started. Undefined for other words.
 */
export function quarterRange(words: string, today: Date = new Date()): DayRange | undefined {
  const m = /\b(?:q([1-4])|(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter)\b/i.exec(words);
  if (!m) return undefined;
  const q = m[1] ? Number(m[1]) : QUARTERS[m[2]!.toLowerCase()]!;
  const named = /\b(\d{4})\b/.exec(words)?.[1];
  let year = Number(named ?? today.getUTCFullYear());
  if (!named && Date.UTC(year, 3 * (q - 1), 1) > today.getTime()) year--;
  return [isoDay(new Date(Date.UTC(year, 3 * (q - 1), 1))), isoDay(new Date(Date.UTC(year, 3 * q, 0)))];
}

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
export function relativeRange(words: string, today: Date = new Date()): DayRange | undefined {
  const m =
    /\b(?:last|past)\s+(?:(\d+|a|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+)?(day|week|month|year)s?\b/i.exec(
      words,
    );
  if (!m) return undefined;
  const unit = m[2]!.toLowerCase();
  if (!m[1] && (unit === "month" || unit === "year")) return undefined;
  const n = m[1] ? (NUMBERS[m[1].toLowerCase()] ?? Number(m[1])) : 1;
  const start = new Date(today);
  if (unit === "day") start.setUTCDate(start.getUTCDate() - n);
  if (unit === "week") start.setUTCDate(start.getUTCDate() - 7 * n);
  if (unit === "month") start.setUTCMonth(start.getUTCMonth() - n);
  if (unit === "year") start.setUTCFullYear(start.getUTCFullYear() - n);
  return [isoDay(start), isoDay(today)];
}

/**
 * The range of "recent" or "latest": the last 30 days, as the period option
 * of the app ("Recent: the last 30 days, the latest imagery"). Undefined for
 * other words.
 */
export function recentRange(words: string, today: Date = new Date()): DayRange | undefined {
  if (!/\b(recent|latest)\b/i.test(words)) return undefined;
  return dayRange(toRange("last-30-days", "whole-year", today));
}

/** The value of an answer, or null when the model says it is not there. */
export const valueOf = (a: ExtractAnswer | undefined) => (a?.value === undefined ? null : a.value);

/**
 * The time range of the answers.
 * - No `when` words: null, the request asks for no time.
 * - A season, a quarter, "the last N days/weeks/months/years", "recent" or
 *   "latest" in the words: the range from code.
 * - Else the start and end dates. With no end, the range runs to today,
 *   e.g. "since 2020".
 * @returns Undefined when the answers make no range, e.g. an end with no start.
 */
export function rangeOfAnswers(
  answers: Record<string, ExtractAnswer | undefined>,
  today: Date = new Date(),
): DayRange | null | undefined {
  const when = valueOf(answers.when);
  if ("when" in answers && when === null) return null;
  if (typeof when === "string") {
    const range =
      seasonRange(when, today) ??
      quarterRange(when, today) ??
      relativeRange(when, today) ??
      recentRange(when, today);
    if (range) return range;
  }
  const start = valueOf(answers.start);
  const end = valueOf(answers.end);
  if (start === null && end === null) return null;
  if (start === null) return undefined;
  return [String(start), end === null ? isoDay(today) : String(end)];
}

/**
 * Whether the place words mean the area the map shows now, e.g. "here",
 * "this area" or "the map view".
 */
export function isMapView(words: string): boolean {
  return /^(?:in |on )?(?:here|this (?:area|region|place|view)|the (?:map|map view|current view))$/i.test(
    words.trim().replace(/[.,!?]+$/, ""),
  );
}

/** A day range in stac-map's datetime input format: the whole first and last day. */
export function toInputRange([start, end]: DayRange): Range {
  return { startDatetime: `${start}T00:00:00`, endDatetime: `${end}T23:59:59` };
}

/** What the app reads from one extraction. Null means "not in the prompt". */
export type Extraction = {
  /** The words that name the place. */
  place: string | null;
  /** The words that name the time. */
  when: string | null;
  range: DayRange | null;
  /** The maximum cloud cover in percent. */
  cloud: number | null;
  /** The raw answers, for the answers panel. */
  answers: Record<string, ExtractAnswer>;
};

/**
 * Reads the place, the time and the cloud cover of a prompt, in the calls of
 * {@link EXTRACT_CALLS} to the `/api/extract` endpoint, at the same time.
 * @throws If the endpoint cannot answer one of the calls.
 */
export async function extract(prompt: string, today: Date = new Date()): Promise<Extraction> {
  const model = import.meta.env.VITE_EXTRACT_MODEL || "cloudflare/clairic";
  // The deployed site calls the Worker in `worker/`; the dev server has its
  // own `/api/extract`.
  const decideUrl = import.meta.env.VITE_DECIDE_URL as string | undefined;
  const url = decideUrl
    ? `${decideUrl.replace(/\/$/, "")}/extract`
    : `${import.meta.env.BASE_URL}api/extract`;
  const ask = async (call: ExtractCall) => {
    const { fields } = call;
    const dated = Object.values(fields).some((f) => f.type === "date");
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        state: callState(call, prompt, today),
        fields,
        // The API rejects a reference date with no date field.
        ...(dated ? { reference_date: isoDay(today) } : {}),
      }),
    });
    if (!response.ok) {
      throw new Error(`extraction failed (${response.status}): ${await response.text()}`);
    }
    const body = (await response.json()) as { answers?: Record<string, ExtractAnswer> };
    return body.answers ?? {};
  };
  const answers: Record<string, ExtractAnswer> = Object.assign({}, ...(await Promise.all(EXTRACT_CALLS.map(ask))));
  const text = (a: ExtractAnswer | undefined) => {
    const v = valueOf(a);
    return typeof v === "string" && v.trim() ? v.trim() : null;
  };
  const cloud = valueOf(answers.cloud);
  return {
    place: text(answers.place),
    when: text(answers.when),
    range: rangeOfAnswers(answers, today) ?? null,
    cloud: typeof cloud === "number" && cloud < 100 ? cloud : null,
    answers,
  };
}
