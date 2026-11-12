/**
 * What `extract.ts` asks an extraction model: the state and the fields of
 * each group, in some variants.
 *
 * - `plain`: the state is the request only, and the instructions say only
 *   what each value is.
 * - `hinted`: the state is the state that jev reads (the app, today's date
 *   and the request), and the instructions give the knowledge that the
 *   option texts give jev: the months of each season and the cloud cover of
 *   words such as "clear sky". The hints come from the app
 *   (`SEASON_OPTIONS`, `CLOUD_OPTIONS`), not from the test prompts.
 * - `compact`: the model reads words and values, and code does the
 *   calendar. The state of jev and the wording of `hinted`, with no season
 *   text. Code makes the range of a season, a quarter, "the last N
 *   days/weeks/months/years", "recent" or "latest" from the `when` words
 *   (seasons and "recent" with `toRange`, as for a jev answer). A short
 *   cloud word map.
 * - `compact-today`: `compact` with a short state: today's date and the
 *   request, with no app context. It gave worse dates.
 * - `terse`: `compact-today` with short phrases ("A day: that day; a month: its
 *   1st"). Kept for the record: it gave worse dates than full sentences.
 *
 * The wording was tuned on `DEV_CASES` (`cases.ts`), not on the prompts of
 * the tests.
 */
import { buildPayload } from "../src/zap/decide";
import { CLOUD_OPTIONS } from "../src/zap/options";
import type { ExtractField } from "./providers";

export type Group = "place" | "time" | "cloud";
export const VARIANTS = ["plain", "hinted", "compact", "compact-today", "terse"] as const;
export type Variant = (typeof VARIANTS)[number];

/** Whether code makes the range of a season, "recent" or "latest" from the `when` words. */
export const calendarInCode = (variant: Variant) =>
  variant === "compact" || variant === "compact-today" || variant === "terse";

/** The state of a request: the request, today's date and the request, or the state that jev reads. */
export function stateOf(variant: Variant, prompt: string, today: Date): string {
  if (variant === "plain") return prompt;
  if (variant === "compact-today" || variant === "terse") {
    return `Today is ${today.toISOString().slice(0, 10)}.\n\n${prompt}`;
  }
  // buildPayload with no fields gives the context, the date and the request.
  return buildPayload(prompt, {}, "", today).state.replace(/\n\nCurrent map:\n$/, "");
}

const PLACE =
  "The words of the request that name the place to search. A place can be " +
  "a city, a region, a country, a landmark or a natural feature, such as " +
  "a mountain range, a river or a desert. The name of an event, e.g. " +
  "'Hurricane Idalia', or of a dataset is not a place.";

const CLOUD =
  "The maximum cloud cover the request asks for. Only a request that " +
  "mentions clouds, cloud cover, a clear sky or cloud-free imagery asks " +
  "for one.";

/** "'cloud-free' is at most 5%", ... from the option texts of the app. */
const CLOUD_WORDS = Object.entries(CLOUD_OPTIONS)
  .filter(([id, text]) => id !== "any" && text.includes(":"))
  .map(([id, text]) => `'${text.split(":")[0]!.toLowerCase()}' is at most ${id}%`)
  .join(", ");

/** "cloud-free 5, clear sky 10, ...": the same words, shorter. */
const CLOUD_MAP = Object.entries(CLOUD_OPTIONS)
  .filter(([id, text]) => id !== "any" && text.includes(":"))
  .map(([id, text]) => `${text.split(":")[0]!.toLowerCase()} ${id}`)
  .join(", ");

const SEASONS =
  "Spring is March to May, summer is June to August, autumn or fall is " +
  "September to November, and winter is December of the year before to " +
  "February.";

// With today's date in the state, the model gives a date range for requests
// with no time, e.g. "orthophotos of Bern". A text field for the words of
// the time tells if there is one, as the place field does. "Recent" and
// "latest" come from the period options of the app.
const WHEN: ExtractField = {
  type: "text",
  instructions:
    "The words of the request that name a time, a date or a period, " +
    "e.g. 'in 2021', 'last summer', 'recent', 'latest' or 'since May'. " +
    "A dataset name or a place is not a time.",
};

export function groupsOf(variant: Variant): Record<Group, Record<string, ExtractField>> {
  if (variant === "plain") {
    return {
      place: { place: { type: "text", instructions: `${PLACE} 'Here' or 'this area' is the area the map shows now.` } },
      time: {
        start: {
          type: "date",
          instructions:
            "The first day of the time period the request asks for. For a month, " +
            "the 1st of that month; for a year, 1 January.",
        },
        end: {
          type: "date",
          instructions:
            "The last day of the time period the request asks for. For a month, " +
            "the last day of that month; for a year, 31 December.",
        },
      },
      cloud: { cloud: { type: "number", unit: "percent", instructions: CLOUD } },
    };
  }
  const place = {
    place: {
      type: "text",
      instructions: `${PLACE} If the request says 'here', 'this area' or 'the map', give those words.`,
    },
  } as const;
  if (variant === "hinted") {
    // The date is in the state and in the reference date. A "Today is" in the
    // instructions too made the model put today's date into other answers.
    return {
      place,
      time: {
        when: WHEN,
        start: {
          type: "date",
          instructions:
            "The first day of the time period the request asks for. For a day, " +
            "that day; for a month, the 1st of that month; for a year, " +
            `1 January. ${SEASONS}`,
        },
        end: {
          type: "date",
          instructions:
            "The last day of the time period the request asks for. For a day, " +
            "that day; for a month, the last day of that month; for a year, " +
            `31 December. ${SEASONS} If the period runs up to now, the reference date.`,
        },
      },
      cloud: {
        cloud: { type: "number", unit: "percent", instructions: `${CLOUD} ${CLOUD_WORDS}.` },
      },
    };
  }
  // The compact variants. A season text on each date also moved months that
  // are not in a season, e.g. "June" to June to August, so code does seasons.
  const terse = variant === "terse";
  return {
    place,
    time: {
      when: WHEN,
      start: {
        type: "date",
        instructions: terse
          ? "The first day of the period. A day: that day; a month: its 1st; a year: 1 January."
          : "The first day of the time period the request asks for. For a day, " +
            "that day; for a month, the 1st of that month; for a year, 1 January.",
      },
      end: {
        type: "date",
        instructions: terse
          ? "The last day of the period. A day: that day; a month: its last day; a year: 31 December. " +
            "If the period runs up to now: the reference date."
          : "The last day of the time period the request asks for. For a day, " +
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
  };
}
