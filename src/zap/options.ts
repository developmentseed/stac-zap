/** [west, south, east, north], as stac-map stores it. */
export type BBox2D = [number, number, number, number];

/** The `place` option for the area the map shows now. */
export const MAP_VIEW = "map-view";

/** The `place` option for "search everywhere", e.g. for an event or a dataset. */
export const NO_PLACE = "no-place";

const MONTHS = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
];

/** Periods a prompt can pick: a year, or a span back from today. */
export function periodOptions(today: Date = new Date()): Record<string, string> {
  const options: Record<string, string> = {
    "last-30-days": "Recent: the last 30 days, the latest imagery",
    "last-12-months": "The last 12 months",
    "all-time": "Any time: the whole record of the collection",
  };
  for (let year = today.getUTCFullYear(); year >= 2015; year--) {
    options[String(year)] = `The whole year ${year}`;
  }
  return options;
}

/** Parts of a year a prompt can pick, used with a year period. */
export const SEASON_OPTIONS: Record<string, string> = {
  "whole-year": "The whole year",
  spring: "Spring (March to May)",
  summer: "Summer (June to August)",
  autumn: "Autumn or fall (September to November)",
  winter: "Winter (December of the year before to February)",
  ...Object.fromEntries(MONTHS.map((month, i) => [`m${i + 1}`, month])),
};

/**
 * Maximum cloud cover options, in percent, from the strictest to no filter.
 * {@link cloudChoice} depends on this order.
 */
export const CLOUD_OPTIONS: Record<string, string> = {
  "5": "Cloud-free: at most 5% cloud cover",
  "10": "Clear sky: at most 10% cloud cover",
  "15": "At most 15% cloud cover",
  "20": "Few clouds: at most 20% cloud cover",
  "25": "At most 25% cloud cover",
  "30": "Some clouds: at most 30% cloud cover",
  "40": "At most 40% cloud cover",
  "50": "Partly cloudy: at most 50% cloud cover",
  "75": "Cloudy: at most 75% cloud cover",
  any: "Any cloud cover, no filter",
};

/**
 * The cloud cover question. With the default question ("Pick 'keep' only if
 * the request has nothing to do with the maximum cloud cover"), jev reads any
 * request for optical imagery as a request about clouds, e.g. "Landsat over
 * Denver", and spreads its answer over the thresholds. Then the median of
 * {@link cloudChoice} adds a filter that the user did not ask for.
 */
export const CLOUD_QUESTION =
  "Which maximum cloud cover does the request ask for? Only a request that " +
  "mentions clouds, cloud cover, a clear sky or cloud-free imagery asks for " +
  "one.";

/** The cloud cover options as the answers panel shows them: the filter used. */
export const CLOUD_SHORT: Record<string, string> = Object.fromEntries(
  Object.keys(CLOUD_OPTIONS).map((id) => [
    id,
    id === "any" ? "no filter" : `≤ ${id}%`,
  ]),
);

/**
 * The cloud cover option from the jev probabilities: the median of the
 * options, from the strictest to no filter. With many options, the
 * probability spreads over the neighbours, e.g. "below 20%" can give 0.45 to
 * 20 and 0.3 to 10. Then no option gets 0.5, but the median is still a good
 * answer.
 * @param probabilities Option id to probability, with `keep`.
 * @param keep The id of the "keep" option.
 * @returns The option id, or null when `keep` has half of the probability or
 *   more.
 */
export function cloudChoice(
  probabilities: Record<string, number>,
  keep: string,
): string | null {
  const keepP = probabilities[keep] ?? 0;
  const options = Object.keys(CLOUD_OPTIONS);
  const total = options.reduce((sum, id) => sum + (probabilities[id] ?? 0), 0);
  if (total === 0 || keepP >= 0.5) return null;
  let cumulative = 0;
  for (const id of options) {
    cumulative += probabilities[id] ?? 0;
    if (cumulative >= total / 2) return id;
  }
  return null;
}

/** A time range in stac-map's datetime input format (UTC, no zone). */
export type Range = { startDatetime: string; endDatetime: string };

const input = (date: Date) => date.toISOString().slice(0, 19);
const utc = (year: number, month: number, day = 1) =>
  new Date(Date.UTC(year, month, day));

/**
 * Turns a period and a season into a time range.
 * @param period A {@link periodOptions} id.
 * @param season A {@link SEASON_OPTIONS} id. Only used with a year period.
 * @returns The range, or null for the whole record.
 */
export function toRange(
  period: string,
  season: string,
  today: Date = new Date(),
): Range | null {
  if (period === "all-time") return null;
  if (period === "last-30-days" || period === "last-12-months") {
    const start = new Date(today);
    if (period === "last-30-days") start.setUTCDate(start.getUTCDate() - 30);
    else start.setUTCFullYear(start.getUTCFullYear() - 1);
    return { startDatetime: input(start), endDatetime: input(today) };
  }
  const year = Number(period);
  const span = (from: Date, to: Date) => ({
    startDatetime: input(from),
    // One second before the next period starts.
    endDatetime: input(new Date(to.getTime() - 1000)),
  });
  switch (season) {
    case "spring":
      return span(utc(year, 2), utc(year, 5));
    case "summer":
      return span(utc(year, 5), utc(year, 8));
    case "autumn":
      return span(utc(year, 8), utc(year, 11));
    case "winter":
      return span(utc(year - 1, 11), utc(year, 2));
    default: {
      const month = /^m(\d+)$/.exec(season);
      if (month) {
        const m = Number(month[1]) - 1;
        return span(utc(year, m), utc(year, m + 1));
      }
      return span(utc(year, 0), utc(year + 1, 0));
    }
  }
}
