import type { BBox2D } from "./options";

/**
 * Finds places by name with Nominatim, the OpenStreetMap geocoder.
 *
 * The public Nominatim server allows one request per second, so requests
 * wait for each other. For heavy use, change `SEARCH_URL` to another server
 * or to a geocoder with the same answers (for example Photon).
 */

const SEARCH_URL = "https://nominatim.openstreetmap.org/search";
/** The number of matches jev can pick from. */
const LIMIT = 5;
/** Wait this long between two requests, in milliseconds. */
const INTERVAL = 1100;
/** A point, such as a peak, gets a box this wide, in degrees. */
const MIN_SIZE = 0.2;

export type Place = {
  id: string;
  /** The name jev reads, e.g. "Lisboa, Portugal (city)". */
  label: string;
  bbox: BBox2D;
};

type Result = {
  osm_type: string;
  osm_id: number;
  display_name: string;
  addresstype?: string;
  type?: string;
  /** [south, north, west, east], as strings. */
  boundingbox: [string, string, string, string];
};

const cache = new Map<string, Promise<Place[]>>();
let queue: Promise<unknown> = Promise.resolve();

/**
 * Finds the places that match a name, best match first.
 * @param query A place name, e.g. "Lisbon".
 * @throws If Nominatim cannot answer.
 */
export function geocode(query: string): Promise<Place[]> {
  const key = query.toLowerCase();
  let places = cache.get(key);
  if (!places) {
    places = queue.then(() => search(query));
    // The next request starts at least INTERVAL after this one.
    queue = places
      .catch(() => {})
      .then(() => new Promise((resolve) => setTimeout(resolve, INTERVAL)));
    places.catch(() => cache.delete(key));
    cache.set(key, places);
  }
  return places;
}

async function search(query: string): Promise<Place[]> {
  const url = new URL(SEARCH_URL);
  url.searchParams.set("q", query);
  url.searchParams.set("format", "jsonv2");
  url.searchParams.set("limit", String(LIMIT));
  // English names, so jev reads them the same way as the prompt.
  url.searchParams.set("accept-language", "en");
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Nominatim: ${response.status}`);
  const results = (await response.json()) as Result[];
  return results.map((result) => {
    const bbox = toBbox(result.boundingbox);
    // The size tells jev the extent, e.g. that "Alps, Italy" is the whole
    // mountain range and not only its Italian part.
    const kind = result.addresstype ?? result.type;
    return {
      id: `${result.osm_type}/${result.osm_id}`,
      label: `${result.display_name} (${kind}, ${sizeOf(bbox)})`,
      bbox,
    };
  });
}

/** The size of a box in words, e.g. "about 900 × 550 km". */
function sizeOf([w, s, e, n]: BBox2D): string {
  const km = 111.32;
  const width = (e - w) * km * Math.cos((((s + n) / 2) * Math.PI) / 180);
  const height = (n - s) * km;
  const round = (v: number) =>
    v < 10 ? Math.max(1, Math.round(v)) : Math.round(v / 10 ** (Math.floor(Math.log10(v)) - 1)) * 10 ** (Math.floor(Math.log10(v)) - 1);
  return `about ${round(width)} × ${round(height)} km`;
}

function toBbox([south, north, west, east]: Result["boundingbox"]): BBox2D {
  const [s, n, w, e] = [south, north, west, east].map(Number) as BBox2D;
  const pad = (lo: number, hi: number): [number, number] => {
    const half = Math.max(0, MIN_SIZE - (hi - lo)) / 2;
    return [lo - half, hi + half];
  };
  const [padW, padE] = pad(w, e);
  const [padS, padN] = pad(s, n);
  return [padW, Math.max(-90, padS), padE, Math.min(90, padN)];
}

/** Small words that do not name a place or a dataset. */
export const STOPWORDS = new Set(
  (
    "a an and any are as at after before by for from in into is it last me " +
    "my near next of on or over show some than that the this to under what " +
    "when where with around above below between during since until"
  ).split(" "),
);

/** The longest span, in words, that can name a place. */
const MAX_SPAN = 4;

/**
 * Every span of up to four words of the prompt that could name a place, in
 * the order they appear. A span does not start or end with a small word such
 * as "the" or "over", so "over Lisbon last" is not a span but "Lisbon" is.
 */
export function placeSpans(prompt: string): string[] {
  const words = prompt
    .split(/\s+/)
    .map((word) => word.replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, ""))
    .filter(Boolean);
  const spans = new Set<string>();
  for (let start = 0; start < words.length; start++) {
    for (let end = start; end < Math.min(words.length, start + MAX_SPAN); end++) {
      const first = words[start]!.toLowerCase();
      const last = words[end]!.toLowerCase();
      if (STOPWORDS.has(first) || STOPWORDS.has(last)) continue;
      spans.add(words.slice(start, end + 1).join(" "));
    }
  }
  return [...spans];
}
