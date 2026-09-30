import type { BBox2D } from "./options";

/**
 * Finds places by name with Photon (https://photon.komoot.io), a geocoder for
 * OpenStreetMap data by komoot.
 *
 * The public Photon server has no fixed request limit, but asks for fair use.
 * Nominatim, the other public OpenStreetMap geocoder, allows one request per
 * second for all users together, which a public demo can easily pass. Photon
 * can also run on its own server; then change `SEARCH_URL`.
 */

const SEARCH_URL = "https://photon.komoot.io/api/";
/** The number of matches jev can pick from. */
const LIMIT = 5;
/** Wait this long between two requests from this page, in milliseconds. */
const INTERVAL = 300;
/** A point, such as a peak, gets a box this wide, in degrees. */
const MIN_SIZE = 0.2;

export type Place = {
  id: string;
  /** The name jev reads, e.g. "Lisbon, Portugal (city, about 17 × 22 km)". */
  label: string;
  bbox: BBox2D;
};

type Feature = {
  geometry: { coordinates: [number, number] };
  properties: {
    osm_type?: string;
    osm_id?: number;
    osm_value?: string;
    type?: string;
    name?: string;
    city?: string;
    county?: string;
    state?: string;
    country?: string;
    /** [west, north, east, south], for areas only. */
    extent?: [number, number, number, number];
  };
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
  url.searchParams.set("limit", String(LIMIT));
  // English names, so jev reads them the same way as the prompt.
  url.searchParams.set("lang", "en");
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Photon: ${response.status}`);
  const { features } = (await response.json()) as { features?: Feature[] };
  return (features ?? []).map((feature, index) => {
    const p = feature.properties;
    const bbox = toBbox(feature);
    // The names from the place up to its country, without repeats, e.g.
    // "Zurich, Zurich, Switzerland" becomes "Zurich, Switzerland". A large
    // place gets its own name only: OpenStreetMap gives the Alps the country
    // "Italy", and then jev reads it as the Italian Alps.
    const large = maxSizeKm(bbox) > LARGE_KM;
    const names = (large ? [p.name] : [p.name, p.city, p.county, p.state, p.country]).filter(
      (name, i, all): name is string => !!name && all.indexOf(name) === i,
    );
    // The size tells jev the extent, e.g. that "Alps" is the whole mountain
    // range and not one peak.
    const kind = p.osm_value ?? p.type ?? "place";
    return {
      id: p.osm_type && p.osm_id ? `${p.osm_type}/${p.osm_id}` : `photon/${index}`,
      label: `${names.join(", ") || query} (${kind}, ${sizeOf(bbox)})`,
      bbox,
    };
  });
}

/** A place wider or taller than this, in km, is labeled with its name only. */
const LARGE_KM = 300;
const KM_PER_DEGREE = 111.32;

/** The width and the height of a box, in km. */
function sizeKm([w, s, e, n]: BBox2D): [number, number] {
  const width = (e - w) * KM_PER_DEGREE * Math.cos((((s + n) / 2) * Math.PI) / 180);
  return [width, (n - s) * KM_PER_DEGREE];
}

function maxSizeKm(bbox: BBox2D): number {
  return Math.max(...sizeKm(bbox));
}

/** The size of a box in words, e.g. "about 900 × 550 km". */
function sizeOf(bbox: BBox2D): string {
  const [width, height] = sizeKm(bbox);
  const round = (v: number) =>
    v < 10 ? Math.max(1, Math.round(v)) : Math.round(v / 10 ** (Math.floor(Math.log10(v)) - 1)) * 10 ** (Math.floor(Math.log10(v)) - 1);
  return `about ${round(width)} × ${round(height)} km`;
}

/** The box of an area, or of a point with the minimum size. */
function toBbox({ geometry, properties }: Feature): BBox2D {
  const [lon, lat] = geometry.coordinates;
  const [w, n, e, s] = properties.extent ?? [lon, lat, lon, lat];
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
