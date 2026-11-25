/**
 * The request generator: makes requests with exact labels, from the STAC
 * snapshot (`training/snapshot.ts`) and the place list (`training/places.py`).
 *
 * Each record is one request with what the app must read from it:
 * - the catalog and the collection (the target, and the other catalogs and
 *   collections with the same data, which are also correct),
 * - the place words (a span of the request), the place they mean, and the
 *   other places with the same name, for the place pick,
 * - the time words (a span) and the day range,
 * - the cloud cover words (a span) and the maximum in percent,
 * - for a follow-up, the current map, so `keep` can be learned.
 *
 * The labels come from the generator, so they are exact. Phase 2 adds
 * paraphrases and topic requests from open-weight models; they keep these
 * labels. The templates and word lists in this file are code of the project.
 *
 * Output: training/data/records.jsonl.
 * Usage: pnpm tsx training/generate.ts [records per collection, default 12]
 */
import { readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { quarterRange, recentRange, relativeRange, seasonRange, type DayRange } from "../src/zap/extract";
import { CASES, DEV_CASES, UNPARSED_CASES } from "../eval/cases";

// ---------------------------------------------------------------- inputs

type Collection = {
  id: string;
  bbox?: number[];
  title?: string;
  description?: string;
  keywords?: string[];
};
type Catalog = { id: string; href: string; description: string; curated: boolean };
type Snapshot = { date: string; catalogs: Catalog[]; collections: Record<string, Collection[]> };
type Place = {
  id: string;
  name: string;
  kind: "city" | "feature";
  type: string;
  country: string | null;
  region: string | null;
  population: number;
  lat: number;
  lon: number;
};

const DATA = new URL("data/", import.meta.url);
const snapshot: Snapshot = JSON.parse(readFileSync(new URL("stac-full.json", DATA), "utf-8"));
const places: Place[] = (JSON.parse(readFileSync(new URL("places.json", DATA), "utf-8")) as Place[])
  .map((p) => ({
    ...p,
    name: p.name.replace(/\s+/g, " ").trim(),
    country: p.country?.trim() || null,
    region: p.region?.trim() || null,
  }))
  // Natural Earth abbreviations such as "S. Fk. Kentucky" are not how people write.
  .filter((p) => !/\b\p{L}{1,3}\.\s/u.test(p.name));
const PER_COLLECTION = Number(process.argv[2] ?? 12);

/**
 * Catalogs that no training record uses. The test split measures how the
 * models do on catalogs and collections they have never seen.
 */
export const HELD_OUT_CATALOGS = ["kagis-katalog", "digital-earth-au", "worldpop-stac-api"];

// ---------------------------------------------------------------- random

let seed = 20261125;
function random(): number {
  // mulberry32
  seed = (seed + 0x6d2b79f5) | 0;
  let t = seed;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
const pick = <T>(xs: readonly T[]): T => xs[Math.floor(random() * xs.length)]!;
const chance = (p: number) => random() < p;
function weighted<T>(xs: readonly T[], weight: (x: T) => number): T {
  const total = xs.reduce((s, x) => s + weight(x), 0);
  let r = random() * total;
  for (const x of xs) if ((r -= weight(x)) <= 0) return x;
  return xs.at(-1)!;
}

// ---------------------------------------------------------------- collections

/** Optical data, where a cloud cover filter makes sense. */
const OPTICAL = /sentinel-?2|landsat|naip|optical|multispectral|modis|aster|hls|orthophoto|aerial|planet|spot|imagery|rgb/i;

/**
 * Generic names for common data, and the main products they mean. A request
 * with a generic name ("Sentinel-2", "elevation") accepts every member of
 * the group, in every catalog. Other collections are named by their own
 * title only. (v0 gave the generic names to every collection whose id or
 * title matched a pattern, so "Sentinel-2" pointed to dozens of mosaics and
 * bands with one correct answer each, and "elevation" to Swiss geology.)
 */
type Group = { names: string[]; members: string[] };
export const GROUPS: Record<string, Group> = {
  s2: {
    names: ["Sentinel-2", "S2", "Sentinel 2", "sentinel2", "Sentinel-2 L2A", "Sentinel-2 imagery", "S2 images"],
    members: [
      "planetary-computer/sentinel-2-l2a", "earth-search/sentinel-2-l2a", "earth-search/sentinel-2-c1-l2a",
      "cdse/sentinel-2-l2a", "astraea-earth-ondemand/sentinel2_l2a", "super-resolved-sentinel-2-2-5m-stac-api/sentinel-2-l2a",
      "digital-earth-africa/s2_l2a", "digital-earth-africa/s2_l2a_c1", "inpe-stac-server/S2_L2A-1",
    ],
  },
  s2l1c: {
    names: ["Sentinel-2 L1C", "Sentinel-2 Level-1C", "S2 L1C", "Sentinel-2 top of atmosphere"],
    members: ["earth-search/sentinel-2-l1c", "cdse/sentinel-2-l1c", "astraea-earth-ondemand/sentinel2_l1c", "dedl-api/EO.ESA.DAT.SENTINEL-2.MSI.L1C"],
  },
  s1: {
    names: ["Sentinel-1", "S1", "Sentinel-1 radar", "SAR", "radar images", "radar"],
    members: [
      "planetary-computer/sentinel-1-rtc", "planetary-computer/sentinel-1-grd", "earth-search/sentinel-1-grd", "cdse/sentinel-1-grd",
      "astraea-earth-ondemand/sentinel1_l1c_grd", "dedl-api/EO.ESA.DAT.SENTINEL-1.L1_GRD", "mtd-stac-api/sentinel1-rtc",
      "digital-earth-africa/s1_rtc", "inpe-stac-server/sentinel-1-rtc-1",
    ],
  },
  landsat: {
    names: ["Landsat", "Landsat imagery", "landsat images", "Landsat surface reflectance"],
    members: [
      "planetary-computer/landsat-c2-l2", "earth-search/landsat-c2-l2", "usgs-landsat-collection-2-api/landsat-c2l2-sr",
      "dedl-api/EO.NASA.DAT.LANDSAT.C2_L2", "digital-earth-africa/landsat_c2l2_ar",
    ],
  },
  naip: {
    names: ["NAIP", "NAIP imagery", "NAIP aerial photos"],
    members: ["planetary-computer/naip", "earth-search/naip", "astraea-earth-ondemand/naip"],
  },
  dem: {
    names: ["elevation", "DEM", "a digital elevation model", "terrain height", "topography"],
    members: [
      "planetary-computer/cop-dem-glo-30", "planetary-computer/cop-dem-glo-90", "planetary-computer/nasadem", "planetary-computer/alos-dem",
      "earth-search/cop-dem-glo-30", "earth-search/cop-dem-glo-90", "eoapi/copernicus-dem", "openaerialmap/cop-dem-glo-30",
      "cdse/cop-dem-glo-30-dged-cog", "cdse/cop-dem-glo-90-dged-cog", "dedl-api/EO.DEM.DAT.COP-DEM_GLO-30-DGED",
      "dedl-api/EO.DEM.DAT.COP-DEM_GLO-90-DGED", "digital-earth-africa/dem_cop_30", "digital-earth-africa/dem_cop_90",
      "digital-earth-africa/nasadem", "digital-earth-africa/dem_srtm", "mtd-stac-api/srtm",
    ],
  },
  landcover: {
    names: ["land cover", "a land cover map", "land use and land cover"],
    members: [
      "planetary-computer/esa-worldcover", "planetary-computer/esa-cci-lc", "digital-earth-africa/esa_worldcover_2021",
      "digital-earth-africa/io_lulc_v2", "digital-earth-africa/cgls_landcover", "digital-earth-africa/cci_landcover",
    ],
  },
};
const groupOf = new Map<string, Group>();
for (const g of Object.values(GROUPS)) for (const m of g.members) groupOf.set(m, g);

/** The ways a request can name a collection. */
function namesOf(c: Collection): string[] {
  // The generic names of a group are added where the collection is used (see `accepted` below).
  const title = (c.title ?? c.id).replace(/\s*\([^)]*\)\s*/g, " ").replace(/\s+/g, " ").trim();
  const names = new Set([title]);
  if (title.length > 50) names.delete(title);
  const short = title.replace(/\b(collection|dataset|data|product|v\d+(\.\d+)*)\b/gi, "").replace(/\s+/g, " ").trim();
  if (short.length >= 3 && short.length <= 50) names.add(short);
  for (const k of c.keywords ?? []) if (k.length >= 6 && k.length <= 30 && chance(0.15)) names.add(k);
  for (const n of names) if (ID_LIKE.test(n) && /[._]/.test(n) && names.size > 1) names.delete(n);
  if (names.size === 0) names.add(c.id.replace(/[_-]+/g, " "));
  return [...names];
}

/** Collections with the same data in other catalogs, by a normalized title. */
const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, "");
const sameData = new Map<string, { catalog: string; collection: string }[]>();
for (const [catalog, list] of Object.entries(snapshot.collections)) {
  for (const c of list) {
    for (const key of new Set([norm(c.id), norm(c.title ?? c.id)])) {
      if (key.length < 4) continue;
      sameData.set(key, [...(sameData.get(key) ?? []), { catalog, collection: c.id }]);
    }
  }
}

// ---------------------------------------------------------------- places

const byName = new Map<string, Place[]>();
for (const p of places) byName.set(p.name, [...(byName.get(p.name) ?? []), p]);
for (const list of byName.values()) list.sort((a, b) => b.population - a.population);
const cities = places.filter((p) => p.kind === "city");
const features = places.filter((p) => p.kind === "feature");

const FEATURE_ARTICLE = /^(river|sea|gulf|bay|desert|strait|range\/mtn|island group|plateau|delta|sound|basin)$/;

/** The words of a place, and the place they mean. */
function placePhrase(pool?: { cities: Place[]; features: Place[] }): { words: string; target: Place; candidates: Place[] } {
  const cs = pool?.cities.length ? pool.cities : cities;
  const fs = pool?.features.length ? pool.features : features;
  const target = chance(pool && !pool.features.length ? 1 : pool && !pool.cities.length ? 0 : 0.75)
    ? weighted(cs, (p) => Math.log10(Math.max(p.population, 15000)) - 3.5)
    : pick(fs);
  const same = byName.get(target.name) ?? [target];
  let words = target.name;
  // A qualifier ("Paris, Texas") makes a place that is not the largest of its name clear.
  const needsQualifier = same[0] !== target;
  if (target.kind === "city" && (needsQualifier || chance(0.2))) {
    const q = target.region && chance(0.5) ? target.region : target.country;
    if (q && q !== target.name) words = `${target.name}${chance(0.7) ? "," : ""} ${q}`;
  }
  if (target.kind === "feature" && FEATURE_ARTICLE.test(target.type) && !/^the\b/i.test(words) && chance(0.6)) {
    words = `the ${words}`;
  }
  // Other places for the pick: the same name first, then a few others.
  const others = places.length ? Array.from({ length: 3 }, () => pick(places)) : [];
  const candidates = [...new Map([...same.slice(0, 6), ...others].map((p) => [p.id, p])).values()];
  return { words, target, candidates };
}

/**
 * The places in the extent of a collection, or undefined for a global (or
 * unknown) extent. A request names a place where the data is.
 */
type BBox = [number, number, number, number];

/** The extent of a collection, or null when it is missing or covers most of the world. */
function regional(b: number[] | undefined): BBox | null {
  if (!b || b.length < 4 || b.some((v) => typeof v !== "number" || !Number.isFinite(v))) return null;
  const box = (b.length === 6 ? [b[0], b[1], b[3], b[4]] : b.slice(0, 4)) as BBox;
  return (box[2] - box[0]) * (box[3] - box[1]) > 0.5 * 360 * 180 ? null : box;
}

const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)]!;

/**
 * The extent of a catalog: the median of the regional extents of its
 * collections, when most of them are regional. Some APIs give a collection
 * a world extent although the data is regional (for example the Swiss
 * federal data); the catalog extent is used for those.
 */
const catalogExtent = new Map<string, BBox>();
for (const [catalog, list] of Object.entries(snapshot.collections)) {
  const boxes = list.map((c) => regional(c.bbox)).filter((b): b is BBox => b !== null);
  if (boxes.length >= 0.5 * list.length && boxes.length > 0) {
    catalogExtent.set(catalog, [0, 1, 2, 3].map((i) => median(boxes.map((b) => b[i]!))) as BBox);
  }
}

function poolOf(c: Collection, catalog: string): { cities: Place[]; features: Place[] } | undefined {
  const box = catalogExtent.has(catalog) ? (regional(c.bbox) ?? catalogExtent.get(catalog)!) : regional(c.bbox);
  if (!box) return undefined;
  const [w, s, e, n] = box;
  const inside = (p: Place) => p.lat >= s && p.lat <= n && (w <= e ? p.lon >= w && p.lon <= e : p.lon >= w || p.lon <= e);
  return { cities: cities.filter(inside), features: features.filter(inside) };
}

/** Titles that look like ids, e.g. "eo.eum.dat.mult.mhsmwhs0100": nobody types them. */
const ID_LIKE = /^[\w.:-]+$/;

// ---------------------------------------------------------------- time

const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
const MON = MONTHS.map((m) => m.slice(0, 3));
const iso = (y: number, m: number, d: number) => new Date(Date.UTC(y, m - 1, d)).toISOString().slice(0, 10);
const lastDay = (y: number, m: number) => new Date(Date.UTC(y, m, 0)).getUTCDate();
const year = (today: Date) => 2015 + Math.floor(random() * (today.getUTCFullYear() - 2015 + 1));

/** Time words and their day range. Null range: the request asks for no time. */
function timePhrase(today: Date): { words: string; range: DayRange } {
  const T = iso(today.getUTCFullYear(), today.getUTCMonth() + 1, today.getUTCDate());
  const y = year(today);
  const m = 1 + Math.floor(random() * 12);
  const month = chance(0.85) ? MONTHS[m - 1]! : MON[m - 1]!;
  const kinds: [number, () => { words: string; range: DayRange }][] = [
    [5, () => ({ words: pick([`in ${y}`, `${y}`, `from ${y}`, `during ${y}`, `for ${y}`]), range: [iso(y, 1, 1), iso(y, 12, 31)] })],
    [5, () => ({ words: pick([`in ${month} ${y}`, `${month} ${y}`, `from ${month} ${y}`, `${month.toLowerCase()} ${y}`]), range: [iso(y, m, 1), iso(y, m, lastDay(y, m))] })],
    [2, () => {
      const d = 1 + Math.floor(random() * lastDay(y, m));
      return { words: pick([`on ${d} ${month} ${y}`, `on ${month} ${d}, ${y}`, `${iso(y, m, d)}`]), range: [iso(y, m, d), iso(y, m, d)] };
    }],
    [2, () => {
      const y2 = Math.min(y + 1 + Math.floor(random() * 3), today.getUTCFullYear());
      return { words: pick([`from ${y} to ${y2}`, `between ${y} and ${y2}`, `${y}-${y2}`]), range: [iso(y, 1, 1), iso(y2, 12, 31)] };
    }],
    [2, () => {
      const m2 = Math.min(m + 1 + Math.floor(random() * 5), 12);
      if (m2 <= m) return { words: `in ${MONTHS[m - 1]} ${y}`, range: [iso(y, m, 1), iso(y, m, lastDay(y, m))] };
      return {
        words: pick([`from ${MONTHS[m - 1]} to ${MONTHS[m2 - 1]} ${y}`, `between ${MONTHS[m - 1]} and ${MONTHS[m2 - 1]} ${y}`]),
        range: [iso(y, m, 1), iso(y, m2, lastDay(y, m2))],
      };
    }],
    [2, () => (chance(0.5)
      ? { words: pick([`since ${y}`, `after ${y - 1}`, `from ${y} until now`, `from ${y} onwards`]), range: [iso(y, 1, 1), T] }
      : { words: `since ${MONTHS[m - 1]} ${y}`, range: [iso(y, m, 1), T] })],
    [3, () => {
      const season = pick(["spring", "summer", "autumn", "fall", "winter"]);
      const words = pick([`${season} ${y}`, `in the ${season} of ${y}`, `last ${season}`, `${season} ${y}`]);
      return { words, range: seasonRange(words, today)! };
    }],
    [1, () => {
      const q = 1 + Math.floor(random() * 4);
      const words = pick([`Q${q} ${y}`, `in the ${["first", "second", "third", "fourth"][q - 1]} quarter of ${y}`]);
      return { words, range: quarterRange(words, today)! };
    }],
    [2, () => {
      const words = pick(["in the last 3 months", "over the past week", "in the past 10 days", "in the last two weeks", "over the last 6 months", "in the last 2 years"]);
      return { words, range: relativeRange(words, today)! };
    }],
    [2, () => {
      const words = pick(["recent", "latest", "the latest", "most recent"]);
      return { words, range: recentRange(words, today)! };
    }],
  ];
  const [, make] = weighted(kinds, ([w]) => w);
  return make();
}

// ---------------------------------------------------------------- cloud

const CLOUD_WORDS: [string, number][] = [
  ["cloud-free", 5], ["cloudless", 5], ["no clouds", 5], ["clear sky", 10], ["clear skies", 10],
  ["few clouds", 20], ["some clouds", 30], ["partly cloudy", 50], ["mostly clear", 15],
];

/** Cloud words and the maximum in percent. Null value: any cloud cover. */
function cloudPhrase(): { words: string; value: number | null } {
  if (chance(0.1)) return { words: pick(["any clouds", "any cloud cover", "clouds are fine"]), value: null };
  if (chance(0.45)) {
    const [w, v] = pick(CLOUD_WORDS);
    return { words: pick([w, `with ${w}`]), value: v };
  }
  const n = pick([1, 2, 3, 5, 5, 8, 10, 10, 12, 15, 20, 20, 25, 30, 35, 40, 50, 60, 70, 80]);
  return {
    words: pick([
      `under ${n}% clouds`, `less than ${n}% cloud cover`, `with at most ${n}% clouds`, `max ${n}% cloud`,
      `below ${n}% cloud cover`, `cloud cover < ${n}%`, `<${n}% clouds`, `up to ${n}% clouds`, `${n}% cloud cover or less`,
    ]),
    value: n,
  };
}

// ---------------------------------------------------------------- requests

type Span = [number, number];
export type RecordOut = {
  id: string;
  split: "train" | "dev" | "test-catalog" | "test-human" | "test-claude";
  request: string;
  /** The reference date of the request, for "last summer" or "recent". */
  today: string;
  /** The map before the request. Null fields: not set. */
  current: { catalog: string | null; collection: string | null; place: string | null };
  catalog: { target: string | null; accepted: string[] };
  collection: { target: string | null; accepted: string[] };
  place: { span: Span | null; words: string | null; map_view: boolean; target: string | null; candidates: string[] };
  time: { span: Span | null; range: DayRange | null };
  cloud: { span: Span | null; value: number | null };
  template: string;
  /** "template", "human:<writer>", or later "<model>@<provider>" for open-model text. */
  source: string;
};

/** Fills a template. Slots are {data}, {place}, {time} and {cloud}; spans are found after the fill. */
function fill(template: string, slots: Record<string, string | null>) {
  let text = "";
  const spans: Record<string, Span> = {};
  for (const part of template.split(/(\{\w+\})/)) {
    const slot = /^\{(\w+)\}$/.exec(part)?.[1];
    if (!slot) {
      text += part;
      continue;
    }
    const value = slots[slot];
    if (!value) continue;
    spans[slot] = [text.length, text.length + value.length];
    text += value;
  }
  // Remove spaces left by empty slots, and move the spans with the text.
  const out: string[] = [];
  const map: number[] = [];
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]!;
    const prev = out.at(-1);
    if (ch === " " && (prev === undefined || prev === " ")) {
      map.push(out.length);
      continue;
    }
    if ((ch === "," || ch === ".") && prev === " ") {
      out.pop();
    }
    map.push(out.length);
    out.push(ch);
  }
  map.push(out.length);
  let request = out.join("").trim().replace(/,\s*$/, "");
  const lead = out.join("").length - out.join("").trimStart().length;
  const moved = Object.fromEntries(
    Object.entries(spans).map(([k, [s, e]]) => [k, [map[s]! - lead, map[e]! - lead] as Span]),
  );
  // Lower case keeps the spans only when it keeps the length (not for "İ").
  if (chance(0.15) && request.toLowerCase().length === request.length) request = request.toLowerCase();
  return { request, spans: moved };
}

const NEW_TEMPLATES = [
  "{data} over {place} {time} {cloud}",
  "{data} of {place} {time}, {cloud}",
  "{data} for {place} {time} {cloud}",
  "show me {data} of {place} {time} {cloud}",
  "I need {data} images of {place} {time} {cloud}",
  "find {data} over {place} {time} {cloud}",
  "{place} {data} {time} {cloud}",
  "{time} {data} of {place} {cloud}",
  "{data} {place} {time}",
  "can you get {data} for {place} {time} with {cloud}",
  "{data} near {place} {time} {cloud}",
  "{data} around {place} {time}",
  "{data} {time} {cloud}",
  "{data} of {place}",
  "{data}",
];
const MAP_VIEW_WORDS = ["here", "this area", "the map view", "this region", "the current view"];
const MAP_TEMPLATES = ["{data} of {place} {time} {cloud}", "{data} {place} {time}", "show {data} for {place} {cloud}"];
/** Follow-ups that change some fields and keep the others. */
const FOLLOW_TEMPLATES: [string, ("data" | "place" | "time" | "cloud")[]][] = [
  ["now {data}", ["data"]],
  ["same for {place}", ["place"]],
  ["and {time}", ["time"]],
  ["now {time} {cloud}", ["time", "cloud"]],
  ["switch to {data}", ["data"]],
  ["{data} instead", ["data"]],
  ["move to {place}", ["place"]],
  ["only {cloud}", ["cloud"]],
  ["what about {place} {time}", ["place", "time"]],
];

const evalPrompts = new Set([...CASES, ...DEV_CASES, ...UNPARSED_CASES].map((c) => norm(c.prompt)));
const today0 = new Date("2026-11-25T12:00:00Z");
const records: RecordOut[] = [];
const seen = new Set<string>();

function splitOf(catalog: string, request: string): RecordOut["split"] {
  if (HELD_OUT_CATALOGS.includes(catalog)) return "test-catalog";
  return createHash("sha1").update(request).digest()[0]! < 13 ? "dev" : "train";
}

const allCollections = Object.entries(snapshot.collections).flatMap(([catalog, list]) =>
  list.map((c) => ({ catalog, c })),
);

for (const { catalog, c } of allCollections) {
  const names = namesOf(c);
  // Common data gets more requests: people ask for it more often.
  const perCollection = groupOf.has(`${catalog}/${c.id}`) ? PER_COLLECTION * 6 : PER_COLLECTION;
  const pool = poolOf(c, catalog);
  // A regional collection with no known place in its extent gets no place.
  const noPlaces = pool && !pool.cities.length && !pool.features.length;
  const optical = OPTICAL.test(`${c.id} ${c.title ?? ""} ${(c.description ?? "").slice(0, 300)}`);
  const key = [norm(c.id), norm(c.title ?? c.id)].find((k) => sameData.has(k));
  const equivalents = key ? sameData.get(key)! : [{ catalog, collection: c.id }];
  // Equivalents in held-out catalogs are not offered to training rows, and the reverse.
  const accepted = equivalents.filter((e) => HELD_OUT_CATALOGS.includes(e.catalog) === HELD_OUT_CATALOGS.includes(catalog));
  for (let n = 0; n < perCollection; n++) {
    // The reference date moves, so "last summer" does not stick to one year.
    const today = new Date(today0.getTime() - Math.floor(random() * 700) * 86400_000);
    const follow = chance(0.15);
    const place = placePhrase(pool);
    const mapView = !follow && chance(0.05);
    const time = chance(follow ? 0.5 : 0.55) ? timePhrase(today) : null;
    const cloud = optical && chance(0.3) ? cloudPhrase() : null;
    // A member of a group is named by a generic name half of the time; then the whole group is correct.
    const group = groupOf.get(`${catalog}/${c.id}`);
    const generic = group && chance(0.5);
    const data = generic ? pick(group.names) : pick(names);
    const ok = generic
      ? group.members
          .map((m) => ({ catalog: m.split("/")[0]!, collection: m.split("/").slice(1).join("/") }))
          .filter((e) => snapshot.collections[e.catalog]?.some((x) => x.id === e.collection))
          .filter((e) => HELD_OUT_CATALOGS.includes(e.catalog) === HELD_OUT_CATALOGS.includes(catalog))
      : accepted;
    let template: string;
    let changes: Set<string>;
    if (follow) {
      const [t, fields] = pick(FOLLOW_TEMPLATES);
      template = t;
      changes = new Set(fields);
    } else {
      template = mapView ? pick(MAP_TEMPLATES) : pick(NEW_TEMPLATES);
      if (!cloud) template = template.replace("with {cloud}", "{cloud}");
      changes = new Set(["data", "place", "time", "cloud"]);
    }
    const has = (slot: string) => changes.has(slot) && template.includes(`{${slot}}`);
    const slots: Record<string, string | null> = {
      data: has("data") ? data : null,
      place: has("place") && !noPlaces ? (mapView ? pick(MAP_VIEW_WORDS) : place.words) : null,
      time: has("time") ? (time?.words ?? null) : null,
      cloud: has("cloud") ? (cloud?.words ?? null) : null,
    };
    if (follow && !Object.values(slots).some(Boolean)) continue;
    // An empty slot takes its preposition with it ("over {place}").
    for (const [slot, value] of Object.entries(slots)) {
      if (!value) template = template.replace(new RegExp(`\\b(?:over|of|for|near|around|with)\\s+\\{${slot}\\}`), `{${slot}}`);
    }
    const { request, spans } = fill(template, slots);
    if (request.length < 3 || seen.has(norm(request)) || evalPrompts.has(norm(request))) continue;
    seen.add(norm(request));
    // A follow-up starts from another collection of the same catalog, or this one.
    const others = snapshot.collections[catalog]!;
    const current = follow
      ? {
          catalog,
          collection: changes.has("data") ? pick(others).id : c.id,
          place: chance(0.8) ? placePhrase(pool).target.name : null,
        }
      : { catalog: null, collection: null, place: null };
    const changesData = Boolean(slots.data);
    const placeWords = slots.place;
    records.push({
      id: createHash("sha1").update(`${catalog}/${c.id}/${request}`).digest("hex").slice(0, 12),
      split: splitOf(catalog, request),
      request,
      today: today.toISOString().slice(0, 10),
      current,
      catalog: changesData
        ? { target: catalog, accepted: [...new Set(ok.map((e) => e.catalog))] }
        : { target: null, accepted: [] },
      collection: changesData
        ? { target: c.id, accepted: ok.filter((e) => e.catalog === catalog).map((e) => e.collection) }
        : { target: null, accepted: [] },
      place: placeWords
        ? {
            span: spans.place ?? null,
            words: placeWords,
            map_view: mapView,
            target: mapView ? null : place.target.id,
            candidates: mapView ? [] : place.candidates.map((p) => p.id),
          }
        : { span: null, words: null, map_view: false, target: null, candidates: [] },
      time: slots.time ? { span: spans.time ?? null, range: time!.range } : { span: null, range: null },
      cloud: slots.cloud ? { span: spans.cloud ?? null, value: cloud!.value } : { span: null, value: null },
      template,
      source: "template",
    });
  }
}

// Check the spans: each one must hold its words, ignoring case, and each label needs its span.
let bad = 0;
for (const r of records) {
  if ((r.place.words && !r.place.span) || (r.time.range && !r.time.span) || (r.cloud.value !== null && !r.cloud.span)) {
    bad++;
    console.error(JSON.stringify(["label without words", r.request, r.template]));
  }
  for (const [span, words] of [[r.place.span, r.place.words], [r.time.span, null], [r.cloud.span, null]] as const) {
    if (!span) continue;
    const text = r.request.slice(span[0], span[1]);
    if (!text.trim() || (words && text.toLowerCase() !== words.toLowerCase())) {
      bad++;
      if (bad <= 8) console.error(JSON.stringify([r.request, span, text, words, r.template]));
    }
  }
}
if (bad) throw new Error(`${bad} spans do not match their words`);

writeFileSync(new URL("records.jsonl", DATA), records.map((r) => JSON.stringify(r)).join("\n") + "\n");
const count = (s: string) => records.filter((r) => r.split === s).length;
console.log(
  `${records.length} records: train ${count("train")}, dev ${count("dev")}, test-catalog ${count("test-catalog")}; ` +
    `place ${records.filter((r) => r.place.words).length}, time ${records.filter((r) => r.time.span).length}, ` +
    `cloud ${records.filter((r) => r.cloud.span).length}, follow-ups ${records.filter((r) => r.current.catalog).length}`,
);
