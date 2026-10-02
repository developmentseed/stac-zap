/**
 * The world knowledge test: can a decision model place a named place on the
 * globe, with latitude and longitude bands as the options? This is what a
 * model needs to answer "where" without a geocoder.
 *
 * For each place, three calls:
 * - flat: one call with 100 latitude and 100 longitude bands.
 * - coarse: 10° bands (18 latitude, 36 longitude).
 * - fine: the coarse band and its neighbours, cut into FINE bands.
 * The error is the distance from the true point to the middle of the chosen
 * bands, for the flat call and for coarse-to-fine.
 *
 * Usage: PROVIDER=openrouter MODEL=typesafe/jev-1.13 pnpm tsx eval/world.ts
 * Optional: FINE=30 (default), RESULTS=eval/results/<date>.
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { ask, modelOfEnv, type Question } from "./providers";

type Band = { lo: number; hi: number; mid: number; label: string };

const fmt = (v: number, pos: string, neg: string) =>
  `${Math.abs(v).toFixed(2)}°${v >= 0 ? pos : neg}`;
/** Longitudes past ±180 wrap. */
const wrap = (lon: number) => ((((lon + 180) % 360) + 360) % 360) - 180;

function bands(min: number, max: number, n: number, axis: "lat" | "lon"): Band[] {
  const [pos, neg] = axis === "lat" ? ["N", "S"] : ["E", "W"];
  const w = (v: number) => (axis === "lon" ? wrap(v) : v);
  return Array.from({ length: n }, (_, i) => {
    const lo = min + ((max - min) * i) / n;
    const hi = min + ((max - min) * (i + 1)) / n;
    return { lo, hi, mid: (lo + hi) / 2, label: `${fmt(w(lo), pos, neg)} to ${fmt(w(hi), pos, neg)}` };
  });
}

const question = (axis: string, b: Band[]): Question => ({
  type: "choice",
  instructions: `Which ${axis} band contains the place in the request?`,
  criteria: Object.fromEntries(b.map((x) => [x.label, x.label])),
});

/** Great-circle distance in km. */
function km(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const r = Math.PI / 180;
  const x =
    Math.sin(((lat2 - lat1) * r) / 2) ** 2 +
    Math.cos(lat1 * r) * Math.cos(lat2 * r) * Math.sin(((lon2 - lon1) * r) / 2) ** 2;
  return 12742 * Math.asin(Math.sqrt(Math.min(1, x)));
}

export const PLACES: [string, number, number][] = [
  ["Lisbon", 38.72, -9.14], ["Reykjavik", 64.15, -21.94], ["Nairobi", -1.29, 36.82],
  ["Ushuaia", -54.8, -68.3], ["Ulaanbaatar", 47.92, 106.92], ["Anchorage", 61.22, -149.9],
  ["Perth, Australia", -31.95, 115.86], ["Kathmandu", 27.72, 85.32], ["Honolulu", 21.31, -157.86],
  ["Timbuktu", 16.77, -3.01], ["Longyearbyen, Svalbard", 78.22, 15.65], ["McMurdo Station", -77.85, 166.67],
  ["Quito", -0.18, -78.47], ["Lagos", 6.52, 3.38], ["Vladivostok", 43.12, 131.89],
  ["Tromsø", 69.65, 18.96], ["Hobart", -42.88, 147.33], ["Lima", -12.05, -77.04],
  ["Cairo", 30.04, 31.24], ["Mount Kilimanjaro", -3.07, 37.35], ["Lake Titicaca", -15.8, -69.4],
  ["Easter Island", -27.11, -109.35], ["Death Valley", 36.5, -117.1], ["Chernobyl", 51.27, 30.22],
  ["Fukushima Daiichi nuclear plant", 37.42, 141.03], ["the Aral Sea", 45.0, 60.0], ["Mount Everest", 27.99, 86.93],
  ["the Galápagos Islands", -0.95, -90.97], ["Bora Bora", -16.5, -151.74], ["Churchill, Manitoba", 58.77, -94.17],
  ["Dakar", 14.69, -17.44], ["Kinshasa", -4.32, 15.31], ["Antananarivo", -18.88, 47.51],
  ["Tbilisi", 41.72, 44.79], ["Almaty", 43.24, 76.95],
  ["cloud-free imagery of the Grand Canyon last summer", 36.1, -112.1],
  ["flooding in Porto Alegre", -30.03, -51.23], ["deforestation in Rondônia", -10.9, -62.8],
];

const { provider, model, tag } = modelOfEnv();
const FINE = Number(process.env.FINE ?? 30);
const dir = process.env.RESULTS ?? `eval/results/${new Date().toISOString().slice(0, 10)}`;
const latencies: number[] = [];
let cost = 0;

/** The chosen latitude and longitude bands. */
async function locate(place: string, lat: Band[], lon: Band[]): Promise<[Band, Band]> {
  const reply = await ask(provider, model, `User request for a map app: ${place}`, {
    lat: question("latitude", lat),
    lon: question("longitude", lon),
  });
  if ("error" in reply) throw new Error(reply.error);
  latencies.push(reply.ms);
  cost += reply.cost;
  const pick = (b: Band[], name: string) => {
    const band = b.find((x) => x.label === reply.answers[name]?.choice);
    if (!band) throw new Error(`no valid ${name} answer`);
    return band;
  };
  return [pick(lat, "lat"), pick(lon, "lon")];
}

const results: { name: string; flat?: number; fine?: number; error?: string }[] = [];
for (const [name, lat, lon] of PLACES) {
  try {
    const [fLat, fLon] = await locate(name, bands(-90, 90, 100, "lat"), bands(-180, 180, 100, "lon"));
    const [cLat, cLon] = await locate(name, bands(-90, 90, 18, "lat"), bands(-180, 180, 36, "lon"));
    const latLo = Math.max(-90, cLat.lo - 10);
    const latHi = Math.min(90, cLat.hi + 10);
    const [nLat, nLon] = await locate(name, bands(latLo, latHi, FINE, "lat"), bands(cLon.lo - 10, cLon.hi + 10, FINE, "lon"));
    results.push({
      name,
      flat: Math.round(km(lat, lon, fLat.mid, fLon.mid)),
      fine: Math.round(km(lat, lon, nLat.mid, wrap(nLon.mid))),
    });
    process.stderr.write(".");
  } catch (error) {
    results.push({ name, error: String(error).slice(0, 200) });
    process.stderr.write("x");
  }
}
console.error();
mkdirSync(dir, { recursive: true });
latencies.sort((a, b) => a - b);
writeFileSync(
  `${dir}/world-${tag}.json`,
  JSON.stringify({ provider, model, fine: FINE, cost, latencies, results }, null, 1),
);
const errors = (k: "flat" | "fine") =>
  results.flatMap((r) => (r[k] === undefined ? [] : [r[k]!])).sort((a, b) => a - b);
const median = (e: number[]) => e[e.length >> 1];
console.log(`${model}: flat median ${median(errors("flat"))} km, coarse-to-fine median ${median(errors("fine"))} km`);
