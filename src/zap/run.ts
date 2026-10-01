import { useStore } from "@developmentseed/stac-map";
import { decide, KEEP, THRESHOLD, type ZapFields, type ZapResult } from "./decide";
import {
  cloudChoice,
  CLOUD_OPTIONS,
  CLOUD_SHORT,
  MAP_VIEW,
  NO_PLACE,
  periodOptions,
  SEASON_OPTIONS,
  toRange,
  type BBox2D,
  type Range,
} from "./options";
import {
  catalogOf,
  DEFAULT_CATALOG,
  fetchJson,
  linkOf,
  listCatalogs,
  hasRenderableItem,
  listCollections,
  shortlist,
  type Collection,
  type Link,
} from "./catalogs";
import { geocode, placeSpans, type Place } from "./geocode";

/**
 * Runs one prompt in two or three jev calls. The collection options come from
 * the catalog, so the first call picks the catalog and the second call picks
 * the collection and the search. For the place, the second call picks the
 * words of the prompt that name it, Nominatim finds the places with that
 * name, and a third call picks the one the prompt means. The changes go
 * through the stac-map store, so a zap does what the same clicks in the panel
 * would do.
 */

/** What a zap opened, e.g. for analytics. */
export type ZapOutcome = { catalog: string; collection: string };

/** One jev call, with the fields it answered, for the probability panel. */
export type ZapStep = { fields: ZapFields; result: ZapResult };

/** How a zap runs, and what it reports while it runs. */
export type ZapListener = {
  /** Also the catalogs of STAC Index, not only the curated ones. */
  allCatalogs: boolean;
  /** Called after each jev call, so the panel can show its answers. */
  onStep: (step: ZapStep) => void;
  /** Called with a line for the user, e.g. when no place has the name. */
  onNote: (note: string) => void;
};

/** The options the last prompt picked, so the next prompt can keep them. */
type Search = {
  place?: Omit<Place, "id">;
  period?: string;
  season: string;
  cloud: string;
};

/**
 * The number of items per search page. Satellite scenes with cloud cover,
 * e.g. Sentinel-2, are large and overlap a lot, and each one is a COG to
 * load, so a few are enough. Tiled data, e.g. 1 km orthophoto tiles, needs
 * more items to cover a city.
 */
const SCENE_LIMIT = "10";
const TILE_LIMIT = "50";
/** How long to wait for stac-map to load a collection, in milliseconds. */
const LOAD_TIMEOUT = 10_000;

let last: Search = { season: "whole-year", cloud: "any" };

/**
 * Runs one prompt against the map.
 * @param prompt What the user typed.
 * @throws If jev or a STAC API cannot answer.
 */
export async function zap(
  prompt: string,
  { allCatalogs, onStep, onNote }: ZapListener,
): Promise<ZapOutcome | null> {
  const store = useStore.getState();
  const catalogs = await listCatalogs(allCatalogs);
  const currentCatalog = catalogOf(catalogs, store.href)?.id ?? null;

  const first: ZapFields = {
    catalog: {
      label: "STAC catalog",
      current: currentCatalog,
      options: Object.fromEntries(
        catalogs.map((c) => [c.id, c.description]),
      ),
      keepable: currentCatalog !== null,
      threshold: 0,
    },
  };
  const firstResult = await decide(prompt, first);
  onStep({ fields: first, result: firstResult });
  const catalogId =
    firstResult.changes.catalog ??
    currentCatalog ??
    validChoice(first, firstResult, "catalog");
  const catalog = catalogs.find((c) => c.id === catalogId) ?? DEFAULT_CATALOG;
  const catalogHref = catalog.href;

  let collections: Collection[];
  try {
    collections = await listCollections(catalogHref);
  } catch (error) {
    // A catalog from STAC Index can be down or block browsers (CORS).
    onNote(`Could not list the collections of ${catalogHref}: ${error}`);
    return null;
  }
  if (collections.length === 0) {
    onNote(`${catalogHref} has no collections`);
    return null;
  }
  const options = shortlist(collections, prompt);
  const currentCollection =
    catalog.id === currentCatalog
      ? collectionOf(store.href, catalogHref)
      : null;
  const second: ZapFields = {
    collection: {
      label: "collection",
      current: currentCollection,
      options: Object.fromEntries(options.map((c) => [c.id, describe(c)])),
      keepable: currentCollection !== null,
      threshold: 0,
    },
    place: {
      label: "place to search",
      current: last.place?.label ?? null,
      // The options are words of the prompt, so jev reads them as they are.
      options: {
        ...Object.fromEntries(placeSpans(prompt).map((span) => [span, span])),
        [MAP_VIEW]: "The area the map shows now, e.g. 'here' or 'this area'",
        [NO_PLACE]:
          "No place: search everywhere. The request names no place, only " +
          "data, an event such as a hurricane, or a time",
      },
      question:
        "Which words of the request name the place to search? A place can be " +
        "a city, a region, a country, a landmark or a natural feature, " +
        "such as a mountain range, a river or a desert. The name of an " +
        "event, e.g. 'Hurricane Idalia', or of a dataset is not a place.",
    },
    period: {
      label: "time period",
      current: last.period ?? null,
      options: periodOptions(),
      // "Recent" can split between "the last 30 days" and "this year". Either
      // is a better filter than none, so a lower bar applies.
      threshold: 0.35,
    },
    season: {
      label: "part of the year",
      current: last.season,
      options: SEASON_OPTIONS,
    },
    cloud: {
      label: "maximum cloud cover",
      current: last.cloud,
      options: CLOUD_OPTIONS,
      short: CLOUD_SHORT,
    },
  };
  const secondResult = await decide(prompt, second);
  // The cloud cover options are an ordered scale, so the median applies, not
  // the 0.5 threshold. The panel shows the median, as it is the filter used.
  const cloud = medianCloud(secondResult, last.cloud);
  onStep({ fields: second, result: secondResult });

  const collectionId =
    secondResult.changes.collection ??
    currentCollection ??
    validChoice(second, secondResult, "collection");
  const chosen = collections.find((c) => c.id === collectionId);
  if (!chosen) return null;
  const collection =
    chosen.id === currentCollection
      ? chosen
      : await pickRenderable(chosen, secondResult, collections, catalogHref, onNote);

  // A new collection starts with no time and cloud filter: the last ones can
  // leave it with no items (e.g. SWISSIMAGE is flown every three years). The
  // place stays, so "now the elevation" searches the same place. Then this
  // prompt's answers apply even when they equal the last value.
  const newCollection = collection.id !== currentCollection;
  const changes = newCollection
    ? { ...acceptedAnswers(second, secondResult), ...secondResult.changes }
    : { ...secondResult.changes };
  if (cloud) changes.cloud = cloud;
  const next: Search = newCollection
    ? { place: last.place, season: "whole-year", cloud: "any" }
    : { ...last };
  if (changes.place === NO_PLACE) {
    next.place = undefined;
  } else if (changes.place === MAP_VIEW) {
    if (store.mapBbox) {
      next.place = { label: "The map view", bbox: store.mapBbox };
    }
  } else if (changes.place) {
    const place = await findPlace(prompt, changes.place, onStep, onNote);
    // With a new collection, the last place is only a guess; a named place
    // that was not found replaces it with no place.
    if (place || newCollection) next.place = place ?? undefined;
  }
  // A place from an earlier prompt can be outside the new collection, e.g.
  // Zurich for Hurricane Idalia in Florida. Then the search finds nothing.
  if (
    newCollection &&
    next.place &&
    !changes.place &&
    !overlaps(next.place.bbox, collection.extent?.spatial?.bbox?.[0])
  ) {
    onNote(`${next.place.label.split(",")[0]} is outside this collection, so the search covers all of it`);
    next.place = undefined;
  }
  if (changes.cloud) next.cloud = changes.cloud;
  if (changes.period) next.period = changes.period;
  if (changes.season) {
    next.season = changes.season;
    // "Last summer" can come back as a season with no year.
    if (!next.period || !/^\d{4}$/.test(next.period)) {
      next.period = latestYearOf(changes.season);
    }
  }
  last = next;

  const collectionHref = hrefOf(collection, catalogHref);
  const [{ searchHref, filters }, cloudQueryable] = await Promise.all([
    searchOf(collection, catalogHref),
    queryableExists(collectionHref, "eo:cloud_cover"),
  ]);
  const hasCloudCover = cloudQueryable && filters;
  // Earth Search has the queryable, but its search ignores a CQL2 filter.
  // stac-map sends only CQL2, so the items would have any cloud cover.
  if (cloudQueryable && !filters && next.cloud !== "any") {
    onNote(`${catalogHref} cannot filter by cloud cover, so the items can have any cloud cover`);
  }
  const bbox: BBox2D | undefined = next.place?.bbox;
  let range = next.period ? toRange(next.period, next.season) : null;
  // A period outside the dates of the collection finds nothing, e.g. 2026
  // for Hurricane Idalia in August 2023. Then the search covers all dates.
  if (range && !inTime(range, collection.extent?.temporal?.interval?.[0])) {
    onNote("The time period is outside this collection's dates, so the search covers all of them");
    range = null;
    next.period = undefined;
    next.season = "whole-year";
  }

  // A time period can be inside the dates of a collection and still find
  // nothing, e.g. 2025 for land cover maps that end in 2024. A test search
  // finds out; then the search covers all dates.
  if (range && !(await hasItems(searchHref, collection.id, range, bbox))) {
    onNote("Nothing matches this time period, so the search covers all dates");
    range = null;
    next.period = undefined;
    next.season = "whole-year";
  }

  await open(collectionHref, () =>
    store.setSearchParams(searchHref, {
      // An empty datetime is no filter: the whole record of the collection.
      startDatetime: range?.startDatetime ?? "",
      endDatetime: range?.endDatetime ?? "",
      limit: cloudQueryable ? SCENE_LIMIT : TILE_LIMIT,
      bbox,
      queryables:
        hasCloudCover && next.cloud !== "any"
          ? { "eo:cloud_cover": { lte: Number(next.cloud) } }
          : {},
    }),
  );
  // deck.gl-raster cannot draw COGs on the globe yet ("implement
  // getBoundingVolume in Globe view"), so a zap always uses the flat map.
  // stac-map sets the projection and the camera from the loaded value, so
  // these go after it.
  store.setProjection("mercator");
  if (bbox) store.setValueBbox(bbox);
  return { catalog: catalog.id, collection: collection.id };
}

/** The number of collections, best first, to check for data stac-map can show. */
const RENDER_CANDIDATES = 4;
/** Collections less likely than this are not checked. */
const MIN_CANDIDATE = 0.02;

/**
 * The collection to open: jev's choice, or the next most likely collection
 * whose items stac-map can show. For example, "Swiss orthophotos" can match
 * both "SWISSIMAGE Background", whose only item is a `.tar` archive, and
 * "SWISSIMAGE 10 cm", which has COGs.
 */
async function pickRenderable(
  chosen: Collection,
  result: ZapResult,
  collections: readonly Collection[],
  catalogHref: string,
  onNote: ZapListener["onNote"],
): Promise<Collection> {
  const probabilities = result.answers.collection?.probabilities ?? {};
  const ranked = [
    chosen,
    ...Object.entries(probabilities)
      .filter(([id, p]) => id !== chosen.id && p >= MIN_CANDIDATE)
      .sort((a, b) => b[1] - a[1])
      .flatMap(([id]) => collections.filter((c) => c.id === id)),
  ].slice(0, RENDER_CANDIDATES);
  const checks = await Promise.all(
    ranked.map((c) => hasRenderableItem(hrefOf(c, catalogHref))),
  );
  const index = checks.indexOf(true);
  if (index === -1) {
    onNote(`stac-map may not show the data of ${chosen.title ?? chosen.id}: its items have no COG`);
    return chosen;
  }
  const picked = ranked[index]!;
  if (index > 0) {
    onNote(
      `${chosen.title ?? chosen.id} has no data stac-map can show, so this ` +
        `opens ${picked.title ?? picked.id} ` +
        `(${Math.round((probabilities[picked.id] ?? 0) * 100)}%)`,
    );
  }
  return picked;
}

function hrefOf(collection: Collection, catalogHref: string): string {
  return linkOf(collection, "self") ?? `${catalogHref}/collections/${collection.id}`;
}

/**
 * Finds the place that a name in the prompt means. With more than one match,
 * jev picks the one that fits the prompt, e.g. the Alps in Europe and not a
 * peak called "The Alps" in Vermont.
 * @returns The place, or null if Nominatim has no match or fails.
 */
async function findPlace(
  prompt: string,
  name: string,
  onStep: ZapListener["onStep"],
  onNote: ZapListener["onNote"],
): Promise<Place | null> {
  let places: Place[];
  try {
    places = await geocode(name);
  } catch (error) {
    onNote(`Could not look up "${name}": ${error}`);
    return null;
  }
  if (places.length === 0) {
    onNote(`No place found for "${name}"`);
    return null;
  }
  // Nominatim can match only other things with the same name, e.g. shops
  // called "Amazon", so jev can also reject all of them.
  const fields: ZapFields = {
    match: {
      label: "place the request means",
      current: null,
      options: {
        ...Object.fromEntries(places.map((p) => [p.id, p.label])),
        [NO_MATCH]: `None of these: the request means a different "${name}"`,
      },
      keepable: false,
      question: `The request names "${name}". Which of these places does it mean?`,
    },
  };
  const result = await decide(prompt, fields);
  onStep({ fields, result });
  const place = places.find((p) => p.id === validChoice(fields, result, "match"));
  if (!place) onNote(`No place found for "${name}"`);
  return place ?? null;
}

/** The `match` option for "none of the places Nominatim found". */
const NO_MATCH = "none";

/**
 * Opens a collection in stac-map and waits until it is loaded.
 *
 * stac-map's search panel reads its parameters only when it mounts. When the
 * href changes, stac-map can keep the panel, and the panel then writes the
 * parameters of the last collection over the new ones. So the open value is
 * closed first, and the parameters are written while nothing is open.
 * @param prepare Writes the search parameters.
 */
async function open(href: string, prepare: () => void): Promise<void> {
  const store = useStore.getState();
  const loaded = new Promise<void>((resolve) => {
    const timer = setTimeout(done, LOAD_TIMEOUT);
    const unsubscribe = useStore.subscribe((state, previous) => {
      if (state.valueBbox && state.valueBbox !== previous.valueBbox) done();
    });
    function done() {
      clearTimeout(timer);
      unsubscribe();
      resolve();
    }
  });
  if (store.href !== null) {
    store.setHref(null);
    await new Promise((resolve) => setTimeout(resolve));
  }
  prepare();
  store.setHref(href);
  await loaded;
}

/** Whether a range overlaps the dates of a collection. Unknown dates: true. */
function inTime(range: Range, interval: (string | null)[] | undefined): boolean {
  if (!interval) return true;
  const start = interval[0] ? Date.parse(interval[0]) : -Infinity;
  const end = interval[1] ? Date.parse(interval[1]) : Infinity;
  return (
    Date.parse(`${range.startDatetime}Z`) <= end &&
    Date.parse(`${range.endDatetime}Z`) >= start
  );
}

/** Whether two boxes overlap. Without a collection box, this is not known: true. */
function overlaps(place: BBox2D, extent: number[] | undefined): boolean {
  if (!extent || extent.length < 4) return true;
  // A 3D box is [west, south, bottom, east, north, top].
  const [w, s, e, n] =
    extent.length >= 6
      ? [extent[0]!, extent[1]!, extent[3]!, extent[4]!]
      : [extent[0]!, extent[1]!, extent[2]!, extent[3]!];
  // A box across the antimeridian has west > east; treat it as global.
  if (w > e) return place[1] <= n && place[3] >= s;
  return place[0] <= e && place[2] >= w && place[1] <= n && place[3] >= s;
}

/**
 * Replaces jev's cloud cover answer with the median option of
 * {@link cloudChoice}, in the answers and in the changes.
 * @param current The cloud cover option before this prompt.
 * @returns The median option, or null to keep the current one.
 */
function medianCloud(result: ZapResult, current: string): string | null {
  const answer = result.answers.cloud;
  if (!answer) return null;
  const cloud = cloudChoice(answer.probabilities, KEEP);
  const choice = cloud ?? KEEP;
  result.answers.cloud = {
    ...answer,
    choice,
    probability: answer.probabilities[choice] ?? 0,
  };
  delete result.changes.cloud;
  if (cloud && cloud !== current) result.changes.cloud = cloud;
  return cloud;
}

/** The answers that are options, not `keep`, with enough probability. */
function acceptedAnswers(
  fields: ZapFields,
  result: ZapResult,
): Record<string, string> {
  return Object.fromEntries(
    Object.entries(result.answers).filter(
      ([name, { choice, probability }]) =>
        choice !== KEEP &&
        choice in (fields[name]?.options ?? {}) &&
        probability >= (fields[name]?.threshold ?? THRESHOLD),
    ).map(([name, { choice }]) => [name, choice]),
  );
}

/** The option jev liked most, even under the threshold, if it is valid. */
function validChoice(
  fields: ZapFields,
  result: ZapResult,
  name: string,
): string | null {
  const choice = result.answers[name]?.choice;
  return choice && choice in fields[name]!.options ? choice : null;
}

function collectionOf(href: string | null, catalogHref: string): string | null {
  const prefix = `${catalogHref}/collections/`;
  if (!href?.startsWith(prefix)) return null;
  return decodeURIComponent(href.slice(prefix.length).split(/[/?#]/)[0]!);
}

/** The most recent year in which the season is over or has started. */
function latestYearOf(season: string, today: Date = new Date()): string {
  const year = today.getUTCFullYear();
  const range = toRange(String(year), season, today);
  const started = range && new Date(`${range.startDatetime}Z`) <= today;
  return String(started ? year : year - 1);
}

/**
 * Whether a search for one item finds one, with the time and the place. When
 * the search fails, the answer is true: this is not known.
 */
async function hasItems(
  searchHref: string,
  collection: string,
  range: Range,
  bbox: BBox2D | undefined,
): Promise<boolean> {
  const url = new URL(searchHref);
  url.searchParams.set("collections", collection);
  url.searchParams.set("datetime", `${range.startDatetime}Z/${range.endDatetime}Z`);
  url.searchParams.set("limit", "1");
  if (bbox) url.searchParams.set("bbox", bbox.join(","));
  try {
    const page = await fetchJson<{ features?: unknown[] }>(url.href);
    return (page.features?.length ?? 1) > 0;
  } catch {
    return true;
  }
}

/**
 * The search link stac-map uses for a collection: the one of its root. The
 * search parameters are stored under this href. Also whether the search takes
 * a CQL2 filter.
 */
async function searchOf(
  collection: Collection,
  catalogHref: string,
): Promise<{ searchHref: string; filters: boolean }> {
  const root = await fetchJson<{ links?: Link[]; conformsTo?: string[] }>(
    linkOf(collection, "root") ?? catalogHref,
  );
  const search = linkOf(root, "search");
  if (!search) throw new Error(`${catalogHref} has no search link`);
  // The Filter extension, e.g. `https://api.stacspec.org/v1.0.0-rc.2/item-search#filter`.
  const filters = (root.conformsTo ?? []).some((c) => /item-search#filter/.test(c));
  return { searchHref: search, filters };
}

async function queryableExists(
  collectionHref: string,
  name: string,
): Promise<boolean> {
  try {
    const queryables = await fetchJson<{ properties?: object }>(
      `${collectionHref}/queryables`,
    );
    return name in (queryables.properties ?? {});
  } catch {
    return false;
  }
}

/** The collection as jev reads it: title and the start of the description. */
function describe(collection: Collection): string {
  const text = (collection.description ?? "")
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/[#*_`>]/g, "")
    .replace(/\s+/g, " ")
    .trim();
  const short = text.length > 120 ? `${text.slice(0, 120)}…` : text;
  const title = collection.title ?? collection.id;
  return short ? `${title}: ${short}` : title;
}
