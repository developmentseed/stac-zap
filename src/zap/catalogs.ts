/**
 * The STAC APIs a prompt can open.
 *
 * By default, the list is curated: catalogs whose data stac-map can show in a
 * browser, with descriptions that tell jev what data they have. With the
 * "all catalogs" switch, the public APIs of STAC Index
 * (https://stacindex.org) are added. Many of those do not work in a browser:
 * some block other sites (CORS), some need a login for their data.
 *
 * The app reads the collections of each catalog when the page loads, and
 * keeps only the catalogs that answer. The collection titles also go into the
 * description, so jev knows what data each catalog has.
 */

import { STOPWORDS } from "./geocode";

export type Catalog = { id: string; href: string; description: string };

export type Link = { rel: string; href: string; title?: string };
export type Collection = {
  id: string;
  title?: string;
  description?: string;
  links?: Link[];
  extent?: { spatial?: { bbox?: number[][] } };
};

const STAC_INDEX_URL = "https://stacindex.org/api/catalogs";
/** jev reads this many characters of a STAC Index summary. */
const MAX_SUMMARY = 160;
/** jev reads this many characters of collection titles per catalog. */
const MAX_TITLES = 300;
/** More options than this is more than jev can take in one question. */
export const MAX_OPTIONS = 250;
/** Read at most this many collections of a catalog. */
const MAX_COLLECTIONS = 1000;
/** A catalog that does not answer in this time is left out, in milliseconds. */
const PROBE_TIMEOUT = 6000;

/**
 * The catalogs a prompt can open by default. For each one, stac-map can show
 * the data of sampled collections in a browser: the items load, and their
 * COGs answer range requests from another site. Checked on 2026-09-30.
 */
const CURATED: Catalog[] = [
  {
    id: "planetary-computer",
    href: "https://planetarycomputer.microsoft.com/api/stac/v1",
    description:
      "Microsoft Planetary Computer: Sentinel-1 radar, Sentinel-2, Landsat, " +
      "Copernicus DEM elevation, ESA WorldCover and IO land cover, NAIP, " +
      "MODIS, climate and weather data",
  },
  {
    id: "earth-search",
    href: "https://earth-search.aws.element84.com/v1",
    description:
      "Earth Search by Element 84: Sentinel-2, Landsat, Sentinel-1 and " +
      "Copernicus DEM on AWS",
  },
  {
    id: "eoapi",
    href: "https://stac.eoapi.dev",
    description:
      "eoAPI DevSeed demo: Maxar Open Data, very high resolution imagery of " +
      "disasters such as floods, hurricanes, wildfires, earthquakes and " +
      "volcanic eruptions",
  },
  {
    id: "openaerialmap",
    href: "https://api.imagery.hotosm.org/stac",
    description:
      "HOT OpenAerialMap: open drone and aerial imagery, and Maxar and " +
      "Vantor open data, often after disasters",
  },
  {
    id: "datageoadminch",
    href: "https://data.geo.admin.ch/api/stac/v1",
    description:
      "data.geo.admin.ch, Swiss federal geodata: SWISSIMAGE aerial " +
      "orthophotos, swissALTI3D elevation, swissSURFACE3D, maps, land use, " +
      "forest and climate data of Switzerland",
  },
  {
    id: "canadian-geospatial-data-collections-datacube",
    href: "https://datacube.services.geo.ca/stac/api",
    description:
      "Canadian Geospatial Data Collections: elevation models of Canada " +
      "(HRDEM lidar, MRDEM), land deformation, AVHRR composites and " +
      "historical aerial photos",
  },
  {
    id: "pgc-data-catalog",
    href: "https://stac.pgc.umn.edu/api/v1",
    description:
      "Polar Geospatial Center: high resolution elevation models of the " +
      "Arctic (ArcticDEM), Antarctica (REMA) and the rest of the world " +
      "(EarthDEM)",
  },
  {
    id: "impact-observatory-stac-api",
    href: "https://api.impactobservatory.com/stac-aws",
    description:
      "Impact Observatory: global 10 m annual land use and land cover maps",
  },
  {
    id: "earth-genome",
    href: "https://stac.earthgenome.org",
    description:
      "Earth Genome: global cloud-free Sentinel-2 yearly mosaics",
  },
  {
    id: "thunen-earth-observation-theo",
    href: "https://eodata.thuenen.de/stac/api/v1",
    description:
      "Thünen Earth Observation: crop type maps, winter cover and " +
      "fractional cover of Germany",
  },
  {
    id: "kentucky-from-above-spatiotemporal-asset-catalog",
    href: "https://spved5ihrl.execute-api.us-west-2.amazonaws.com",
    description:
      "Kentucky From Above: aerial orthophotos and lidar elevation models " +
      "of Kentucky, USA",
  },
  {
    id: "paituli-stac-finland",
    href: "https://paituli.csc.fi/geoserver/ogc/stac/v1",
    description:
      "Paituli (Finland): elevation and surface models, forest inventory, " +
      "CORINE land cover and Sentinel-1 backscatter of Finland",
  },
  {
    id: "digitale-orthophotos-niedersachsen",
    href: "https://dop.stac.lgln.niedersachsen.de",
    description:
      "Digital orthophotos of Lower Saxony (Niedersachsen), Germany: aerial " +
      "imagery",
  },
  {
    id: "kagis-katalog",
    href: "https://gis.ktn.gv.at/api/stac/v1",
    description: "KAGIS: aerial orthophotos of Carinthia (Kärnten), Austria",
  },
];

/**
 * Catalogs that are not in STAC Index, for the "all catalogs" list. stac-map
 * cannot show their data yet (e.g. VEDA assets have `s3://` links only), so
 * they are not in the default list.
 */
const EXTRA: Catalog[] = [
  {
    id: "veda",
    href: "https://openveda.cloud/api/stac",
    description:
      "NASA VEDA: NASA science datasets such as night lights, air quality " +
      "(NO2), greenhouse gases, fires, floods and disaster events",
  },
];

/** The catalog to open when jev does not pick one. */
export const DEFAULT_CATALOG = CURATED[0]!;

type IndexEntry = {
  slug: string;
  url: string;
  title: string;
  summary?: string | null;
  access?: string;
  isApi?: boolean;
  isPrivate?: boolean;
};

const catalogLists = new Map<boolean, Promise<Catalog[]>>();
const collectionLists = new Map<string, Promise<Collection[]>>();

/**
 * The catalogs that answer, curated first, with their collection titles.
 * This runs once per page load and list; call it early, so it is ready for
 * the first prompt.
 * @param all Also the catalogs of STAC Index, which stac-map cannot always
 * show. Off by default.
 */
export function listCatalogs(all = false): Promise<Catalog[]> {
  let list = catalogLists.get(all);
  if (!list) {
    list = (all ? fetchIndex().catch(() => []) : Promise.resolve([]))
      .then(async (entries) => {
        const candidates = all ? [...CURATED, ...EXTRA, ...entries] : CURATED;
        const unique = candidates.filter(
          (c, i) => candidates.findIndex((d) => d.href === c.href) === i,
        );
        const probed = await Promise.all(unique.map(probe));
        const reachable = probed.filter((c): c is Catalog => c !== null);
        return reachable.length > 0 ? reachable : CURATED;
      });
    catalogLists.set(all, list);
  }
  return list;
}

/** The catalog with its collection titles, or null if it does not answer. */
async function probe(catalog: Catalog): Promise<Catalog | null> {
  try {
    const collections = await listCollections(
      catalog.href,
      AbortSignal.timeout(PROBE_TIMEOUT),
    );
    if (collections.length === 0) return null;
    const titles = collections.map((c) => c.title ?? c.id).join(", ");
    const short =
      titles.length > MAX_TITLES ? `${titles.slice(0, MAX_TITLES)}…` : titles;
    return { ...catalog, description: `${catalog.description}. Collections: ${short}` };
  } catch {
    return null;
  }
}

/**
 * The collections of a catalog. Some APIs answer `/collections` with a web
 * page, not JSON; for those, the child links of the landing page are used.
 * @throws If the catalog does not answer.
 */
export function listCollections(
  catalogHref: string,
  signal?: AbortSignal,
): Promise<Collection[]> {
  let list = collectionLists.get(catalogHref);
  if (!list) {
    list = pagedCollections(catalogHref, signal)
      .catch(() => childCollections(catalogHref, signal))
      .then((collections) => collections.slice(0, MAX_COLLECTIONS));
    list.catch(() => collectionLists.delete(catalogHref));
    collectionLists.set(catalogHref, list);
  }
  return list;
}

/** The collections of `/collections`, and of its next pages. */
async function pagedCollections(
  catalogHref: string,
  signal?: AbortSignal,
): Promise<Collection[]> {
  const collections: Collection[] = [];
  let href: string | undefined =
    `${catalogHref}/collections?limit=${MAX_OPTIONS}`;
  // Some APIs send fewer collections per page than the limit asks for.
  while (href && collections.length < MAX_COLLECTIONS) {
    const page: { collections?: Collection[]; links?: Link[] } =
      await fetchJson(href, signal);
    if (!Array.isArray(page.collections)) throw new Error("No collections");
    collections.push(...page.collections);
    const next = linkOf(page, "next");
    href = next && next !== href ? next : undefined;
  }
  return collections;
}

/** The collections that the landing page links to as children. */
async function childCollections(
  catalogHref: string,
  signal?: AbortSignal,
): Promise<Collection[]> {
  const root = await fetchJson<{ links?: Link[] }>(catalogHref, signal);
  return (root.links ?? [])
    .filter((link) => link.rel === "child" && link.href.includes("/collections/"))
    .map((link) => {
      const id = decodeURIComponent(link.href.split("/collections/")[1]!.split(/[/?#]/)[0]!);
      return {
        id,
        title: link.title ?? id,
        links: [
          { rel: "self", href: link.href },
          { rel: "root", href: catalogHref },
        ],
      };
    });
}

/**
 * The collections that jev can pick from for a prompt. A catalog with more
 * collections than a question can take keeps the ones whose title and
 * description share the most words with the prompt. Words match on their
 * first six letters, so "orthophotos" matches "orthophotomosaic".
 */
export function shortlist(
  collections: readonly Collection[],
  prompt: string,
): Collection[] {
  if (collections.length <= MAX_OPTIONS) return [...collections];
  const stems = prompt
    .toLowerCase()
    .split(/[^\p{L}\p{N}]+/u)
    .filter((word) => word.length >= 3 && !STOPWORDS.has(word))
    .map((word) => word.slice(0, 6));
  const score = (c: Collection) => {
    const text = `${c.id} ${c.title ?? ""} ${c.description ?? ""}`.toLowerCase();
    return stems.filter((stem) => text.includes(stem)).length;
  };
  return collections
    .map((collection, index) => ({ collection, index, score: score(collection) }))
    .sort((a, b) => b.score - a.score || a.index - b.index)
    .slice(0, MAX_OPTIONS)
    .map(({ collection }) => collection);
}

type Asset = {
  href: string;
  type?: string;
  alternate?: Record<string, { href?: string }>;
  bands?: unknown[];
  "raster:bands"?: unknown[];
  "eo:bands"?: unknown[];
};

/**
 * The URL of a COG that stac-map can show, or undefined. The same rules as
 * `getCogHref` in stac-map: a GeoTIFF with 1, 3 or 4 bands and an HTTP link.
 */
function renderableHref(asset: Asset): string | undefined {
  if (!asset.type?.startsWith("image/tiff; application=geotiff")) return undefined;
  const bands = (asset["raster:bands"] ?? asset.bands ?? asset["eo:bands"])?.length;
  if (bands !== undefined && ![1, 3, 4].includes(bands)) return undefined;
  if (asset.href.startsWith("http")) return asset.href;
  return Object.values(asset.alternate ?? {}).find((alt) =>
    alt.href?.startsWith("http"),
  )?.href;
}

const renderable = new Map<string, Promise<boolean>>();

/**
 * Whether stac-map can show the data of a collection, from its first item.
 * A collection can have items with archives only, e.g. a `.tar` file. When
 * the items cannot be read, the answer is true: this is not known.
 */
export function hasRenderableItem(collectionHref: string): Promise<boolean> {
  let result = renderable.get(collectionHref);
  if (!result) {
    result = fetchJson<{ features?: { assets?: Record<string, Asset> }[] }>(
      `${collectionHref}/items?limit=1`,
      AbortSignal.timeout(PROBE_TIMEOUT),
    ).then(
      ({ features }) => {
        const item = features?.[0];
        return !!item && Object.values(item.assets ?? {}).some(renderableHref);
      },
      () => true,
    );
    renderable.set(collectionHref, result);
  }
  return result;
}

export async function fetchJson<T>(href: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(href, {
    headers: { Accept: "application/json" },
    signal,
  });
  if (!response.ok) throw new Error(`${href}: ${response.status}`);
  return (await response.json()) as T;
}

export function linkOf(value: { links?: Link[] }, rel: string): string | undefined {
  return value.links?.find((link) => link.rel === rel)?.href;
}

/** The catalog that an href is in, e.g. a collection of it. */
export function catalogOf(
  list: readonly Catalog[],
  href: string | null,
): Catalog | null {
  if (!href) return null;
  // The longest match, so a catalog inside another one wins.
  return (
    list
      .filter((c) => href === c.href || href.startsWith(`${c.href}/`))
      .sort((a, b) => b.href.length - a.href.length)[0] ?? null
  );
}

async function fetchIndex(): Promise<Catalog[]> {
  const response = await fetch(STAC_INDEX_URL);
  if (!response.ok) throw new Error(`STAC Index: ${response.status}`);
  const entries = (await response.json()) as IndexEntry[];
  return entries
    .filter(
      (e) =>
        e.isApi &&
        e.access === "public" &&
        !e.isPrivate &&
        // openEO back ends are listed as APIs, but they are not STAC APIs.
        !/openeo/i.test(e.url) &&
        // A URL with a query usually carries a key.
        !e.url.includes("?"),
    )
    .map((e) => ({
      id: e.slug,
      href: e.url.replace(/\/+$/, ""),
      description: describe(e),
    }));
}

function describe(entry: IndexEntry): string {
  const summary = (entry.summary ?? "")
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\s+/g, " ")
    .trim();
  const short =
    summary.length > MAX_SUMMARY ? `${summary.slice(0, MAX_SUMMARY)}…` : summary;
  return short ? `${entry.title}: ${short}` : entry.title;
}
