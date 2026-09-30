/**
 * Gives assets the same key in each item, so stac-map can show them.
 *
 * stac-map shows the items of a search as one mosaic of one asset key, e.g.
 * `visual`. Some catalogs use the file name as the key, e.g. data.geo.admin.ch
 * has `swissimage-dop10_2019_2680-1244_0.1_2056.tif` in the item
 * `swissimage-dop10_2019_2680-1244`. Then each key is in one item only, and
 * the mosaic shows one tile or none. This wraps `fetch` and, in item lists,
 * removes the item id from the start of such keys: the key above becomes
 * `0.1_2056.tif`, the same in each item.
 */

type Item = { id?: string; assets?: Record<string, unknown> };

/** Item lists: STAC API searches and the items of a collection. */
const ITEM_LIST = /\/(search|items)(\?|$)/;

export function normalizeAssetKeys(): void {
  const next = globalThis.fetch;
  globalThis.fetch = async (input, init) => {
    const response = await next(input, init);
    const href =
      input instanceof Request ? input.url : input instanceof URL ? input.href : input;
    if (
      !ITEM_LIST.test(new URL(href, location.href).pathname + "?") ||
      !response.ok ||
      !response.headers.get("content-type")?.includes("json")
    ) {
      return response;
    }
    const body = (await response.clone().json()) as { features?: Item[] };
    if (!body.features?.some(hasIdKeys)) return response;
    body.features.forEach(renameKeys);
    return new Response(JSON.stringify(body), {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    });
  };
}

function hasIdKeys(item: Item): boolean {
  return !!item.id && Object.keys(item.assets ?? {}).some((key) => key.startsWith(item.id!));
}

function renameKeys(item: Item): void {
  if (!item.id || !item.assets) return;
  const assets: Record<string, unknown> = {};
  for (const [key, asset] of Object.entries(item.assets)) {
    const short = key.startsWith(item.id)
      ? key.slice(item.id.length).replace(/^[-_.]+/, "")
      : key;
    // Two keys cannot become one: then the long key stays.
    assets[short && !(short in assets) ? short : key] = asset;
  }
  item.assets = assets;
}
