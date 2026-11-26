/**
 * Saves the catalogs and collections for the training data: the curated
 * catalogs and the public APIs of STAC Index that answer, as the app lists
 * them with the "All STAC Index catalogs" switch. For each collection it
 * keeps the fields the generator reads (title, description, keywords,
 * providers, extent) and the licence of the metadata.
 *
 * Output: training/data/stac-full.json (not in git; published in the
 * dataset `developmentseed/stac-zap-data`).
 *
 * Usage: pnpm tsx training/snapshot.ts
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { listCatalogs, listCollections, type Collection } from "../src/zap/catalogs";

type Full = Collection & {
  keywords?: string[];
  license?: string;
  providers?: { name?: string }[];
};

const curated = new Set((await listCatalogs(false)).map((c) => c.href));
const catalogs = await listCatalogs(true);
console.log(`${catalogs.length} catalogs answer (${curated.size} curated)`);

const collections: Record<string, unknown[]> = {};
const failed: string[] = [];
let next = 0;
async function worker() {
  while (next < catalogs.length) {
    const catalog = catalogs[next++]!;
    try {
      const list = (await listCollections(catalog.href)) as Full[];
      collections[catalog.id] = list.map((c) => ({
        id: c.id,
        title: c.title,
        description: c.description,
        keywords: c.keywords,
        license: c.license,
        providers: c.providers?.map((p) => p.name).filter(Boolean),
        bbox: c.extent?.spatial?.bbox?.[0],
        interval: c.extent?.temporal?.interval?.[0],
      }));
      console.log(catalog.id, list.length);
    } catch (error) {
      failed.push(catalog.id);
      console.error(`${catalog.id}: ${error}`);
    }
  }
}
await Promise.all(Array.from({ length: 8 }, worker));

// A curated catalog that is down now keeps its collections of the field-test
// snapshot (id, title and description only).
const eval_ = JSON.parse(readFileSync(new URL("../eval/data/stac-snapshot.json", import.meta.url), "utf-8"));
for (const id of failed) {
  if (eval_.collections[id]?.length) {
    collections[id] = eval_.collections[id];
    console.log(`${id}: ${eval_.collections[id].length} collections from the field-test snapshot`);
  }
}

mkdirSync(new URL("data/", import.meta.url), { recursive: true });
writeFileSync(
  new URL("data/stac-full.json", import.meta.url),
  JSON.stringify({
    date: new Date().toISOString().slice(0, 10),
    catalogs: catalogs
      .filter((c) => collections[c.id])
      .map((c) => ({ ...c, curated: curated.has(c.href) })),
    collections,
  }),
);
console.log(`saved ${Object.keys(collections).length} catalogs, ${Object.values(collections).flat().length} collections; failed: ${failed.length}`);
