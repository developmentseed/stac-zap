/**
 * Saves the curated catalogs and their collections, as the app lists them,
 * to `data/stac-snapshot.json`. The field test reads this file, so all
 * models get the same options, even when a catalog changes.
 *
 * Usage: pnpm tsx eval/snapshot.ts
 */
import { writeFileSync } from "node:fs";
import { listCatalogs, listCollections } from "../src/zap/catalogs";

const catalogs = await listCatalogs(false);
const collections: Record<string, { id: string; title?: string; description?: string }[]> = {};
for (const catalog of catalogs) {
  try {
    const list = await listCollections(catalog.href);
    // The questions use only these fields.
    collections[catalog.id] = list.map(({ id, title, description }) => ({ id, title, description }));
  } catch (error) {
    console.error(`${catalog.id}: ${error}`);
    collections[catalog.id] = [];
  }
  console.log(catalog.id, collections[catalog.id]!.length);
}
writeFileSync(
  new URL("data/stac-snapshot.json", import.meta.url),
  JSON.stringify({ date: new Date().toISOString().slice(0, 10), catalogs, collections }),
);
