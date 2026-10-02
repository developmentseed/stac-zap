/**
 * The test prompts of the field test, with the accepted answers per field.
 *
 * A list holds the accepted option ids. "no change" accepts `keep` and the
 * option that means no filter (all-time, whole-year, any). A missing field is
 * not scored. `cat` is the catalog whose collections the collection question
 * offers, when it is not the first accepted catalog.
 */
import { MAP_VIEW, NO_PLACE } from "../src/zap/options";

export const NO_CHANGE = "no change";

export type Case = {
  prompt: string;
  catalog: string[];
  cat?: string;
  collection?: string[];
  place?: string[];
  period?: string[] | typeof NO_CHANGE;
  season?: string[] | typeof NO_CHANGE;
  cloud?: string[] | typeof NO_CHANGE;
};

const PC = "planetary-computer";
const S2 = ["sentinel-2-l2a"];
const LS = ["landsat-c2-l2"];
const S1 = ["sentinel-1-rtc", "sentinel-1-grd"];
const N = NO_CHANGE;

export const CASES: Case[] = [
  { prompt: "Sentinel-2 over Lisbon last summer with few clouds", catalog: [PC], collection: S2, place: ["Lisbon"], period: ["2026"], season: ["summer"], cloud: ["15", "20", "25"] },
  { prompt: "I want S2 images in Johor Malaysia with less than 20% cloud cover", catalog: [PC], collection: S2, place: ["Johor", "Johor Malaysia"], period: N, season: N, cloud: ["15", "20"] },
  { prompt: "Landsat over Denver in July 2024", catalog: [PC], collection: LS, place: ["Denver"], period: ["2024"], season: ["m7"], cloud: N },
  { prompt: "cloud-free imagery of the Grand Canyon", catalog: [PC], collection: [...S2, ...LS], place: ["Grand Canyon"], period: N, season: N, cloud: ["5", "10"] },
  { prompt: "Sentinel-1 radar of the Netherlands in January 2025", catalog: [PC], collection: S1, place: ["Netherlands"], period: ["2025"], season: ["m1"], cloud: N },
  { prompt: "elevation of Mount Everest", catalog: [PC, "pgc-data-catalog"], cat: PC, collection: ["cop-dem-glo-30", "cop-dem-glo-90", "nasadem", "alos-dem"], place: ["Mount Everest", "Everest"], period: N, season: N, cloud: N },
  { prompt: "land cover of Kenya", catalog: [PC], collection: ["esa-worldcover"], place: ["Kenya"], period: N, season: N, cloud: N },
  { prompt: "Swiss orthophotos of Zurich", catalog: ["datageoadminch"], collection: ["ch.swisstopo.swissimage-dop10", "ch.swisstopo.swissimage"], place: ["Zurich"], period: N, season: N, cloud: N },
  { prompt: "Maxar imagery after Hurricane Idalia", catalog: ["eoapi", "openaerialmap"], cat: "eoapi", collection: ["MAXAR_Hurricane_Idalia_Florida_Aug23"], place: [NO_PLACE], period: N, season: N },
  { prompt: "ArcticDEM elevation of Svalbard", catalog: ["pgc-data-catalog"], collection: ["arcticdem-mosaics-v4.1-2m", "arcticdem-mosaics-v4.1-10m", "arcticdem-mosaics-v4.1-32m", "arcticdem-mosaics-v3.0-2m", "arcticdem-mosaics-v3.0-10m", "arcticdem-mosaics-v3.0-32m", "arcticdem-strips-s2s041-2m"], place: ["Svalbard"], period: N, season: N, cloud: N },
  { prompt: "NAIP imagery of Austin, Texas from 2022", catalog: [PC], collection: ["naip"], place: ["Austin", "Austin Texas"], period: ["2022"], season: N, cloud: N },
  { prompt: "crop types in Lower Saxony", catalog: ["thunen-earth-observation-theo"], collection: ["crop-type-map-latest", "crop-type-map-v302"], place: ["Lower Saxony"], period: N, season: N, cloud: N },
  { prompt: "aerial photos of Hannover", catalog: ["digitale-orthophotos-niedersachsen"], collection: ["DOP"], place: ["Hannover"], period: N, season: N, cloud: N },
  { prompt: "recent Sentinel-2 of the Amazon rainforest", catalog: [PC], collection: S2, place: ["Amazon rainforest", "Amazon"], period: ["last-30-days"], season: N },
  { prompt: "MODIS snow cover over the Alps in winter 2024", catalog: [PC], collection: ["modis-10A1-061", "modis-10A2-061"], place: ["Alps"], period: ["2024"], season: ["winter"], cloud: N },
  { prompt: "Landsat of the Aral Sea in 2016, under 10% clouds", catalog: [PC], collection: LS, place: ["Aral Sea"], period: ["2016"], season: N, cloud: ["10"] },
  { prompt: "Sentinel-2 of this area, mostly clear", catalog: [PC], collection: S2, place: [MAP_VIEW], period: N, season: N, cloud: ["10", "15", "20"] },
  { prompt: "Sentinel-2 Level-1C over Rome", catalog: ["earth-search"], collection: ["sentinel-2-l1c"], place: ["Rome"], period: N, season: N },
  { prompt: "Kentucky lidar elevation near Louisville", catalog: ["kentucky-from-above-spatiotemporal-asset-catalog"], collection: ["dem-phase1", "dem-phase2", "dem-phase3", "laz-phase1", "laz-phase2", "laz-phase3"], place: ["Louisville"], period: N, season: N, cloud: N },
  { prompt: "global cloud-free Sentinel-2 mosaic of Lagos for 2023", catalog: ["earth-genome"], place: ["Lagos"], period: ["2023"], season: N },
  { prompt: "drone imagery of Kathmandu", catalog: ["openaerialmap"], collection: ["openaerialmap"], place: ["Kathmandu"], period: N, season: N, cloud: N },
  { prompt: "Landsat over Iceland in autumn 2021 with less than 40% clouds", catalog: [PC], collection: LS, place: ["Iceland"], period: ["2021"], season: ["autumn"], cloud: ["40"] },
  { prompt: "radar images of Lisbon in the last 12 months", catalog: [PC], collection: S1, place: ["Lisbon"], period: ["last-12-months"], season: N, cloud: N },
  { prompt: "Sentinel-2 over Cotopaxi in December 2025 with less than 30% clouds", catalog: [PC], collection: S2, place: ["Cotopaxi"], period: ["2025"], season: ["m12"], cloud: ["30"] },
  { prompt: "Finnish forest inventory near Helsinki", catalog: ["paituli-stac-finland"], place: ["Helsinki"], period: N, season: N, cloud: N },
  { prompt: "Carinthia orthophotos of Klagenfurt", catalog: ["kagis-katalog"], collection: ["KAGIS_coll_ortho_klagenfurt_2025460", "KAGIS_coll_ortho_klagenfurt_2022460"], place: ["Klagenfurt"], period: N, season: N, cloud: N },
  { prompt: "Canadian high resolution DEM of Vancouver", catalog: ["canadian-geospatial-data-collections-datacube"], collection: ["hrdem-mosaic-1m", "hrdem-mosaic-2m", "hrdem-lidar"], place: ["Vancouver"], period: N, season: N, cloud: N },
  { prompt: "Sentinel-2 of Nairobi in spring 2023, any clouds", catalog: [PC], collection: S2, place: ["Nairobi"], period: ["2023"], season: ["spring"], cloud: ["any", "keep"] },
  { prompt: "Landsat imagery from March 2019 of Lake Titicaca", catalog: [PC], collection: LS, place: ["Lake Titicaca", "Titicaca"], period: ["2019"], season: ["m3"], cloud: N },
  { prompt: "wildfires in Los Angeles in January 2025", catalog: ["eoapi"], collection: ["WildFires-LosAngeles-Jan-2025"], place: ["Los Angeles"], period: ["2025"], season: ["m1"] },
];
