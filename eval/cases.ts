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

/**
 * A prompt that stac-zap cannot parse today: the correct answer is not in
 * the option lists of the app, so a decision model can only select the
 * nearest wrong option. An extraction model can give the exact value.
 * One field is tested per prompt. Dates assume the test date, 2026-10-02.
 */
export type UnparsedCase = {
  id: string;
  prompt: string;
  /** The limit of the app that makes the prompt fail. */
  why?: string;
  /** The first and last day, both included, as `YYYY-MM-DD`. Null: no time filter. */
  range?: [string, string] | null;
  /** True when the range runs back from today, e.g. "the last 3 months". */
  relative?: boolean;
  /** The maximum cloud cover in percent. Null: no cloud cover filter. */
  cloud?: number | null;
  /** The words that name the place, or `MAP_VIEW`. */
  place?: string;
};

export const UNPARSED_CASES: UnparsedCase[] = [
  { id: "U1", prompt: "Sentinel-2 of Lyon between March 2023 and June 2024", range: ["2023-03-01", "2024-06-30"], why: "A period is one year" },
  { id: "U2", prompt: "Landsat of Cairo since 2020", range: ["2020-01-01", "2026-10-02"], relative: true, why: "No open-ended period" },
  { id: "U3", prompt: "Landsat of Perth from 2018 to 2020", range: ["2018-01-01", "2020-12-31"], why: "No multi-year period" },
  { id: "U4", prompt: "Sentinel-1 of Jakarta on 15 January 2024", range: ["2024-01-15", "2024-01-15"], why: "The smallest unit is a month" },
  { id: "U5", prompt: "Sentinel-2 of Manila in the first two weeks of August 2025", range: ["2025-08-01", "2025-08-14"], why: "The smallest unit is a month" },
  { id: "U6", prompt: "Sentinel-2 of Oslo in the last 3 months", range: ["2026-07-02", "2026-10-02"], relative: true, why: "Relative periods are only 30 days and 12 months" },
  { id: "U7", prompt: "Sentinel-2 of Kyiv over the past week", range: ["2026-09-25", "2026-10-02"], relative: true, why: "Relative periods are only 30 days and 12 months" },
  { id: "U8", prompt: "Sentinel-2 of Lima in the second quarter of 2022", range: ["2022-04-01", "2022-06-30"], why: "Seasons are March to May and so on, not quarters" },
  { id: "U9", prompt: "Sentinel-2 of Toronto from April to September 2023", range: ["2023-04-01", "2023-09-30"], why: "One season or one month only" },
  { id: "U10", prompt: "Landsat of the Aral Sea in 2000", range: ["2000-01-01", "2000-12-31"], why: "Years start at 2015" },
  { id: "U11", prompt: "Landsat of Perth under 12% clouds", cloud: 12, why: "The buckets are 10% and 15%" },
  { id: "U12", prompt: "Sentinel-2 of Accra with at most 35% cloud cover", cloud: 35, why: "The buckets are 30% and 40%" },
  { id: "U13", prompt: "Sentinel-2 of Reykjavik, below 3% clouds", cloud: 3, why: "The strictest bucket is 5%" },
  { id: "U14", prompt: "Landsat of Hanoi with up to 60% clouds", cloud: 60, why: "The buckets are 50% and 75%" },
  { id: "U15", prompt: "Sentinel-2 over Santa Cruz de la Sierra in 2024", place: "Santa Cruz de la Sierra", why: "Five words; a place span has at most four" },
  { id: "U16", prompt: "Sentinel-2 of Ciudad Autónoma de Buenos Aires", place: "Ciudad Autónoma de Buenos Aires", why: "Five words; a place span has at most four" },
];

/**
 * The prompts used to tune the instructions of the extraction models. None
 * of them is in `CASES` or `UNPARSED_CASES`, so the tests stay unseen. Same
 * form as `UNPARSED_CASES`; one prompt can test more than one field.
 */
export const DEV_CASES: UnparsedCase[] = [
  { id: "D1", prompt: "Sentinel-2 of Madrid on 3 May 2022", range: ["2022-05-03", "2022-05-03"] },
  { id: "D2", prompt: "Landsat of Quito since 2019", range: ["2019-01-01", "2026-10-02"], relative: true },
  { id: "D3", prompt: "Sentinel-2 of Berlin over the past 2 months", range: ["2026-08-02", "2026-10-02"], relative: true },
  { id: "D4", prompt: "MODIS of Chile in March 2021", range: ["2021-03-01", "2021-03-31"] },
  { id: "D5", prompt: "NAIP of Ohio in 2021", range: ["2021-01-01", "2021-12-31"] },
  { id: "D6", prompt: "Sentinel-2 of Seoul in autumn 2022", range: ["2022-09-01", "2022-11-30"] },
  { id: "D7", prompt: "Sentinel-2 of Oslo in winter 2023", range: ["2022-12-01", "2023-02-28"] },
  { id: "D8", prompt: "Landsat of Utah last winter", range: ["2025-12-01", "2026-02-28"] },
  { id: "D9", prompt: "latest Sentinel-2 of Cape Town", range: ["2026-09-02", "2026-10-02"], relative: true },
  { id: "D10", prompt: "elevation of the Andes", range: null },
  { id: "D11", prompt: "Sentinel-2 of Quebec with less than 15% clouds", range: null, cloud: 15 },
  { id: "D12", prompt: "soil moisture of Spain", range: null },
  { id: "D13", prompt: "building footprints of Nairobi", range: null },
  { id: "D14", prompt: "Landsat of the Nile delta", range: null, cloud: null },
  { id: "D15", prompt: "orthophotos of Bern", range: null },
  { id: "D16", prompt: "Sentinel-1 of Rotterdam port", range: null },
  { id: "D17", prompt: "Sentinel-2 of Porto in May 2023", range: ["2023-05-01", "2023-05-31"] },
  { id: "D18", prompt: "Landsat of Calgary from June to August 2021", range: ["2021-06-01", "2021-08-31"] },
  { id: "D19", prompt: "Sentinel-2 of Montreal in fall 2020", range: ["2020-09-01", "2020-11-30"] },
  { id: "D20", prompt: "Sentinel-2 of Athens last summer", range: ["2026-06-01", "2026-08-31"] },
  { id: "D21", prompt: "Sentinel-2 of Bogota, clear sky", cloud: 10 },
  { id: "D22", prompt: "Sentinel-2 of Kampala, partly cloudy is fine", cloud: 50 },
  { id: "D23", prompt: "Landsat of Bali in 2022", range: ["2022-01-01", "2022-12-31"], cloud: null },
  { id: "D24", prompt: "Sentinel-2 of Dakar, any cloud cover", cloud: null },
  { id: "D25", prompt: "Sentinel-2 here in June 2024", place: MAP_VIEW, range: ["2024-06-01", "2024-06-30"] },
  { id: "D26", prompt: "NAIP of the Everglades", place: "Everglades" },
  { id: "D27", prompt: "Sentinel-2 of Nantes in the last 6 weeks", range: ["2026-08-21", "2026-10-02"], relative: true },
  { id: "D28", prompt: "Sentinel-1 of Mumbai over the past 10 days", range: ["2026-09-22", "2026-10-02"], relative: true },
  { id: "D29", prompt: "Landsat of Phoenix in the last two years", range: ["2024-10-02", "2026-10-02"], relative: true },
  { id: "D30", prompt: "Sentinel-2 of Ghent in Q3 2021", range: ["2021-07-01", "2021-09-30"] },
  { id: "D31", prompt: "Sentinel-2 of Pune in the first quarter of 2020", range: ["2020-01-01", "2020-03-31"] },
  { id: "D32", prompt: "Landsat of Tucson for the fourth quarter of 2019", range: ["2019-10-01", "2019-12-31"] },
];
