# stac-zap

Zap mode for [stac-map](https://github.com/developmentseed/stac-map). Type one
request, for example "recent cloud-free Sentinel-2 over Lisbon". The app
opens the catalog and the collection, and searches for the place, the time and
the cloud cover.

The [jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
decision model does not write text. It selects one option from each list that
the app gives it, and it gives a probability for each option. Thus, each search
is a valid STAC query. The panel above the prompt shows each selection and its
probability.

stac-map is an npm dependency. This app does not change it.

## Run

1. Install the dependencies:

   ```shell
   pnpm install
   ```

2. Put your OpenRouter key in `.env`:

   ```shell
   OPENROUTER_API_KEY=sk-or-...
   ```

3. Start the dev server:

   ```shell
   pnpm dev
   ```

The dev server reads the key from `.env` for each request. It sends the key
only to OpenRouter. The key does not go into the browser bundle.

## Deploy

The site is static, on GitHub Pages. A Cloudflare Worker (`worker/`) keeps
the OpenRouter key and forwards the jev calls. The Worker:

- accepts only requests from the site, `https://developmentseed.org`,
- forwards only to the OpenRouter decisions API, always with the jev model,
- accepts about 30 requests each minute from one visitor.

Give the OpenRouter key a budget limit. The budget limit is the last check.

To deploy the Worker, do these steps one time:

```shell
cd worker
npx wrangler login
npx wrangler secret put OPENROUTER_API_KEY
npx wrangler deploy
```

`wrangler secret put` asks for the key. `wrangler deploy` shows the Worker
URL. Set this URL as the `DECIDE_URL` repository variable (not a secret),
then run the "Deploy to GitHub Pages" workflow again:

```shell
gh variable set DECIDE_URL --repo developmentseed/stac-zap --body https://stac-zap-decide.<account>.workers.dev
gh workflow run pages.yml --repo developmentseed/stac-zap
```

Each push to `main` deploys the site again.

## How it works

### Catalogs

By default, jev selects from 14 curated STAC APIs. For each of these
catalogs, stac-map can show the data of sampled collections in a browser.
The list is in `src/zap/catalogs.ts`.

The "All STAC Index catalogs" switch below the search bar adds the public
APIs of [STAC Index](https://stacindex.org). stac-map cannot always show the
data of these catalogs. Some block requests from other sites (CORS), and some
need a login for their data.

When the page opens, the app gets the collections of each catalog. It removes
the catalogs that do not answer. It adds the collection titles to the
description of each catalog, so jev knows what data each catalog has.

### Calls

One request makes two or three calls to jev:

1. jev selects the catalog.
2. The app gets the collections of that catalog. Then jev selects the
   collection, the time period, the part of the year and the maximum cloud
   cover. jev also selects the words of the request that name the place, for
   example "Lisbon". The options are all the groups of one to four words in
   the request.
3. The app sends these words to [Nominatim](https://nominatim.org/), the
   OpenStreetMap geocoder. Then jev selects the place that the request means
   from the places that Nominatim finds. For example, for "elevation around
   Mount Everest", jev selects the peak in Nepal and not a place with the
   same name in Florida. If no place fits, jev selects "None of these", and
   the app keeps the last place.

Nominatim gives the bounding box of the place. The app uses it for the search
and for the map view. A point, for example a peak, gets a box of 0.2 degrees.

Each question has a `keep` option. A field changes only when jev selects an
option that is not `keep` and the probability is 0.5 or more.

The app then writes the search parameters to the stac-map store and opens the
collection. stac-map runs the search.

| File | Contents |
| --- | --- |
| `src/zap/decide.ts` | The jev payload and the answers. A port of the awii zap mode. |
| `src/zap/catalogs.ts` | The curated catalogs, STAC Index, and the collections of each catalog. |
| `src/zap/options.ts` | The time and cloud cover options. |
| `src/zap/geocode.ts` | The Nominatim search, and the groups of words that can name a place. |
| `src/zap/run.ts` | The two jev calls, and the changes to the stac-map store. |
| `src/ZapBar.tsx` | The prompt and the probability panel. |
| `vite.config.ts` | The `/api/decide` endpoint, which adds the key. |

## Limits

- The dev server is the only server that has `/api/decide`. A static
  deployment does not have it.
- The public Nominatim server permits one request each second. The app waits
  between requests. For a public demo with many users, use a different
  geocoder server. Change `SEARCH_URL` in `src/zap/geocode.ts`.
- The app does not send a Planetary Computer asset URL without a token.
  `src/planetary-computer.ts` adds a SAS token to each asset request, because
  stac-map adds tokens only to thumbnails.
- stac-map reads the search parameters only when the search panel opens. Thus,
  to search the same collection again, the app closes the collection and opens
  it again.
