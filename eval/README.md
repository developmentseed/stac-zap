# Decision model tests

These scripts compare decision models on the questions that stac-zap asks.
A decision model gets a text and a set of questions with fixed options. It
returns one option per question and a probability for each option. The
results and the conclusions are in
[docs/decision-models.md](../docs/decision-models.md).

The scripts import the question code of the app (`src/zap`). Thus, the tests
ask the same questions as the app.

## Tests

| Script | Test |
|---|---|
| `fields.ts` | The field test. For each prompt in `cases.ts`, the model answers the six questions of the app: catalog, collection, place, time period, part of the year and maximum cloud cover. The script scores each answer as the app applies it. |
| `world.ts` | The world knowledge test. The model finds 38 places with latitude and longitude bands as the options. The script measures the distance from the true point to the chosen bands. |
| `summary.ts` | Prints the results of a run as Markdown tables. |
| `heatmap.ts` | Draws the field test results as an SVG heatmap. |
| `snapshot.ts` | Saves the catalogs and their collections to `data/stac-snapshot.json`. |

## Before you start

1. Install the packages with `pnpm install`.
2. Put the API keys in `.env` at the root of the repository:
   - `OPENROUTER_API_KEY` for the models on OpenRouter.
   - `CODIV_API_KEY` for the models on Codiv.

## Run a test

The `PROVIDER` and `MODEL` variables select the model. `PROVIDER` is
`openrouter` (the default) or `codiv`.

```bash
PROVIDER=openrouter MODEL=typesafe/jev-1.13 pnpm tsx eval/fields.ts
PROVIDER=codiv MODEL=openjev-latest pnpm tsx eval/world.ts
```

The scripts write the results to `eval/results/<date>/`. To write to a
different folder, set `RESULTS`. To ask only some fields again, set `FIELDS`,
for example `FIELDS=cloud`. The script then replaces only these fields in the
result file.

Then print the tables and draw the heatmap:

```bash
pnpm tsx eval/summary.ts eval/results/2026-10-02
pnpm tsx eval/heatmap.ts eval/results/2026-10-02 docs/decision-models-heatmap.svg
rsvg-convert -w 2400 docs/decision-models-heatmap.svg -o docs/decision-models-heatmap.png
```

## Rate limits

The scripts wait between calls to stay below the rate limits:

- Codiv: 60 requests per minute.
- OpenRouter free models (`:free`): 20 requests per minute.

Do not run two Codiv models at the same time. They share the rate limit.

## The catalog snapshot

The collection question offers the collections of a catalog. Catalogs change
over time. Thus, `fields.ts` reads the collections from
`data/stac-snapshot.json` and not from the catalogs. To update the snapshot,
run `pnpm tsx eval/snapshot.ts`. Compare results only if the runs used the
same snapshot.

## The expected answers

`cases.ts` holds the expected answers. A person wrote them. For some prompts,
more than one answer is correct. For example, "cloud-free imagery" accepts a
maximum of 5% or 10%. The expected answers assume that today is 2 October
2026. Thus, "last summer" is the summer of 2026.
