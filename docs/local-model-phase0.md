# Local model, phase 0

Date: 25 November 2026.

This is phase 0 of the plan to replace Jev and Clairic with small open models
that run in the browser. The plan is in
[developmentseed/stac-zap-models](https://huggingface.co/developmentseed/stac-zap-models/blob/main/PLAN.md).
Phase 0 answers three questions before any training:

1. How good are small open encoders with no fine-tuning?
2. Can an encoder make the shortlist of collections, so the decider scores
   only a few options?
3. How long do the encoders take in a browser tab?

It also measures a rules-only extractor, with no model, as the floor for the
extraction model.

## Summary

- **With no fine-tuning, the encoders are far from Jev.** The best one is
  correct on 80 of 172 field-test questions. Jev 1.13 is correct on 166. Thus
  phase 0 only gives the baseline, and training is necessary.
- **The shortlist works with no fine-tuning.** Granite Embedding Small R2
  puts a correct collection in its top 16 for all 28 prompts. The keyword
  score of the app does this for 18 of 28.
- **Rules with chrono-node read most places, times and cloud covers.** They
  are correct on 82 of 85 field-test values (Clairic: 84, Jev: 80) and on 13
  of 16 unparsed requests (Clairic: 16). These numbers are optimistic: the
  rules were written with the test prompts in view.
- **In the browser, the input length and the number of options set the
  time.** The 30M model scores 16 options of 64 tokens in 0.3 s with 4
  threads, and in 0.5 s with 1 thread. 250 options take 4.5 to 16 s. Thus the
  decider must score a shortlist, and the collection embeddings must be made
  before the request.

## The models

All models are Apache-2.0. IBM states that the Granite embedding models are
not trained on MS MARCO, which has a non-commercial licence.

| Model | Parameters | Architecture | Used here as |
|---|---|---|---|
| `ibm-granite/granite-embedding-30m-english` | 30M | RoBERTa, 6 layers | bi-encoder |
| `ibm-granite/granite-embedding-small-english-r2` | 47M | ModernBERT | bi-encoder |
| `ibm-granite/granite-embedding-reranker-english-r2` | 150M | ModernBERT | cross-encoder (reference only, too large for the page) |

The ONNX files are from `onnx-community` (int8, "q8") and, for the reranker,
from `jrc2139/granite-embedding-reranker-english-r2-ONNX` (int8 for ARM64).
The models run with transformers.js, in Node.js for the tests and in Chromium
for the times.

## Test 1: the field test with no fine-tuning

`eval/fields.ts` with `PROVIDER=local`. The scorer is in
`eval/local/decide.ts`. A bi-encoder embeds the query and each option, and
the probabilities are a softmax of the cosine similarities (temperature
0.02). The cross-encoder scores each (query, option) pair. The query is the
request ("prompt"), or the question text and the request ("full").

| Model | Query | Catalog | Collection | Place | Time period | Part of year | Cloud cover | Total |
|---|---|---|---|---|---|---|---|---|
| Jev 1.13 (for comparison) | | 28/30 | 27/27 | 30/30 | 26/30 | 30/30 | 25/25 | **166/172** |
| Granite reranker R2 | prompt | **25/30** | **17/27** | 0/30 | 16/30 | 21/30 | 1/25 | 80/172 |
| Granite small R2 | full | 21/30 | 13/27 | 1/30 | 16/30 | 21/30 | 1/25 | 73/172 |
| Granite small R2 | prompt | 17/30 | 16/27 | 0/30 | 17/30 | 21/30 | 1/25 | 72/172 |
| Granite 30M | full | 11/30 | 12/27 | 1/30 | 16/30 | 21/30 | 18/25 | 69/172 |
| Granite 30M | prompt | 10/30 | 13/27 | 0/30 | 17/30 | 22/30 | 1/25 | 63/172 |

What the results show:

- The models with no fine-tuning know the topic of a catalog or collection,
  but not the questions of the app. For example, they do not know when to
  select `keep`, and they do not select the words of the place.
- Most correct answers on the time period and the part of the year are for
  requests with no time. The model selects `keep` or no filter because the
  probabilities are spread out.
- The place question needs a different model. Its options are groups of
  words from the request. In the plan, a tagger reads the place words. The
  decider only selects among the places that Nominatim finds for them.

## Test 2: the shortlist

`eval/local/recall.ts` ranks all options for each field-test prompt: all 13
catalogs, and all collections of the expected catalog (up to 513, for
data.geo.admin.ch). The table gives the number of prompts with a correct
answer in the top k. "Worst" is the rank of the worst prompt.

| Ranker | Catalog @1 | @5 | @16 | Collection @1 | @5 | @16 | @32 | Worst collection rank |
|---|---|---|---|---|---|---|---|---|
| Keyword score of the app (`shortlist`) | 13/30 | 25/30 | 30/30 | 3/28 | 13/28 | 18/28 | 25/28 | 135 |
| Granite 30M | 10/30 | 18/30 | 30/30 | 14/28 | 21/28 | 23/28 | 26/28 | 66 |
| **Granite small R2** | **17/30** | **27/30** | 30/30 | **17/28** | **26/28** | **28/28** | 28/28 | **10** |

The keyword score of the app only orders the collections of a catalog with
more than 250 collections. It is not made for a short list.

Granite small R2 with no fine-tuning is a good retriever: the decider can
score its top 16 and lose no correct collection on this test. 28 prompts is
a small test. Phase 3 measures the retriever again on the held-out catalogs.

## Test 3: the rules-only extractor

`eval/local/extract.ts` answers the fields of the extractions API with no
model:

- Time: chrono-node finds dates such as "March 2023" or "15 January 2024". A
  regular expression finds years. The time words run from the first to the
  last date, with a word before them such as "since" or "between". Then the
  time rules of the app make the range for seasons, quarters, "the last N
  days" and "recent".
- Cloud cover: a percent number in a request that mentions clouds, or the
  cloud words of the app, for example "clear sky" is 10%.
- Place: words that start with a capital letter after "over", "of", "in" or
  "near". Words that name data, such as "Sentinel" or "Landsat", are not a
  place.

`MODEL=local:rules VARIANT=compact APP_CALLS=1 pnpm tsx eval/extract.ts`, with
`SET=field`, `unparsed` and `dev`:

| Extractor | Field test: place | Time | Cloud cover | Total | Unparsed requests | Dev requests |
|---|---|---|---|---|---|---|
| Clairic, compact, app calls | 30/30 | 29/30 | 25/25 | **84/85** | **16/16** | 36/36 |
| **Rules only** | 29/30 | 28/30 | 25/25 | 82/85 | 13/16 | 35/36 |
| Jev 1.13 (options) | 30/30 | 25/30 | 25/25 | 80/85 | 0/16 | |

**The numbers of the rules are optimistic.** The rules were written in one
pass, without tuning on the results, but with the test prompts in view, for
example "mostly clear" and "not too cloudy". The new hand-written test set of
the plan will give a fair number.

The rules fail on these requests:

| Request | Rules | Correct |
|---|---|---|
| NAIP imagery of Austin, Texas from 2022 | 2022 to today | 2022 |
| Landsat imagery from March 2019 of Lake Titicaca | March 2019 to today | March 2019 |
| Sentinel-2 of Toronto from April to September 2023 | April 2023 to today | April to September 2023 |
| Landsat of Calgary from June to August 2021 | June 2021 to today | June to August 2021 |
| Sentinel-2 of Manila in the first two weeks of August 2025 | August 2025 | 1 to 14 August 2025 |
| Maxar imagery after Hurricane Idalia | Hurricane Idalia | no place |
| Sentinel-2 over Santa Cruz de la Sierra in 2024 | Santa Cruz | Santa Cruz de la Sierra |

Most failures are about what the words mean: "from 2022" can mean "in 2022"
or "since 2022", and a capital letter does not make a place. These are the
decisions for the tagger. Code is good at the dates after the words are
known.

## Test 4: time in the browser

`eval/local/bench/` loads a model in headless Chromium (ONNX Runtime Web,
WASM, int8, page cross-origin isolated for threads) and times the work of
one request. The computer is an 8-core ARM Chromebook (Linux container), CPU
only.

| Model | Threads | Tokens per input | Load | 1 query | 16 pairs | 250 texts |
|---|---|---|---|---|---|---|
| Granite 30M | 1 | 128 | 3.5 s | 9 ms | 1.0 s | 15.9 s |
| Granite 30M | 4 | 128 | 3.4 s | 10 ms | 0.58 s | 9.2 s |
| Granite 30M | 1 | 64 | 3.3 s | 9 ms | 0.52 s | 7.8 s |
| **Granite 30M** | **4** | **64** | 3.5 s | 11 ms | **0.30 s** | 4.5 s |
| Granite small R2 | 1 | 128 | 4.3 s | 46 ms | 2.7 s | 41.7 s |
| Granite small R2 | 4 | 128 | 4.6 s | 19 ms | 1.5 s | 24.3 s |
| Granite small R2 | 1 | 64 | 4.3 s | 40 ms | 1.3 s | 20.7 s |
| Granite small R2 | 4 | 64 | 4.4 s | 21 ms | 0.81 s | 12.3 s |

"Load" includes the download from the Hugging Face Hub (granite 30M int8:
about 30 MB). "16 pairs" is the decider on a shortlist. "250 texts" is the
cost to embed the collections of a large catalog, or to score all of them
with no shortlist.

What the results show:

- **The time of the decider is about linear in the tokens.** Short option
  texts (the title and about 120 characters, as the app shows them to Jev)
  halve the time.
- **The R2 model is two to three times slower than the 30M model** in WASM,
  although it has only 1.6 times the parameters.
- **Embedding a query is fast** (10 to 50 ms). Embedding 250 collections is
  not. Thus the embeddings of the curated catalogs are made at build time,
  and the embeddings of STAC Index catalogs are made by a scheduled job and
  shipped with the snapshot, not in the page.
- A zap has three decider calls (catalog, collection, place). With the 30M
  model, 16 options and 64 tokens, the three calls take about 1.6 s with
  one thread and 0.9 s with four. The Jev calls take about 1 s today.

## Decisions for the next phases

1. **Retriever:** Granite small R2 as the base. Embeddings of the options
   are made before the request; only the query is embedded in the page.
2. **Decider:** Granite 30M as the base, fine-tuned as a cross-encoder on 8
   to 16 options of at most 64 tokens. Phase 3 also tests a fine-tuned
   bi-encoder as the decider for the catalog and the collection, which takes
   only one query embedding.
3. **Extractor:** a tagger on Granite 30M, with chrono-node and the rules of
   `eval/local/extract.ts` for the values. The main work of the tagger is
   the meaning of words such as "from" and "after", and which capitalized
   words are a place.
4. **Test set:** the new hand-written test set comes before the training
   (phase 1), because the field-test prompts are now known.

## Run the tests

```shell
# Field test, local models with no fine-tuning (QUERY=full for the question text in the query)
PROVIDER=local MODEL=bi:onnx-community/granite-embedding-small-english-r2-ONNX pnpm tsx eval/fields.ts
# Shortlist
pnpm tsx eval/local/recall.ts bi:onnx-community/granite-embedding-30m-english-ONNX bi:onnx-community/granite-embedding-small-english-r2-ONNX
# Rules-only extractor
SET=unparsed MODEL=local:rules VARIANT=compact APP_CALLS=1 pnpm tsx eval/extract.ts
# Browser times
pnpm vite --config eval/local/bench/vite.config.ts &
pnpm tsx eval/local/bench/run.ts
```

The results are in `eval/results/phase0/`.
