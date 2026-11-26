# Local open model

Date: 26 November 2026 (updated for the shared model v7). The first report is [local-model-phase0.md](local-model-phase0.md). The plan is in
[developmentseed/stac-zap-models](https://huggingface.co/developmentseed/stac-zap-models/blob/main/PLAN.md).

stac-zap can now answer a request with open models that run in the browser, with no OpenRouter call. The model list
below the search bar selects "Jev + Clairic (OpenRouter)" or "Local open model (in the browser)". See the README,
"Local open model", to run it.

![The local model answers "Sentinel-2 over Lisbon last summer with few clouds"](local-model-demo.png)

## Design

**One model**: IBM Granite Embedding 30M (Apache-2.0), fine-tuned, with three outputs (v6 and later). The page
downloads it once: `model_q4.onnx.gz`, 19.3 MB on the wire (int4 embedding table, int8 weights, gzip, unzipped in the
browser), the tokenizer (0.7 MB), the settings and `vectors.json` (0.3 MB, the vectors of the 1,021 curated
collections), plus ONNX Runtime Web (5.6 MB, jsDelivr) and transformers.js. It loads from the site or straight from
the public model repo on the Hugging Face Hub.

| Output | Job |
|---|---|
| retriever (`last_hidden_state`, CLS) | Shortlists 16 collections of all known catalogs for the request. |
| decider (`decider_logits`) | Picks one of the 16, or `keep`. **The catalog follows from the collection**: there is no catalog question. The pick combines the decider and the retriever: the decider's probability times the retriever's softmax over the shortlist, to the power 0.5. Also picks the place among the Photon results. |
| tagger (`tag_logits`) | Marks the place, time and cloud cover words. Code makes the values: `src/zap/local/rules.ts` with chrono-node. |

Training (`training/train_hf_jobs.py`): stage 1 trains the encoder as a retriever (1 epoch, contrastive, with hard
negatives from Granite R2); stage 2 trains the decider, the tagger and the retriever together (2 epochs, learning rate
3e-5). One HF Jobs run on an L4 GPU, 24 minutes.

Training data: requests from templates with exact labels (`training/generate.ts`), from the STAC snapshot and
GeoNames and Natural Earth places; no output of a closed model. Common data has groups: "Sentinel-2", "Landsat",
"elevation" and so on accept every main product, with Planetary Computer preferred; "aerial photos" or "lidar" of a
place accept the collections that cover the place.

## Results

Field test (30 requests, curated catalogs) and the extraction tests of `eval/cases.ts`, in Node with the int8 files
the page loads (`eval/local/pipeline.ts`). test-claude-v1: 240 requests that Claude wrote; it shares the habits of
the generator, so its scores are optimistic.

| | Jev 1.13 | Clairic | v5: three models (95 MB) | v6: shared | v7: shared | **v7 + retriever vote (demo)** |
|---|---|---|---|---|---|---|
| Field test: catalog | 28/30 | | 18/30 | 10/30 | 19/30 | **26/30** |
| Field test: catalog and collection | 27/27 | | 17/28 | 10/28 | 18/28 | **24/28** |
| Field test: place, time, cloud | 80/85 | 84/85 | 82/85 | 81/85 | 80/85 | 80/85 |
| Requests the option lists cannot hold | 0/16 | 16/16 | 15/16 | 15/16 | 15/16 | 15/16 |
| test-claude-v1: catalog | | | 202/240 | 173/240 | 182/240 | 196/240 |
| test-claude-v1: catalog and collection | | | 162/240 | 152/240 | 150/240 | **169/240** |

The weight of the retriever vote: 0.25 gave 24/30 and 23/28 on the field test, 0.5 gave 26/30 and 24/28, 1.0 gave
21/30 and 18/28. v6 trained longer (2 + 3 epochs, learning rate 5e-5): better on the generator's own dev split, worse
on everything else.

In the browser (headless Chromium, ARM Chromebook, one thread), ten requests in a row opened the expected collection
nine times: Sentinel-2 Lisbon (Planetary Computer), SWISSIMAGE for Zurich, Copernicus DEM for Mount Everest, Landsat
Perth (≤ 12%), "now in 2021" (keeps Landsat, sets 2021), Hurricane Idalia, "same for Hurricane Ian", ESA WorldCover
for Kenya, and the Lower Saxony orthophotos for "aerial photos of Hannover". "Crop types in Lower Saxony" opened a
Swiss soil layer. A zap takes 5 to 15 s, with the geocoder and the STAC requests.

## Download size

Measured in Chromium, bytes on the wire for the model file (`scripts/measure-download.ts`):

| File | On the wire | Field test: catalog / both | test-claude: both |
|---|---|---|---|
| `model_quantized.onnx`: weight-only int8, per channel | 31.4 MB | 26/30 · 24/28 | 169/240 |
| **`model_q4.onnx.gz`: int4 embedding table (blocks of 32, fp16 scales), int8 weights, gzip** | **19.3 MB** | 25/30 · 23/28 | 163/240 |

62% of the int8 file is the embedding table (50,265 word pieces × 384); it takes int4 well. int4 in the layer weights
broke the model: the decider's top answer agreed with fp32 on 13 to 29% of questions
(`training/tools/weight_int4.py`). The Hub sends model files without compression, so the page fetches a gzip copy
and unzips it with `DecompressionStream` (`worker.ts`). Brotli would save no more: int8 and int4 weights hardly
compress (31.4 MB → 27.4 MB with brotli, 27.8 MB with gzip).

## Problems found and fixed

1. **Dynamic int8 broke the models** (ONNX Runtime `quantize_dynamic` also makes the activations int8). The decider
   agreed with fp32 on 47% of questions. Weight-only int8 (`training/tools/weight_only.py`): 95%, the same size.
   The model computes in fp32, so it is slower than the phase 0 bench.
2. **The generator gave generic names to every matching collection**: "Sentinel-2" pointed to dozens of mosaics, each
   the only correct answer. Groups fixed it (field-test collection 12/28 → 22/28).
3. **After the first zap the model kept the open collection** for every request, as it saw `keep` mostly in
   follow-ups. Training now adds an open collection to 40% of new requests, and the page asks twice: first as a new
   request; only when no collection gets 75% does it offer `keep`.
4. **Long current collection ids** made the query longer than the model input; pairs are now cut longest first, as
   in the page.
5. **One shared model lost accuracy** (v6). Lighter training (v7) and the retriever vote fixed it.
6. **Loading from the Hub**: transformers.js reads a URL only in its remote mode; `worker.ts` sets the host and the
   path template for a base that starts with `https://`.

## Limits

- **Place pick**: it follows the order of Photon. "Perth" gives Perth, Scotland, and "the Matterhorn" a hill in
  Connecticut. Photon gives no population, so the model has no sense of how important a place is.
- **Topic requests**: "crop types in Lower Saxony" opens a Swiss soil layer; "drone imagery", "wildfires" are weak.
- **Tagger**: "only cloud-free" with no other words is not read; "Mount Everest" became "Mount" (the geocoder still
  found it).
- **English only**: the 30M model does not read the German and French titles of the Swiss catalog well.
- **Download**: about 27 MB on the first visit (19.3 MB of it the model), then from the browser cache.

## Next

- Topic and paraphrase requests from open models (phase 2), for "drone imagery", "wildfires", "radar images".
- Place pick: Photon's importance or the place type in the option text.
- A test set written by people.
