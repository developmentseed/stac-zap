# Licence review for the local models

Date: 25 November 2026. Phase 1 of the plan
([PLAN.md](https://huggingface.co/developmentseed/stac-zap-models/blob/main/PLAN.md)).

The models, the training data and the code must be open source. This file
lists each input, its licence, and the decision. "To confirm" marks a point
that needs a person to read the terms.

## Rule

No output of a closed model is a label, a target, a paraphrase or a request
in the training data or the test data. This includes Jev (TypeSafe), Clairic
(Cloudflare), and Claude. The Jev and Clairic results in `eval/results/` are
baselines to compare against, nothing more.

## Base models

| Model | Licence | Training data | Decision |
|---|---|---|---|
| `ibm-granite/granite-embedding-30m-english` | Apache-2.0 | IBM: "we do not use the popular MS-MARCO retrieval dataset in our training corpus due to its non-commercial license"; data from a governance and clearance process | Use (decider, extractor) |
| `ibm-granite/granite-embedding-small-english-r2` | Apache-2.0 | Same statement as above | Use (retriever) |
| `ibm-granite/granite-embedding-reranker-english-r2` | Apache-2.0 | Same family | Phase 0 comparison only (150M, too large for the page) |
| `cross-encoder/ms-marco-*`, most small rerankers | Apache-2.0 | Fine-tuned on MS MARCO (non-commercial) | Do not use |

The ONNX files of the released models are exported by this project from the
IBM weights, not taken from third-party conversions.

## Generator and judge models (OpenRouter)

| Model | Role | Licence | Decision |
|---|---|---|---|
| `deepseek/deepseek-v4.1-flash` (`deepseek-ai/DeepSeek-V4.1-Flash`) | Generator: paraphrases, collection phrases, topic requests | MIT | Use |
| `qwen/qwen3.8-27b` (`Qwen/Qwen3.8-27B`) | Judge, and second generator | Apache-2.0 | Use |
| `minimax/minimax-m3` | Judge in gazet | MiniMax Community License: commercial use of the model and its derivatives needs a notice or an authorization | Do not use |
| Models with no open weights on OpenRouter (for example `qwen/qwen3.7-flash`, `qwen/qwen3.8-max-*`) | | Closed | Do not use |

A judge only says keep or drop. Its text is not kept in the data. Even so,
the judge is open-weight, so the selection of the data is reproducible with
open models.

### Providers

The licence of the weights is not enough: the terms of the company that runs
the model also apply. Z.ai's terms forbid training other models on the
outputs; in gazet, 210 rows served by Z.ai had to be removed and the models
trained again.

- Each call sets `provider.only` to an allowlist of third-party inference
  hosts, with `allow_fallbacks: false` and fp8 or better:
  DeepInfra, Parasail, Nebius, Together, Fireworks, CoreWeave, Crusoe.
  These hosts served gazet's published data. **To confirm:** their current
  terms on the ownership and the use of outputs.
- First-party APIs of model makers are not used (Z.ai, DeepSeek, Alibaba,
  MiniMax, Moonshot, Baidu), because their terms can restrict the use of
  outputs.
- Each generated row records the model, the provider and the date.

## Data sources

| Source | Use | Licence | Decision |
|---|---|---|---|
| STAC APIs: the 13 curated catalogs and the STAC Index APIs that answer (`training/snapshot.ts`) | Collection ids, titles, descriptions (cut to 400 characters), keywords, providers, extent | The `license` field of a collection is the licence of the **data**, not of the metadata. Most APIs state no licence for the metadata | Use, with the URL of each catalog. **To confirm:** publishing descriptions. Fallback: publish ids and titles only, and the script that downloads the descriptions |
| [STAC Index](https://stacindex.org) | The list of public STAC APIs | No licence stated | Use the list of URLs only |
| [GeoNames](https://www.geonames.org) `cities15000`, `admin1CodesASCII`, `countryInfo` | Place names, countries, populations; same-name candidates for the place pick | CC BY 4.0 | Use, with attribution |
| [Natural Earth](https://www.naturalearthdata.com) 10m physical features (from gazet's index, `developmentseed/gazet-sql-data`) | Names of rivers, lakes, mountain ranges, deserts, seas, islands | Public domain | Use |
| Overture divisions | | ODbL for divisions (share-alike) | Not used, so the dataset has no share-alike part |
| Nominatim / OpenStreetMap | At run time only, as today | ODbL; the usage policy forbids bulk queries | Not used to make training data |
| stac-zap test prompts (`eval/cases.ts`, 82 prompts) | Test only, never training | Written by people at Development Seed | Use as test data |

## Software

| Package | Use | Licence |
|---|---|---|
| chrono-node | Dates from time words | MIT |
| @huggingface/transformers (transformers.js) | Models in Node.js and the browser | Apache-2.0 |
| onnxruntime-web, onnxruntime-node | ONNX inference | MIT |

## Attribution for the dataset card

- Place names: GeoNames (CC BY 4.0), Natural Earth (public domain).
- Collection metadata: the STAC APIs listed in `catalogs.jsonl`, with their URLs.
- Generated text: DeepSeek-V4.1-Flash (MIT) and Qwen3.8-27B (Apache-2.0), through the providers recorded in each row.
