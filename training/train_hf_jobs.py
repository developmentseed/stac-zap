# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.4",
#   "transformers>=4.46",
#   "huggingface_hub>=0.26",
#   "onnx>=1.17",
#   "onnxruntime>=1.20",
#   "numpy",
# ]
# ///
"""Trains the local models of stac-zap and exports them as int8 ONNX.

- decider: ibm-granite/granite-embedding-30m-english as a cross-encoder. It scores (question, option) pairs; a
  softmax over the options of one question gives the probabilities, as Jev does. Questions: catalog, collection
  (on the shortlist of the retriever), place pick. Each question has `keep` when the map has a current value.
- tagger: the same base as a token classifier: BIO tags for the place, time and cloud cover words.
- retriever: ibm-granite/granite-embedding-small-english-r2 with no fine-tuning (phase 0: 28/28 in the top 16). It
  makes the collection shortlists here, as the app will.

Data: the dataset developmentseed/stac-zap-data (generated/v0, snapshot, places, testset). Output: the model repo
developmentseed/stac-zap-models, folders decider/<run>/ and tagger/<run>/ in the layout transformers.js reads
(config.json, tokenizer files, onnx/model.onnx, onnx/model_quantized.onnx), and report-<run>.json.

Run on HF Jobs:
  hf jobs uv run --flavor l4x1 --timeout 2h --secrets HF_TOKEN --detach training/train_hf_jobs.py --run v1
Smoke test on a laptop CPU:
  uv run training/train_hf_jobs.py --run smoke --max-records 300 --epochs 1 --no-upload --device cpu
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import onnx
from huggingface_hub import HfApi, hf_hub_download
from onnx import helper, numpy_helper
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

DATA_REPO = "developmentseed/stac-zap-data"
MODEL_REPO = "developmentseed/stac-zap-models"
BASE = "ibm-granite/granite-embedding-30m-english"
RETRIEVER = "ibm-granite/granite-embedding-small-english-r2"
SHORTLIST = 16
TAGS = ["O", "B-PLACE", "I-PLACE", "B-TIME", "I-TIME", "B-CLOUD", "I-CLOUD"]
KEEP = "keep"
NONE = "none"

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--epochs", type=int, default=2)
p.add_argument("--tag-epochs", type=int, default=3)
p.add_argument("--lr", type=float, default=5e-5)
p.add_argument("--groups-per-batch", type=int, default=16)
p.add_argument("--max-len", type=int, default=96)
p.add_argument("--max-records", type=int, default=0)
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--no-upload", action="store_true")
p.add_argument("--seed", type=int, default=13)
p.add_argument("--data", default="v1", help="folder of generated/ in the dataset")
args = p.parse_args()
random.seed(args.seed)
torch.manual_seed(args.seed)
dev = torch.device(args.device)
OUT = Path(os.environ.get("OUT_DIR", "/tmp/stac-zap-train")) / args.run
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)


# ---------------------------------------------------------------- data


def fetch(path):
    return hf_hub_download(DATA_REPO, path, repo_type="dataset")


def jsonl(path):
    return [json.loads(line) for line in open(fetch(path))]


snapshot = json.load(open(fetch("snapshot/stac-2026-11-25.json")))
places = {pl["id"]: pl for pl in json.load(open(fetch("places/places.json")))}
splits = {s: jsonl(f"generated/{args.data}/{s}.jsonl") for s in ["train", "dev", "test-catalog"]}
splits["test-claude"] = jsonl("testset/test-claude-v1.jsonl")
if args.max_records:
    splits = {k: v[: args.max_records] for k, v in splits.items()}
log({k: len(v) for k, v in splits.items()})

HELD_OUT = {"kagis-katalog", "digital-earth-au", "worldpop-stac-api"}
catalogs = {c["id"]: c for c in snapshot["catalogs"]}
curated = [c["id"] for c in snapshot["catalogs"] if c.get("curated")]
collections = {cat: {c["id"]: c for c in cols} for cat, cols in snapshot["collections"].items()}


def short(text, n):
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def catalog_text(cid):
    return f"{cid}: {short(catalogs[cid]['description'], 300)}"


def collection_text(cat, cid):
    c = collections[cat].get(cid, {"id": cid})
    return f"{short(c.get('title') or cid, 100)} ({cid}): {short(c.get('description'), 160)}"


def place_text(pid):
    if pid == NONE:
        return "None of these places"
    pl = places[pid]
    where = ", ".join(x for x in [pl.get("region"), pl.get("country")] if x)
    kind = "city" if pl["kind"] == "city" else pl["type"]
    pop = f", {pl['population']:,} people" if pl.get("population") else ""
    return f"{pl['name']}{', ' + where if where else ''} ({kind}{pop})"


def query(question, r, current=None):
    return f"{question} | request: {r['request']} | current: {current or 'not set'}"


# ---------------------------------------------------------------- retriever shortlist


@torch.no_grad()
def embed(model, tok, texts, bs=128):
    out = []
    for i in range(0, len(texts), bs):
        x = tok(texts[i : i + bs], padding=True, truncation=True, max_length=256, return_tensors="pt").to(dev)
        h = model(**x).last_hidden_state[:, 0]
        out.append(F.normalize(h.float(), dim=-1).cpu())
    return torch.cat(out) if out else torch.zeros(0, 384)


log("retriever: embedding collections and requests")
rtok = AutoTokenizer.from_pretrained(RETRIEVER)
rmodel = AutoModel.from_pretrained(RETRIEVER).to(dev).eval()
col_keys = [(cat, cid) for cat, cols in collections.items() for cid in cols]
col_vecs = embed(rmodel, rtok, [collection_text(cat, cid) for cat, cid in col_keys])
col_index = {}
for i, (cat, cid) in enumerate(col_keys):
    col_index.setdefault(cat, []).append(i)
all_records = [r for s in splits.values() for r in s]
req_vecs = embed(rmodel, rtok, [r["request"] for r in all_records])
req_vec = {id(r): req_vecs[i] for i, r in enumerate(all_records)}
del rmodel
torch.cuda.empty_cache() if dev.type == "cuda" else None


def shortlist(r, cat, k=SHORTLIST):
    idx = col_index.get(cat, [])
    if not idx:
        return []
    sims = col_vecs[idx] @ req_vec[id(r)]
    order = torch.argsort(sims, descending=True)[:k].tolist()
    return [col_keys[idx[j]][1] for j in order]


# ---------------------------------------------------------------- questions


def questions_of(r, train):
    """The decider questions of a record: (kind, query, [(option id, text)], set of correct ids)."""
    qs = []
    cur = r.get("current") or {}
    # Catalog. Training uses curated-only lists half of the time, as the app does by default.
    cat_target = r["catalog"]["target"]
    accepted = set(r["catalog"]["accepted"]) or ({cat_target} if cat_target else set())
    pool = [c for c in catalogs if not (train and c in HELD_OUT)]
    if train and cat_target in curated and random.random() < 0.5:
        pool = [c for c in curated if c not in HELD_OUT]
    if train and len(pool) > SHORTLIST:
        keep_ok = [c for c in pool if c in accepted]
        others = random.sample([c for c in pool if c not in accepted], SHORTLIST - len(keep_ok))
        pool = keep_ok + others
        random.shuffle(pool)
    opts = [(c, catalog_text(c)) for c in pool]
    if cur.get("catalog"):
        opts.append((KEEP, f"keep: keep the current catalog ({cur['catalog']})"))
    gold = (accepted & set(pool)) if cat_target else ({KEEP} if cur.get("catalog") else set())
    if gold:
        qs.append(("catalog", query("Which STAC catalog?", r, cur.get("catalog")), opts, gold))

    # Collection, on the shortlist of the catalog of the target (or the current catalog).
    cat = cat_target or cur.get("catalog")
    if cat in collections:
        ids = shortlist(r, cat)
        col_ok = set(r["collection"]["accepted"]) or ({r["collection"]["target"]} if r["collection"]["target"] else set())
        if train and col_ok and not col_ok & set(ids):
            ids = ids[:-1] + [random.choice(sorted(col_ok))]
        opts = [(c, collection_text(cat, c)) for c in ids]
        if cur.get("collection"):
            opts.append((KEEP, f"keep: keep the current collection ({cur['collection']})"))
        gold = (col_ok & set(ids)) if r["collection"]["target"] else ({KEEP} if cur.get("collection") else set())
        if gold or not train:
            qs.append(("collection", query("Which collection?", r, cur.get("collection")), opts, gold))

    # Place pick: the places with the words of the request.
    pl = r["place"]
    if pl.get("target") and pl.get("candidates"):
        cands = [c for c in pl["candidates"] if c in places]
        # The generator keeps the 6 largest places of a name; a smaller target must still be an option.
        if pl["target"] not in cands and pl["target"] in places:
            cands.append(pl["target"])
        gold = {pl["target"]}
        if train and random.random() < 0.15:
            cands = [c for c in cands if c != pl["target"]]
            gold = {NONE}
        random.shuffle(cands) if train else None
        opts = [(c, place_text(c)) for c in cands] + [(NONE, place_text(NONE))]
        qs.append(("place", query(f"Which place is '{pl['words']}'?", r), opts, gold))
    return qs


# ---------------------------------------------------------------- decider


tok = AutoTokenizer.from_pretrained(BASE)


def encode_pairs(q, texts):
    return tok([q] * len(texts), texts, padding=True, truncation="only_second", max_length=args.max_len, return_tensors="pt")


def group_logits(model, groups):
    """One forward pass for all pairs of the groups; returns a list of logit vectors."""
    qs, ts, sizes = [], [], []
    for _, q, opts, _ in groups:
        qs += [q] * len(opts)
        ts += [t for _, t in opts]
        sizes.append(len(opts))
    x = tok(qs, ts, padding=True, truncation="only_second", max_length=args.max_len, return_tensors="pt").to(dev)
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        logits = model(**x).logits.squeeze(-1).float()
    return list(torch.split(logits, sizes))


def train_decider():
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = None
    for epoch in range(args.epochs):
        groups = [q for r in splits["train"] for q in questions_of(r, True) if q[3]]
        random.shuffle(groups)
        n = math.ceil(len(groups) / args.groups_per_batch)
        if sched is None:
            sched = get_linear_schedule_with_warmup(opt, int(0.05 * n * args.epochs), n * args.epochs)
        model.train()
        total = 0.0
        for b in range(n):
            batch = groups[b * args.groups_per_batch : (b + 1) * args.groups_per_batch]
            losses = []
            for logits, (_, _, opts, gold) in zip(group_logits(model, batch), batch):
                target = torch.tensor([1.0 if o in gold else 0.0 for o, _ in opts], device=dev)
                if target.sum() == 0:
                    continue  # no correct option among the options: nothing to learn
                target = target / target.sum()
                losses.append(-(target * F.log_softmax(logits, -1)).sum())
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss is {loss.item()} at epoch {epoch} step {b}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            total += loss.item()
            if b % 200 == 0:
                log(f"decider epoch {epoch} step {b}/{n} loss {total / (b + 1):.4f}")
        log(f"decider epoch {epoch} done, {len(groups)} questions, loss {total / n:.4f}")
    return model


@torch.no_grad()
def answer(model, groups, temps=None):
    model.eval()
    out = []
    for i in range(0, len(groups), 32):
        batch = groups[i : i + 32]
        for logits, (kind, _, opts, gold) in zip(group_logits(model, batch), batch):
            out.append((kind, [o for o, _ in opts], logits.cpu(), gold))
    return out


def fit_temperatures(answers):
    """One temperature per question kind, on the dev answers (as geoparse does)."""
    temps = {}
    for kind in {a[0] for a in answers}:
        rows = [a for a in answers if a[0] == kind and a[3]]
        best = (1e9, 1.0)
        for t in [0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0]:
            nll = 0.0
            for _, ids, logits, gold in rows:
                logp = F.log_softmax(logits / t, -1)
                nll -= torch.logsumexp(logp[[i for i, o in enumerate(ids) if o in gold]], 0).item()
            best = min(best, (nll / max(len(rows), 1), t))
        temps[kind] = best[1]
    return temps


def decider_metrics(answers, temps, records=None):
    m = {}
    for kind in sorted({a[0] for a in answers}):
        rows = [a for a in answers if a[0] == kind]
        scored = [a for a in rows if a[3]]
        top = [ids[int(torch.argmax(lg))] in gold for _, ids, lg, gold in scored]
        # Gold reachable at all (for the collection: is it on the shortlist?).
        reach = [bool(set(ids) & gold) for _, ids, _, gold in rows]
        conf = [float(F.softmax(lg / temps.get(kind, 1.0), -1).max()) for _, _, lg, _ in scored]
        m[kind] = {
            "questions": len(rows),
            "top1": round(sum(top) / max(len(top), 1), 4),
            "gold_in_options": round(sum(reach) / max(len(reach), 1), 4),
            "mean_confidence": round(sum(conf) / max(len(conf), 1), 4),
        }
    return m


# ---------------------------------------------------------------- tagger


def tag_example(r):
    enc = tok(r["request"], truncation=True, max_length=64, return_offsets_mapping=True)
    labels = []
    spans = [("PLACE", r["place"].get("span")), ("TIME", r["time"].get("span")), ("CLOUD", r["cloud"].get("span"))]
    for s, e in enc["offset_mapping"]:
        if s == e:
            labels.append(-100)
            continue
        lab = "O"
        for name, span in spans:
            if span and s >= span[0] and e <= span[1] + 0:
                lab = name
        labels.append(lab)
    out, prev = [], "O"
    for lab in labels:
        if lab == -100:
            out.append(-100)
            continue
        if lab == "O":
            out.append(0)
        else:
            out.append(TAGS.index(("I-" if prev == lab else "B-") + lab))
        prev = lab
    enc.pop("offset_mapping")
    enc["labels"] = out
    return enc


def has_all_spans(r):
    """Test records from the sheet have no cloud span; the tagger is scored on the other spans."""
    return not (r["cloud"]["value"] is not None and r["cloud"]["span"] is None)


def train_tagger():
    model = AutoModelForTokenClassification.from_pretrained(
        BASE, num_labels=len(TAGS), id2label=dict(enumerate(TAGS)), label2id={t: i for i, t in enumerate(TAGS)}
    ).to(dev)
    data = [tag_example(r) for r in splits["train"] if has_all_spans(r)]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    bs = 32
    n = math.ceil(len(data) / bs)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * n * args.tag_epochs), n * args.tag_epochs)
    for epoch in range(args.tag_epochs):
        random.shuffle(data)
        model.train()
        total = 0.0
        for b in range(n):
            batch = data[b * bs : (b + 1) * bs]
            x = tok.pad([{k: v for k, v in e.items() if k != "labels"} for e in batch], return_tensors="pt").to(dev)
            width = x["input_ids"].shape[1]
            y = torch.tensor([e["labels"] + [-100] * (width - len(e["labels"])) for e in batch], device=dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                loss = model(**x, labels=y).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"tagger loss is {loss.item()} at epoch {epoch} step {b}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            total += loss.item()
        log(f"tagger epoch {epoch} loss {total / n:.4f}")
    return model


@torch.no_grad()
def tag_spans(model, text):
    """The words of each kind: the first run of tokens with that tag (B- or I-), as text."""
    model.eval()
    enc = tok(text, truncation=True, max_length=64, return_offsets_mapping=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping")[0].tolist()
    pred = model(**enc.to(dev)).logits[0].argmax(-1).tolist()
    spans, prev = {}, None
    for (s, e), t in zip(offsets, pred):
        if s == e:
            continue
        kind = None if TAGS[t] == "O" else TAGS[t][2:]
        if kind and kind == prev and spans[kind][2]:
            spans[kind][1] = e
        elif kind and kind not in spans:
            spans[kind] = [s, e, True]
        if prev and prev != kind and prev in spans:
            spans[prev][2] = False  # the first run of this kind has ended
        prev = kind
    return {k: text[s:e] for k, (s, e, _) in spans.items()}


def norm_words(s):
    s = (s or "").lower().replace(",", " ").replace(".", " ")
    words = [w for w in s.split() if w not in {"the", "in", "on", "of", "over", "for"}]
    return " ".join(words)


def tagger_metrics(model, records):
    m = {}
    for kind, field in [("PLACE", "place"), ("TIME", "time"), ("CLOUD", "cloud")]:
        ok = n = 0
        for r in records:
            if kind == "CLOUD" and r["cloud"]["span"] is None and r["cloud"]["value"] is not None:
                continue  # no cloud words marked in the test sheet
            span = r[field].get("span")
            want = r["request"][span[0] : span[1]] if span else None
            got = tag_spans(model, r["request"]).get(kind)
            n += 1
            ok += norm_words(got) == norm_words(want) if (got and want) else (got is None and want is None)
        m[kind.lower()] = {"records": n, "exact_words": round(ok / max(n, 1), 4)}
    return m


# ---------------------------------------------------------------- export


def weight_only_int8(src, dst, min_size=4096):
    m = onnx.load(src)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    consumers = {}
    for n in g.node:
        for k, name in enumerate(n.input):
            consumers.setdefault(name, []).append((n, k))
    new_inits, new_nodes, drop = [], [], set()
    for name, init in inits.items():
        w = numpy_helper.to_array(init)
        uses = consumers.get(name, [])
        if w.dtype != np.float32 or w.ndim != 2 or w.size < min_size or not uses:
            continue
        ops = {(n.op_type, k) for n, k in uses}
        if ops == {("MatMul", 1)}:
            axis = 1  # x @ W: one scale per output column
        elif ops == {("Gather", 0)}:
            axis = 0  # embedding table: one scale per row
        else:
            continue
        scale = np.abs(w).max(axis=1 - axis) / 127.0
        scale[scale == 0] = 1.0
        shape = [1, -1] if axis == 1 else [-1, 1]
        q = np.clip(np.round(w / scale.reshape(shape)), -127, 127).astype(np.int8)
        new_inits += [
            numpy_helper.from_array(q, f"{name}_q"),
            numpy_helper.from_array(scale.astype(np.float32), f"{name}_scale"),
            numpy_helper.from_array(np.zeros_like(scale, dtype=np.int8), f"{name}_zp"),
        ]
        new_nodes.append(helper.make_node("DequantizeLinear", [f"{name}_q", f"{name}_scale", f"{name}_zp"], [name], axis=axis, name=f"{name}_dq"))
        drop.add(name)
    kept = [i for i in g.initializer if i.name not in drop]
    del g.initializer[:]
    g.initializer.extend(kept + new_inits)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(new_nodes + nodes)
    onnx.save(m, dst)
    return len(drop)



class Squeeze(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids, attention_mask=attention_mask).logits


def export(model, name, output_axes):
    """fp32 ONNX, and weight-only int8 as model_quantized.onnx (training/tools/weight_only.py). Dynamic int8
    (quantize_dynamic) also turns the activations into int8; it broke v1 (decider top answer agreed with fp32 on
    47% of questions; weight-only int8: 95%)."""

    d = OUT / name
    (d / "onnx").mkdir(parents=True, exist_ok=True)
    model = model.float().cpu().eval()
    model.save_pretrained(d)  # config.json (and weights, for later fine-tuning)
    tok.save_pretrained(d)
    x = tok(["a | request: b"], ["c"], return_tensors="pt")
    torch.onnx.export(
        Squeeze(model),
        (x["input_ids"], x["attention_mask"]),
        str(d / "onnx/model.onnx"),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "logits": output_axes},
        opset_version=17,
        dynamo=False,
    )
    weight_only_int8(str(d / "onnx/model.onnx"), str(d / "onnx/model_quantized.onnx"))
    # Parity: int8 ONNX against the PyTorch model.
    import onnxruntime as ort

    sess = ort.InferenceSession(str(d / "onnx/model_quantized.onnx"))
    y = sess.run(None, {"input_ids": x["input_ids"].numpy(), "attention_mask": x["attention_mask"].numpy()})[0]
    with torch.no_grad():
        ref = model(**x).logits.numpy()
    log(f"{name}: onnx int8 max abs diff {float(np.abs(y - ref).max()):.4f}, size {(d / 'onnx/model_quantized.onnx').stat().st_size / 1e6:.1f} MB")
    return d


# ---------------------------------------------------------------- run

report = {"run": args.run, "base": BASE, "retriever": RETRIEVER, "args": vars(args), "data": {k: len(v) for k, v in splits.items()}}

decider = train_decider()
dev_answers = answer(decider, [q for r in splits["dev"] for q in questions_of(r, False)])
temps = fit_temperatures(dev_answers)
report["temperatures"] = temps
report["decider"] = {"dev": decider_metrics(dev_answers, temps)}
for s in ["test-catalog", "test-claude"]:
    report["decider"][s] = decider_metrics(answer(decider, [q for r in splits[s] for q in questions_of(r, False)]), temps)
log("decider", json.dumps(report["decider"], indent=1))

tagger = train_tagger()
report["tagger"] = {s: tagger_metrics(tagger, splits[s]) for s in ["dev", "test-catalog", "test-claude"]}
log("tagger", json.dumps(report["tagger"], indent=1))

ddir = export(decider, "decider", {0: "batch"})
json.dump({"temperatures": temps, "shortlist": SHORTLIST, "max_length": args.max_len, "retriever": RETRIEVER}, open(ddir / "decider.json", "w"), indent=1)
tdir = export(tagger, "tagger", {0: "batch", 1: "seq"})
report["minutes"] = round((time.time() - t0) / 60, 1)
json.dump(report, open(OUT / f"report-{args.run}.json", "w"), indent=1)
log("report", OUT / f"report-{args.run}.json")

if not args.no_upload:
    api = HfApi()
    for name in ["decider", "tagger"]:
        api.upload_folder(repo_id=MODEL_REPO, folder_path=str(OUT / name), path_in_repo=f"{name}/{args.run}",
                          commit_message=f"{name} {args.run}")
    api.upload_file(path_or_fileobj=str(OUT / f"report-{args.run}.json"), path_in_repo=f"reports/report-{args.run}.json",
                    repo_id=MODEL_REPO, commit_message=f"report {args.run}")
    log("uploaded to", MODEL_REPO)
