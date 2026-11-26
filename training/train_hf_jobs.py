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
"""Trains the local model of stac-zap and exports it as ONNX (weight-only int8).

One Granite 30M encoder (ibm-granite/granite-embedding-30m-english) with three uses, so the page downloads one
model (v6 and later; v1 to v5 were three models, see the git history):
- retriever: the normalized CLS vector of a text. It shortlists the collections of all known catalogs.
- decider: a head on the CLS vector of a (question, option) pair. It picks a collection of the shortlist (the
  catalog follows from it), or `keep`, or a place among the geocoder results. A softmax over the options of one
  question, with a calibrated temperature, gives the probabilities.
- tagger: a head on each token: BIO tags for the place, time and cloud cover words.
Stage 1 trains the retriever (contrastive, with hard negatives from R2, ibm-granite/granite-embedding-small-english-r2);
stage 2 trains the three together.

Data: the dataset developmentseed/stac-zap-data (generated/<version>, snapshot, places, testset). Output: the model repo
developmentseed/stac-zap-models, folder shared/<run>/ (model/ in the layout transformers.js reads, stac-zap.json),
and reports/report-<run>.json.

Run on HF Jobs:
  hf jobs uv run --flavor l4x1 --timeout 3h --secrets HF_TOKEN --detach training/train_hf_jobs.py --run v6
Smoke test on a laptop CPU:
  uv run training/train_hf_jobs.py --run smoke --max-records 60 --epochs 1 --no-upload --device cpu
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
from onnx import TensorProto, helper, numpy_helper
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

DATA_REPO = "developmentseed/stac-zap-data"
MODEL_REPO = "developmentseed/stac-zap-models"
BASE = "ibm-granite/granite-embedding-30m-english"
RETRIEVER = "ibm-granite/granite-embedding-small-english-r2"
SHORTLIST = 16
PREFERRED = 0.6
RETRIEVER_WEIGHT = 0.5
KEEP_AUGMENT = 0.4
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
p.add_argument("--retriever-epochs", type=int, default=2)
p.add_argument("--data", default="v3", help="folder of generated/ in the dataset")
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
    # No population: the geocoder of the app (Photon) does not give it.
    return f"{pl['name']}{', ' + where if where else ''} ({kind})"


def query(question, r, current=None):
    return f"{question} | request: {r['request']} | current: {current or 'not set'}"


# ---------------------------------------------------------------- shared model (v6)
#
# One Granite 30M encoder with three uses, so the page downloads one model:
# - retriever: the normalized CLS vector of a single text,
# - decider: a head on the CLS vector of a (question, option) pair,
# - tagger: a head on each token.
# Stage 1 trains the retriever; stage 2 trains the decider, the tagger and the retriever together, so the encoder
# keeps all three abilities.


class Out:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class Shared(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(BASE, add_pooling_layer=False, dtype=torch.float32)
        h = self.enc.config.hidden_size
        self.dec = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(h, h), torch.nn.Tanh(), torch.nn.Dropout(0.1), torch.nn.Linear(h, 1))
        self.tag = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(h, len(TAGS)))

    def forward(self, input_ids, attention_mask, **_):
        hidden = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return Out(last_hidden_state=hidden, logits=self.dec(hidden[:, 0]), tag_logits=self.tag(hidden))


class TagView(torch.nn.Module):
    """The shared model as a token classifier, for the tagger metrics."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, **x):
        return Out(logits=self.m(**x).tag_logits)


# ---------------------------------------------------------------- retriever (catalog follows collection)
#
# The decider picks a collection among the shortlist of all known catalogs; the catalog follows from it. The
# shortlist comes from a retriever: Granite 30M trained as a bi-encoder (contrastive, in-batch and hard negatives
# from the R2 model). The page embeds each collection once (or loads the vectors with the site), then only the
# request per zap.


def pair_key(cat, cid):
    return f"{cat}/{cid}"


def doc_text(cat, cid):
    """A collection as the retriever and the decider read it: catalog, title, id, start of the description."""
    return f"{cat.replace('-', ' ')} | {collection_text(cat, cid)}"


@torch.no_grad()
def embed(model, tok_, texts, bs=128, max_length=128):
    model.eval()
    out = []
    for i in range(0, len(texts), bs):
        x = tok_(texts[i : i + bs], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            h = model(**x).last_hidden_state[:, 0]
        out.append(F.normalize(h.float(), dim=-1).cpu())
    return torch.cat(out) if out else torch.zeros(0, 384)


doc_keys = [(cat, cid) for cat, cols in collections.items() for cid in cols]
doc_index = {pair_key(c, i): n for n, (c, i) in enumerate(doc_keys)}
doc_texts = [doc_text(c, i) for c, i in doc_keys]
curated_set = set(curated)
all_records = [r for sp in splits.values() for r in sp]


def pairs_of(r):
    """Every correct catalog/collection pair of a record."""
    pairs = set(r["collection"].get("pairs") or [])
    if not pairs and r["collection"].get("accepted"):  # test sheet: catalogs x collections that exist
        pairs = {pair_key(c, i) for c in r["catalog"]["accepted"] for i in r["collection"]["accepted"] if pair_key(c, i) in doc_index}
    if not pairs and r["collection"].get("target") and r["catalog"].get("target"):
        pairs = {pair_key(r["catalog"]["target"], r["collection"]["target"])}
    return {p for p in pairs if p in doc_index}


log("R2: embedding collections and requests (hard negatives, and the comparison)")
rtok = AutoTokenizer.from_pretrained(RETRIEVER)
r2 = AutoModel.from_pretrained(RETRIEVER).to(dev)
r2_docs = embed(r2, rtok, doc_texts, max_length=256)
r2_reqs = embed(r2, rtok, [r["request"] for r in all_records], max_length=256)
r2_req = {id(r): r2_reqs[n] for n, r in enumerate(all_records)}
del r2
torch.cuda.empty_cache() if dev.type == "cuda" else None

tok = AutoTokenizer.from_pretrained(BASE)


def pool_ids(train, curated_only):
    return [n for n, (c, _) in enumerate(doc_keys) if (not train or c not in HELD_OUT) and (not curated_only or c in curated_set)]


POOL_ALL_TRAIN, POOL_CUR_TRAIN = pool_ids(True, False), pool_ids(True, True)
POOL_ALL, POOL_CUR = pool_ids(False, False), pool_ids(False, True)


def top_k(qvec, docs, pool, k):
    sims = docs[pool] @ qvec
    return [pool[j] for j in torch.topk(sims, min(k, len(pool))).indices.tolist()]


RETRIEVER_DATA = None


def retriever_loss(model, batch, temp=0.05):
    """InfoNCE with in-batch negatives and one hard negative (a top R2 result that is not correct) per request."""
    pos, neg = [], []
    for r in batch:
        ok = pairs_of(r)
        pos.append(doc_texts[doc_index[random.choice(sorted(ok))]])
        cands = [d for d in top_k(r2_req[id(r)], r2_docs, POOL_ALL_TRAIN, 12) if pair_key(*doc_keys[d]) not in ok]
        neg.append(doc_texts[random.choice(cands)] if cands else doc_texts[random.randrange(len(doc_texts))])
    q = tok([r["request"] for r in batch], padding=True, truncation=True, max_length=64, return_tensors="pt").to(dev)
    d = tok(pos + neg, padding=True, truncation=True, max_length=128, return_tensors="pt").to(dev)
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        qv = F.normalize(model(**q).last_hidden_state[:, 0].float(), dim=-1)
        dv = F.normalize(model(**d).last_hidden_state[:, 0].float(), dim=-1)
    logits = qv @ dv.T / temp
    # A positive of another request in the batch that is also correct is not a negative.
    mask = torch.zeros_like(logits, dtype=torch.bool)
    texts = pos + neg
    for a, r in enumerate(batch):
        ok_texts = {doc_texts[doc_index[p]] for p in pairs_of(r)}
        for c, text in enumerate(texts):
            if c != a and text in ok_texts:
                mask[a, c] = True
    return F.cross_entropy(logits.masked_fill(mask, -1e4), torch.arange(len(batch), device=dev))


def train_retriever(model, epochs=2, bs=96, lr=3e-5):
    """Stage 1: the encoder as a retriever."""
    data = [r for r in splits["train"] if pairs_of(r)]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n = math.ceil(len(data) / bs)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * n * epochs), n * epochs)
    for epoch in range(epochs):
        random.shuffle(data)
        model.train()
        total = 0.0
        for b in range(n):
            loss = retriever_loss(model, data[b * bs : (b + 1) * bs])
            if not torch.isfinite(loss):
                raise RuntimeError(f"retriever loss is {loss.item()}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            total += loss.item()
            if b % 100 == 0:
                log(f"retriever epoch {epoch} step {b}/{n} loss {total / (b + 1):.4f}")
        log(f"retriever epoch {epoch} loss {total / n:.4f}")
    return model


def recall(docs, reqs, records, pool, ks=(1, 5, 16, 32)):
    hits = {k: 0 for k in ks}
    n = 0
    for r in records:
        ok = pairs_of(r)
        if not ok:
            continue
        n += 1
        top = top_k(reqs[id(r)], docs, pool, max(ks))
        for k in ks:
            hits[k] += any(pair_key(*doc_keys[d]) in ok for d in top[:k])
    return {f"@{k}": round(hits[k] / max(n, 1), 4) for k in ks} | {"n": n}


# ---------------------------------------------------------------- questions


def questions_of(r, train):
    """The decider questions of a record: (kind, query, [(option id, text)], set of correct ids)."""
    qs = []
    cur = dict(r.get("current") or {})
    ok = pairs_of(r)
    # A full new request while another collection is open: keep is an option, but not the answer. (v3 saw keep only
    # in follow-ups, so after the first zap it kept the open collection for every request.)
    if train and not cur.get("collection") and r["collection"]["target"] and random.random() < KEEP_AUGMENT:
        other = doc_keys[random.randrange(len(doc_keys))]
        if pair_key(*other) not in ok:
            cur = {"catalog": other[0], "collection": other[1], "place": None}
    # Collection, on the shortlist of all known catalogs (curated only half of the time in training, as the app
    # does by default; at test time, curated only when every correct pair is curated).
    if ok or cur.get("collection"):
        cur_only = random.random() < 0.5 if train else all(p.split("/")[0] in curated_set for p in ok) if ok else False
        pool = (POOL_CUR_TRAIN if cur_only else POOL_ALL_TRAIN) if train else (POOL_CUR if cur_only else POOL_ALL)
        ids = top_k(req_vec[id(r)], doc_vecs, pool, SHORTLIST)
        keys = [pair_key(*doc_keys[d]) for d in ids]
        if train and ok and not ok & set(keys):
            cand = [p for p in ok if doc_index[p] in set(pool)]
            if cand:
                keys = keys[:-1] + [random.choice(sorted(cand))]
        opts = [(k, doc_texts[doc_index[k]]) for k in keys]
        if cur.get("collection"):
            opts.append((KEEP, f"keep: keep the current collection ({cur['catalog']}/{cur['collection']})"))
        if r["collection"]["target"]:
            gold = ok & set(keys)
        else:
            gold = {KEEP} if cur.get("collection") else set()
        current = f"{cur['catalog']}/{cur['collection']}" if cur.get("collection") else None
        # Soft target: the preferred answer gets PREFERRED of the weight, the other correct ones share the rest.
        weights = None
        pref = r["collection"].get("preferred")
        if pref in gold and len(gold) > 1:
            weights = {k: (PREFERRED if k == pref else (1 - PREFERRED) / (len(gold) - 1)) for k in gold}
        if gold or not train:
            qs.append(("collection", query("Which collection?", r, current), opts, gold, weights))

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
        # The order of the geocoder: most important first. Population stands in for it here; in training, some
        # neighbours swap, as a geocoder's order is not always by size.
        cands.sort(key=lambda c: -(places[c].get("population") or 0))
        if train:
            for i in range(len(cands) - 1):
                if random.random() < 0.15:
                    cands[i], cands[i + 1] = cands[i + 1], cands[i]
        opts = [(c, f"[rank {i + 1}] {place_text(c)}") for i, c in enumerate(cands)] + [(NONE, place_text(NONE))]
        qs.append(("place", query(f"Which place is '{pl['words']}'?", r), opts, gold, None))
    return qs


# ---------------------------------------------------------------- decider


def encode_pairs(q, texts):
    return tok([q] * len(texts), texts, padding=True, truncation="longest_first", max_length=args.max_len, return_tensors="pt")


def group_logits(model, groups):
    """One forward pass for all pairs of the groups; returns a list of logit vectors."""
    qs, ts, sizes = [], [], []
    for _, q, opts, *_ in groups:
        qs += [q] * len(opts)
        ts += [t for _, t in opts]
        sizes.append(len(opts))
    x = tok(qs, ts, padding=True, truncation="longest_first", max_length=args.max_len, return_tensors="pt").to(dev)
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        logits = model(**x).logits.squeeze(-1).float()
    return list(torch.split(logits, sizes))


def decider_loss(model, batch):
    losses = []
    for logits, (_, _, opts, gold, weights) in zip(group_logits(model, batch), batch):
        target = torch.tensor([(weights or {}).get(o, 1.0) if o in gold else 0.0 for o, _ in opts], device=dev)
        if target.sum() == 0:
            continue  # no correct option among the options: nothing to learn
        target = target / target.sum()
        losses.append(-(target * F.log_softmax(logits, -1)).sum())
    return torch.stack(losses).mean() if losses else None


def tagger_loss(model, batch):
    x = tok.pad([{k: v for k, v in e.items() if k != "labels"} for e in batch], return_tensors="pt").to(dev)
    width = x["input_ids"].shape[1]
    y = torch.tensor([e["labels"] + [-100] * (width - len(e["labels"])) for e in batch], device=dev)
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        logits = model(**x).tag_logits.float()
    return F.cross_entropy(logits.reshape(-1, len(TAGS)), y.reshape(-1), ignore_index=-100)


def train_joint(model):
    """Stage 2: each step adds the decider loss, a tagger batch and a retriever batch."""
    tag_data = [tag_example(r) for r in splits["train"] if has_all_spans(r)]
    ret_data = [r for r in splits["train"] if pairs_of(r)]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = None
    ti = ri = 0
    for epoch in range(args.epochs):
        groups = [q for r in splits["train"] for q in questions_of(r, True) if q[3]]
        log(f"joint epoch {epoch}: {len(groups)} questions")
        random.shuffle(groups)
        n = math.ceil(len(groups) / args.groups_per_batch)
        if sched is None:
            sched = get_linear_schedule_with_warmup(opt, int(0.05 * n * args.epochs), n * args.epochs)
        model.train()
        totals = {"decider": 0.0, "tagger": 0.0, "retriever": 0.0}
        for b in range(n):
            dl = decider_loss(model, groups[b * args.groups_per_batch : (b + 1) * args.groups_per_batch])
            if ti + 32 > len(tag_data):
                random.shuffle(tag_data)
                ti = 0
            tl = tagger_loss(model, tag_data[ti : ti + 32])
            ti += 32
            if ri + 48 > len(ret_data):
                random.shuffle(ret_data)
                ri = 0
            rl = retriever_loss(model, ret_data[ri : ri + 48])
            ri += 48
            loss = tl + RETRIEVER_WEIGHT * rl + (dl if dl is not None else 0)
            if not torch.isfinite(loss):
                raise RuntimeError(f"joint loss is {loss.item()} at epoch {epoch} step {b}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            totals["decider"] += dl.item() if dl is not None else 0
            totals["tagger"] += tl.item()
            totals["retriever"] += rl.item()
            if b % 200 == 0 or b < 3:
                log(f"joint epoch {epoch} step {b}/{n} " + " ".join(f"{k} {v / (b + 1):.4f}" for k, v in totals.items()))
        log(f"joint epoch {epoch} done, {len(groups)} questions, " + " ".join(f"{k} {v / n:.4f}" for k, v in totals.items()))
    return model


@torch.no_grad()
def answer(model, groups, temps=None):
    model.eval()
    out = []
    for i in range(0, len(groups), 32):
        batch = groups[i : i + 32]
        for logits, (kind, _, opts, gold, *_) in zip(group_logits(model, batch), batch):
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



def quantize_blocks(w, axis_len, block, bits):
    """w: [..., n] with the blocked axis last. Returns (q as uint8 0..2^bits-1, scale [..., n_blocks, 1], shape)."""
    block = block or axis_len
    assert axis_len % block == 0, f"block {block} does not divide {axis_len}"
    shape = w.shape[:-1] + (axis_len // block, block)
    wb = w.reshape(shape)
    qmax = 2 ** (bits - 1) - 1  # symmetric: -qmax..qmax, stored with an offset of qmax + 1
    scale = np.abs(wb).max(axis=-1, keepdims=True) / qmax
    scale[scale == 0] = 1.0
    q = np.clip(np.round(wb / scale), -qmax, qmax).astype(np.int16) + (qmax + 1)
    return q.astype(np.uint8), scale.astype(np.float32), shape


def dequant_nodes(name, q, scale, block_shape, out_shape, bits, perm=None, fp16_scale=False):
    """Initializers and nodes that rebuild the fp32 tensor `name` from q (uint8) and scale."""
    inits, nodes = [], []
    n = f"{name}__"
    offset = float(2 ** (bits - 1))
    flat = q.reshape(-1)
    if bits == 4:
        packed = (flat[0::2] << 4) | flat[1::2]
        inits.append(numpy_helper.from_array(packed.astype(np.uint8), n + "packed"))
        inits.append(numpy_helper.from_array(np.array([4], dtype=np.uint8), n + "four"))
        inits.append(numpy_helper.from_array(np.array([15], dtype=np.uint8), n + "mask"))
        inits.append(numpy_helper.from_array(np.array([-1, 1], dtype=np.int64), n + "col"))
        nodes += [
            helper.make_node("BitShift", [n + "packed", n + "four"], [n + "hi"], direction="RIGHT"),
            helper.make_node("BitwiseAnd", [n + "packed", n + "mask"], [n + "lo"]),
            helper.make_node("Reshape", [n + "hi", n + "col"], [n + "hi2"]),
            helper.make_node("Reshape", [n + "lo", n + "col"], [n + "lo2"]),
            helper.make_node("Concat", [n + "hi2", n + "lo2"], [n + "pairs"], axis=1),
        ]
        src = n + "pairs"
    else:
        inits.append(numpy_helper.from_array(flat, n + "q"))
        src = n + "q"
    inits += [
        numpy_helper.from_array(np.array(block_shape, dtype=np.int64), n + "bshape"),
        numpy_helper.from_array(np.array(out_shape, dtype=np.int64), n + "oshape"),
        numpy_helper.from_array(np.array(offset, dtype=np.float32), n + "offset"),
        numpy_helper.from_array(scale.astype(np.float16) if fp16_scale else scale, n + "scale16" if fp16_scale else n + "scale"),
    ]
    if fp16_scale:
        nodes.append(helper.make_node("Cast", [n + "scale16"], [n + "scale"], to=TensorProto.FLOAT))
    last = n + "w"
    nodes += [
        helper.make_node("Reshape", [src, n + "bshape"], [n + "qb"]),
        helper.make_node("Cast", [n + "qb"], [n + "f"], to=TensorProto.FLOAT),
        helper.make_node("Sub", [n + "f", n + "offset"], [n + "c"]),
        helper.make_node("Mul", [n + "c", n + "scale"], [n + "s"]),
        helper.make_node("Reshape", [n + "s", n + "oshape"], [last if perm else name]),
    ]
    if perm:
        nodes.append(helper.make_node("Transpose", [last], [name], perm=perm))
    return inits, nodes


def quantize_int4(src, dst, emb_bits=4, emb_block=64, mm_bits=8, mm_block=0, min_size=4096, fp16_scale=True):
    m = onnx.load(src)
    g = m.graph
    consumers = {}
    for node in g.node:
        for k, name in enumerate(node.input):
            consumers.setdefault(name, []).append((node.op_type, k))
    new_inits, new_nodes, drop = [], [], set()
    for init in g.initializer:
        w = numpy_helper.to_array(init)
        uses = set(consumers.get(init.name, []))
        if w.dtype != np.float32 or w.ndim != 2 or w.size < min_size:
            continue
        if uses == {("Gather", 0)}:
            # [vocab, hidden], blocks along hidden.
            q, scale, bshape = quantize_blocks(w, w.shape[1], emb_block, emb_bits)
            inits, nodes = dequant_nodes(init.name, q, scale, bshape, w.shape, emb_bits, fp16_scale=fp16_scale)
        elif uses == {("MatMul", 1)}:
            # [in, out], blocks along in: quantize the transpose ([out, in]), then transpose back.
            wt = np.ascontiguousarray(w.T)
            q, scale, bshape = quantize_blocks(wt, wt.shape[1], mm_block, mm_bits)
            inits, nodes = dequant_nodes(init.name, q, scale, bshape, wt.shape, mm_bits, perm=[1, 0], fp16_scale=fp16_scale)
        else:
            continue
        new_inits += inits
        new_nodes += nodes
        drop.add(init.name)
    kept = [i for i in g.initializer if i.name not in drop]
    del g.initializer[:]
    g.initializer.extend(kept + new_inits)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(new_nodes + nodes)
    # BitwiseAnd needs opset 18.
    for op in m.opset_import:
        if op.domain in ("", "ai.onnx") and op.version < 18:
            op.version = 18
    onnx.save(m, dst)
    return len(drop)



class Heads(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        o = self.m(input_ids=input_ids, attention_mask=attention_mask)
        return o.last_hidden_state, o.logits.squeeze(-1), o.tag_logits


def export_shared(model, temps):
    """One ONNX file with three outputs: last_hidden_state (retriever: CLS, normalized), decider_logits (one per
    input), tag_logits. The layout transformers.js reads: <run>/model/{config.json, tokenizer, onnx/}, and the
    settings in <run>/stac-zap.json."""
    d = OUT / "shared"
    (d / "model/onnx").mkdir(parents=True, exist_ok=True)
    model = model.float().cpu().eval()
    model.enc.config.save_pretrained(d / "model")
    tok.save_pretrained(d / "model")
    torch.save(model.state_dict(), d / "shared.pt")
    x = tok(["Which collection? | request: Sentinel-2 over Lisbon | current: not set"], ["planetary computer | Sentinel-2"], return_tensors="pt")
    torch.onnx.export(
        Heads(model), (x["input_ids"], x["attention_mask"]), str(d / "model/onnx/model.onnx"),
        input_names=["input_ids", "attention_mask"], output_names=["last_hidden_state", "decider_logits", "tag_logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                      "last_hidden_state": {0: "batch", 1: "seq"}, "decider_logits": {0: "batch"}, "tag_logits": {0: "batch", 1: "seq"}},
        opset_version=17, dynamo=False,
    )
    weight_only_int8(str(d / "model/onnx/model.onnx"), str(d / "model/onnx/model_quantized.onnx"))
    # The file the page loads: int4 embedding table (blocks of 32, fp16 scales), int8 weights, and its gzip copy.
    # int4 in the weights broke the model (decider agreement with fp32 13-29%); see training/tools/weight_int4.py.
    quantize_int4(str(d / "model/onnx/model.onnx"), str(d / "model/onnx/model_q4.onnx"), 4, 32, 8, 0)
    import gzip
    import shutil
    with open(d / "model/onnx/model_q4.onnx", "rb") as fi, gzip.open(d / "model/onnx/model_q4.onnx.gz", "wb", compresslevel=9) as fo:
        shutil.copyfileobj(fi, fo)
    import onnxruntime as ort

    sess = ort.InferenceSession(str(d / "model/onnx/model_quantized.onnx"))
    got = sess.run(None, {"input_ids": x["input_ids"].numpy(), "attention_mask": x["attention_mask"].numpy()})
    with torch.no_grad():
        ref = Heads(model)(x["input_ids"], x["attention_mask"])
    diffs = [float(np.abs(g - r.numpy()).max()) for g, r in zip(got, ref)]
    log(f"shared: onnx int8 max abs diff (hidden, decider, tags) {diffs}, size {(d / 'model/onnx/model_quantized.onnx').stat().st_size / 1e6:.1f} MB")
    json.dump({
        "temperatures": temps, "shortlist": SHORTLIST, "decider_max_length": args.max_len, "query_max_length": 64,
        "doc_max_length": 128, "tags": TAGS, "pooling": "cls", "normalize": True,
        "design": "one encoder, three heads; catalog follows collection",
        "doc_text": "catalog with spaces | title (id): description",
    }, open(d / "stac-zap.json", "w"), indent=1)
    return d


# ---------------------------------------------------------------- run

report = {"run": args.run, "base": BASE, "retriever": RETRIEVER, "args": vars(args), "data": {k: len(v) for k, v in splits.items()}}

model = Shared().to(dev)
train_retriever(model, epochs=args.retriever_epochs)
# The decider learns on the shortlists of the stage 1 retriever; the metrics below use the final encoder.
log("embedding collections with the stage 1 retriever")
doc_vecs = embed(model, tok, doc_texts)
req_vecs_all = embed(model, tok, [r["request"] for r in all_records], max_length=64)
req_vec = {id(r): req_vecs_all[n] for n, r in enumerate(all_records)}

log("stage 2: decider, tagger and retriever together")
train_joint(model)
log("embedding collections with the final encoder")

doc_vecs = embed(model, tok, doc_texts)
req_vecs_all = embed(model, tok, [r["request"] for r in all_records], max_length=64)
req_vec = {id(r): req_vecs_all[n] for n, r in enumerate(all_records)}
report["retriever"] = {}
for sname in ["dev", "test-catalog", "test-claude"]:
    for pname, pool in [("all", POOL_ALL), ("curated", POOL_CUR)]:
        recs = splits[sname] if pname == "all" else [r for r in splits[sname] if pairs_of(r) and all(p.split("/")[0] in curated_set for p in pairs_of(r))]
        report["retriever"][f"{sname}/{pname}"] = {
            "shared-30m": recall(doc_vecs, req_vec, recs, pool),
            "r2-untrained": recall(r2_docs, r2_req, recs, pool),
        }
log("retriever", json.dumps(report["retriever"], indent=1))

dev_answers = answer(model, [q for r in splits["dev"] for q in questions_of(r, False)])
temps = fit_temperatures(dev_answers)
report["temperatures"] = temps
report["decider"] = {"dev": decider_metrics(dev_answers, temps)}
for s_ in ["test-catalog", "test-claude"]:
    report["decider"][s_] = decider_metrics(answer(model, [q for r in splits[s_] for q in questions_of(r, False)]), temps)
log("decider", json.dumps(report["decider"], indent=1))

report["tagger"] = {s_: tagger_metrics(TagView(model), splits[s_]) for s_ in ["dev", "test-catalog", "test-claude"]}
log("tagger", json.dumps(report["tagger"], indent=1))

sdir = export_shared(model, temps)
report["minutes"] = round((time.time() - t0) / 60, 1)
json.dump(report, open(OUT / f"report-{args.run}.json", "w"), indent=1)
log("report", OUT / f"report-{args.run}.json")

if not args.no_upload:
    api = HfApi()
    api.upload_folder(repo_id=MODEL_REPO, folder_path=str(OUT / "shared"), path_in_repo=f"shared/{args.run}",
                      commit_message=f"shared model {args.run}")
    api.upload_file(path_or_fileobj=str(OUT / f"report-{args.run}.json"), path_in_repo=f"reports/report-{args.run}.json",
                    repo_id=MODEL_REPO, commit_message=f"report {args.run}")
    log("uploaded to", MODEL_REPO)
