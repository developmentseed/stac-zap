"""Compares int8 variants of an ONNX model with its fp32 file: how often the top answer agrees, and the size.

Usage: uv run --no-project --with onnx --with onnxruntime --with transformers --with numpy \
         python training/tools/quant_check.py <model dir with onnx/model.onnx> decider|tagger
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoTokenizer

d, kind = Path(sys.argv[1]), sys.argv[2]
tok = AutoTokenizer.from_pretrained(d)
recs = [json.loads(l) for l in open(Path(__file__).parents[1] / "data/test-claude.jsonl")]
snap = json.load(open(Path(__file__).parents[1] / "data/stac-full.json"))
cats = [c for c in snap["catalogs"] if c.get("curated")]


def inputs():
    for r in recs[:120]:
        if kind == "tagger":
            yield tok([r["request"]], return_tensors="np")
        else:
            q = f"Which STAC catalog? | request: {r['request']} | current: not set"
            texts = [f"{c['id']}: {' '.join(c['description'].split())[:300]}" for c in cats]
            yield tok([q] * len(texts), texts, padding=True, truncation="only_second", max_length=96, return_tensors="np")


def run(path):
    s = ort.InferenceSession(str(path))
    outs = []
    for x in inputs():
        y = s.run(None, {"input_ids": x["input_ids"].astype(np.int64), "attention_mask": x["attention_mask"].astype(np.int64)})[0]
        outs.append(y)
    return outs


ref = run(d / "onnx/model.onnx")
variants = {
    "matmul+gather, int8, per tensor (v1)": dict(weight_type=QuantType.QInt8),
    "matmul only, int8, per channel": dict(weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["MatMul"]),
    "matmul+gather, int8, per channel": dict(weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["MatMul", "Gather"]),
    "matmul+gather, uint8, per channel": dict(weight_type=QuantType.QUInt8, per_channel=True, op_types_to_quantize=["MatMul", "Gather"]),
    "matmul+gather, uint8, per tensor": dict(weight_type=QuantType.QUInt8, op_types_to_quantize=["MatMul", "Gather"]),
}
sys.path.insert(0, str(Path(__file__).parent))
from weight_only import weight_only_int8

variants = {"weight-only int8, per channel": None, **{k: v for k, v in variants.items() if "per channel" in k and "matmul+gather, int8" in k}}
for name, kw in variants.items():
    out = Path("/tmp") / f"q_{kind}.onnx"
    if kw is None:
        weight_only_int8(str(d / "onnx/model.onnx"), str(out))
    else:
        quantize_dynamic(str(d / "onnx/model.onnx"), str(out), **kw)
    got = run(out)
    if kind == "tagger":
        agree = np.mean([np.mean(a.argmax(-1) == b.argmax(-1)) for a, b in zip(ref, got)])
    else:
        agree = np.mean([a[:, 0].argmax() == b[:, 0].argmax() for a, b in zip(ref, got)])
    diff = max(float(np.abs(a - b).max()) for a, b in zip(ref, got))
    print(f"{name:40s} agree {agree:.3f}  max diff {diff:6.2f}  size {out.stat().st_size / 1e6:5.1f} MB", flush=True)
