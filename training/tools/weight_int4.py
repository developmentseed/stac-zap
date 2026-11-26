"""Weight-only quantization with 4-bit or 8-bit blocks, for a smaller download.

Each large weight matrix is stored as integers with one scale per block of values, and a few standard ONNX nodes turn
it back into fp32: unpack (for 4 bits: two values per byte, BitShift and BitwiseAnd), Cast, Sub, Mul, Reshape. All of
them have constant inputs, so ONNX Runtime folds them into an fp32 weight when it loads the model: the model computes
in fp32, the file holds 4 or 8 bits per weight. No special operator is needed, so it runs on any ONNX Runtime build,
including the WebAssembly one.

- The embedding table ([vocab, hidden], read by Gather): blocks along the hidden axis.
- MatMul weights ([in, out]): blocks along the input axis, one scale per block and output column.
Scales are stored as fp16 (a Cast makes them fp32).

Usage: python weight_int4.py <fp32 model.onnx> <out.onnx> [--emb-bits 4] [--emb-block 64] [--mm-bits 8] [--mm-block 0]
A block of 0 is one block for the whole axis (per row or per column).
"""

import argparse

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


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


def quantize(src, dst, emb_bits=4, emb_block=64, mm_bits=8, mm_block=0, min_size=4096, fp16_scale=True):
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


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--emb-bits", type=int, default=4, choices=[4, 8])
    p.add_argument("--emb-block", type=int, default=64)
    p.add_argument("--mm-bits", type=int, default=8, choices=[4, 8])
    p.add_argument("--mm-block", type=int, default=0)
    a = p.parse_args()
    print(quantize(a.src, a.dst, a.emb_bits, a.emb_block, a.mm_bits, a.mm_block), "weights quantized")
