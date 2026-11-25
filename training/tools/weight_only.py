"""Weight-only int8: stores the large weight matrices as int8 (symmetric, one scale per output channel) with a
DequantizeLinear node, so the file is about a quarter of fp32 while the model computes in fp32. ONNX Runtime folds the
DequantizeLinear nodes into fp32 weights when it loads the model.

Usage: python weight_only.py <in.onnx> <out.onnx>
"""
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper


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


if __name__ == "__main__":
    print(weight_only_int8(sys.argv[1], sys.argv[2]), "weights stored as int8")
