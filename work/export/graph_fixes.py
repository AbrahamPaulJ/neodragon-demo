"""Reusable ONNX graph passes for the QNN port.

retarget_extreme_constants() is the general form of trap #2's fix. The
dangerous mask value is never written in Python -- it comes from
`torch.finfo(dtype).min` inside transformers (T5Stack.forward) and from `-inf`
inside the SDPA lowering (pyramid MMDiT). Both land in the graph as Constant
nodes or initializers, so one graph-level pass covers every module in the
pipeline.
"""

import numpy as np
import onnx
from onnx import numpy_helper

EXTREME = 1e30


def _fix_array(arr, fill):
    if arr.dtype.kind != "f" or arr.size == 0:
        return None
    bad = ~np.isfinite(arr) | (np.abs(arr) > EXTREME)
    if not bad.any():
        return None
    out = arr.copy()
    out[bad & (arr < 0)] = fill
    out[bad & (arr > 0)] = -fill
    # non-finite entries have no sign under the comparisons above; -inf/-nan
    # are handled by the < 0 branch, +inf by the > 0 branch, nan by neither.
    nan = np.isnan(arr)
    if nan.any():
        out[nan] = fill
    return out


def retarget_extreme_constants(model, fill=-100.0, verbose=True):
    """Replace non-finite / >1e30 float constants with a finite `fill`."""
    n = 0
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        new = _fix_array(arr, fill)
        if new is not None:
            init.CopyFrom(numpy_helper.from_array(new, init.name))
            n += 1
            if verbose:
                print(f"    initializer {init.name}: {arr.min()} -> {new.min()}")
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.name == "value" and attr.t.ByteSize():
                arr = numpy_helper.to_array(attr.t)
                new = _fix_array(arr, fill)
                if new is not None:
                    attr.t.CopyFrom(numpy_helper.from_array(new, attr.t.name))
                    n += 1
                    if verbose:
                        print(f"    {node.op_type} {node.name}: {arr.min()} -> {new.min()}")
    return n


def scan_extreme(model):
    """Report every float constant that is non-finite or above 1e30."""
    bad = []

    def check(arr, where):
        if arr.dtype.kind != "f" or arr.size == 0:
            return
        if not np.all(np.isfinite(arr)):
            bad.append((where, "non-finite", float(np.nanmin(arr))))
        elif np.abs(arr).max() > EXTREME:
            bad.append((where, "extreme", float(arr.min())))

    for init in model.graph.initializer:
        check(numpy_helper.to_array(init), f"initializer {init.name}")
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.name == "value" and attr.t.ByteSize():
                check(numpy_helper.to_array(attr.t), f"{node.op_type} {node.name}")
    return bad


def op_histogram(model):
    h = {}
    for n in model.graph.node:
        h[n.op_type] = h.get(n.op_type, 0) + 1
    return h


def load(path):
    return onnx.load(str(path))


def save(model, path):
    onnx.save(model, str(path))
    return path
