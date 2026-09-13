"""Phase 1a, part 2 -- produce the QNN-ready DistilT5 ONNX.

  1. rewrite the -3.4e38 mask constant to a finite value (trap #2, T5 side)
  2. fold the constant subgraphs left over from the relative-position-bucket
     computation, which is static once seq is pinned at 128
  3. verify against the PyTorch reference
  4. report the op set the QNN converter will have to cover
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import onnx
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
OUT = ROOT / "work" / "onnx"

from graph_fixes import (load, op_histogram, retarget_extreme_constants,  # noqa: E402
                         save, scan_extreme)

MASK_FILL = -100.0
SEQ = 128


def ort_session(path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def ort_fold(src, dst):
    """Bake ORT's constant folding / graph optimisation into a saved model."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = str(dst)
    ort.InferenceSession(str(src), so, providers=["CPUExecutionProvider"])
    return dst


def run(path, feeds):
    return ort_session(path).run(None, feeds)[0]


def main():
    src = OUT / "distilt5_baseline.onnx"
    if not src.exists():
        sys.exit(f"missing {src} -- run export_distilt5.py first")

    rng = np.random.default_rng(0)
    ids = np.zeros((1, SEQ), dtype=np.int64)
    n_real = 14
    ids[0, :n_real] = rng.integers(100, 30000, n_real)
    ids[0, n_real - 1] = 1  # eos
    am = np.zeros((1, SEQ), dtype=np.int64)
    am[0, :n_real] = 1
    feeds = {"input_ids": ids, "attention_mask": am}

    print("=" * 78)
    print("PHASE 1a part 2 -- QNN-ready DistilT5 ONNX")
    print("=" * 78)

    ref = run(src, feeds)
    print(f"\n[reference] baseline ONNX out {ref.shape} max|.|={np.abs(ref).max():.4f}")

    # ---- 1. mask constant ---------------------------------------------------
    print(f"\n[1] retarget extreme constants -> {MASK_FILL}")
    m = load(src)
    n = retarget_extreme_constants(m, MASK_FILL)
    fixed = save(m, OUT / "distilt5_maskfix.onnx")
    print(f"    rewrote {n} constant(s)")
    left = scan_extreme(load(fixed))
    print(f"    remaining unsafe constants: {len(left)}")

    got = run(fixed, feeds)
    err = np.abs(got - ref).max()
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    print(f"    vs baseline: max|diff|={err:.3e}  rel={rel:.3e}")

    # ---- 2. constant folding ------------------------------------------------
    print("\n[2] fold static subgraphs (seq is pinned at 128)")
    before = op_histogram(load(fixed))
    folded = ort_fold(fixed, OUT / "distilt5_qnn.onnx")
    after = op_histogram(load(folded))
    print(f"    nodes {sum(before.values())} -> {sum(after.values())}")
    gone = {k: before.get(k, 0) - after.get(k, 0) for k in set(before) | set(after)}
    for k in sorted(gone, key=lambda x: -gone[x]):
        if gone[k]:
            print(f"      {k:<12} {before.get(k,0):>4} -> {after.get(k,0):>4}")

    got = run(folded, feeds)
    err = np.abs(got - ref).max()
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    print(f"    vs baseline: max|diff|={err:.3e}  rel={rel:.3e}")

    # ---- 3. final report ----------------------------------------------------
    final = load(folded)
    print(f"\n[3] final graph: {OUT/'distilt5_qnn.onnx'}")
    print(f"    size {(OUT/'distilt5_qnn.onnx').stat().st_size/1024**2:.1f} MB")
    print(f"    unsafe constants: {len(scan_extreme(final))}")
    print(f"    opset: {[(o.domain or 'ai.onnx', o.version) for o in final.opset_import]}")
    for i in final.graph.input:
        dims = [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]
        print(f"    input  {i.name:<16} {dims}")
    for o in final.graph.output:
        dims = [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]
        print(f"    output {o.name:<16} {dims}")

    print("\n    op histogram:")
    h = op_histogram(final)
    for k in sorted(h, key=lambda x: -h[x]):
        print(f"      {h[k]:>5}  {k}")

    # ops that commonly need attention on HTP
    watch = {"Pow", "Sqrt", "ReduceMean", "Softmax", "Tanh", "Erf", "Where",
             "Log", "Min", "Abs", "Less", "Cast", "Div"}
    present = sorted(watch & set(h))
    print(f"\n    ops to confirm against the QNN op list: {', '.join(present)}")


if __name__ == "__main__":
    main()
