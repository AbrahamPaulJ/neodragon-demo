"""Phase 1a v4 -- fp16 activations throughout the graph.

Hypothesis: the graph declares fp32 tensors even though HTP executes in fp16 and
the context binary stores fp16 weights, so there may be conversion overhead or
some ops running in fp32. Declaring fp16 might remove it.

Known risk, stated up front: T5's RMSNorm computes mean(h^2). With residual
scaling S=16 the peak residual is 20,594, so h^2 peaks at 4.2e8 -- far beyond
fp16's 65504. The fp32 graph is safe because HTP's fused rms_norm handles
accumulation internally. Forcing fp16 at the ONNX level may break that, which is
exactly what this experiment measures.

`--scale` lets the residual be held small enough that h^2 (and its sum) stay in
fp16 range, as a fallback if plain fp16 conversion degrades accuracy.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import onnx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
ONNX = ROOT / "work" / "onnx"
IO = ROOT / "work" / "device" / "io"

from graph_fixes import op_histogram, scan_extreme  # noqa: E402

FP16_MAX = 65504.0


def analyse_risk(scale):
    """Report whether mean(h^2) can be computed in fp16 at this scale."""
    peak_unscaled = 329509.0
    n_outlier = 5           # channels persistently above 1% of peak (trap-audit §3)
    peak = peak_unscaled / scale
    sq = peak ** 2
    approx_sum = n_outlier * sq
    print(f"  S={scale:<6g} peak residual {peak:>10,.1f}  "
          f"h^2 peak {sq:>12,.0f} {'OK ' if sq < FP16_MAX else 'OVF'}  "
          f"~sum(h^2) {approx_sum:>13,.0f} {'OK' if approx_sum < FP16_MAX else 'OVF'}")
    return sq < FP16_MAX and approx_sum < FP16_MAX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="distilt5_folded_qnn")
    ap.add_argument("--dst", default="distilt5_fp16_qnn")
    args = ap.parse_args()

    from onnxconverter_common import float16

    print("=" * 76)
    print("PHASE 1a v4 -- fp16 activation graph")
    print("=" * 76)

    print("\n[risk] can mean(h^2) be held in fp16?")
    for s in (16, 256, 2048, 4096, 8192):
        analyse_risk(s)
    print("  (current build uses S=16, so a plain fp16 interior should overflow the")
    print("   norm unless the runtime accumulates wider -- that is the open question)")

    src = ONNX / f"{args.src}.onnx"
    m = onnx.load(str(src))
    print(f"\n[src] {src.name}  {src.stat().st_size/1024**2:.1f} MB  "
          f"{len(m.graph.node)} nodes")

    m16 = float16.convert_float_to_float16(
        m, keep_io_types=True, disable_shape_infer=False)

    # onnxconverter_common leaves Cast nodes that still declare to=FLOAT while
    # their consumers now expect FLOAT16. Retarget any Cast that is not feeding
    # a graph output.
    out_names = {o.name for o in m16.graph.output}
    fixed = 0
    for node in m16.graph.node:
        if node.op_type != "Cast":
            continue
        if any(o in out_names for o in node.output):
            continue
        for attr in node.attribute:
            if attr.name == "to" and attr.i == onnx.TensorProto.FLOAT:
                attr.i = onnx.TensorProto.FLOAT16
                fixed += 1
    if fixed:
        print(f"  retargeted {fixed} Cast node(s) FLOAT -> FLOAT16")
    m16 = onnx.shape_inference.infer_shapes(m16, strict_mode=False)

    dst = ONNX / f"{args.dst}.onnx"
    onnx.save(m16, str(dst))
    print(f"[fp16] {dst.name}  {dst.stat().st_size/1024**2:.1f} MB  "
          f"{len(m16.graph.node)} nodes")

    h0, h1 = op_histogram(m), op_histogram(m16)
    added = {k: h1.get(k, 0) - h0.get(k, 0) for k in set(h0) | set(h1)}
    print("  node deltas: " + ", ".join(f"{k} {v:+d}" for k, v in sorted(added.items())
                                        if v))
    print(f"  unsafe constants: {len(scan_extreme(m16))}")

    # ---- CPU sanity check ---------------------------------------------------
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(dst), so, providers=["CPUExecutionProvider"])
    itype = {i.name: i.type for i in sess.get_inputs()}
    print(f"\n[io] {itype} -> {[(o.name, o.type) for o in sess.get_outputs()]}")

    print("\n[cpu check] note: ORT's CPU EP may internally widen fp16 ops, so a")
    print("            clean result here does NOT prove the device will be clean.")
    for i in range(3):
        ids = np.fromfile(IO / f"input_ids_{i}.raw", dtype=np.int32).reshape(1, 128)
        am = np.fromfile(IO / f"attention_mask_{i}.raw", dtype=np.int32).reshape(1, 128)
        if "int64" in itype["input_ids"]:
            ids, am = ids.astype(np.int64), am.astype(np.int64)
        out = sess.run(None, {"input_ids": ids, "attention_mask": am})[0].astype(np.float32)
        ref = np.fromfile(IO / f"ref_{i}.raw", dtype=np.float32).reshape(1, 128, 4096)
        bad = int(np.isnan(out).sum() + np.isinf(out).sum())
        if bad:
            print(f"  case {i}: nan+inf={bad}  -- fp16 interior BREAKS the graph")
            continue
        d = ref - out
        snr = 20 * np.log10(np.linalg.norm(ref) / np.linalg.norm(d))
        print(f"  case {i}: SNR={snr:6.2f} dB  max|d|={np.abs(d).max():.3e}  nan+inf=0")


if __name__ == "__main__":
    main()
