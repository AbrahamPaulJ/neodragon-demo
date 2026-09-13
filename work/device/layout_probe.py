"""Is a dumped rank-3 tensor a LAYOUT permutation of the ONNX reference, or genuinely wrong?

Trap #23: QNN has a canonical layout for rank 3 (NCF/NFC) exactly as it does for rank 4,
and it picks per tensor. A dumped raw is written in the DEVICE's layout, so a tensor the
backend decided to keep as NFC reads as garbage against an NCF reference -- the rank-3
sibling of trap #32, which is the same thing for conv outputs at rank 4.

Scores the device raw against the reference under every rank-3 reinterpretation, so a
permutation announces itself as a near-perfect score in one of the non-identity rows.

  usage: py -3.10 work/device/layout_probe.py --tensor _Add_23_output_0 --stage 1
"""

import argparse
import itertools
import re
import sys
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parent
ND = W.parent.parent
sys.path.insert(0, str(W))
sys.path.insert(0, str(ND / "work" / "export"))
sys.path.insert(0, str(ND / "work" / "audit"))


def snr(ref, got):
    ref = np.asarray(ref, np.float64).ravel()
    got = np.asarray(got, np.float64).ravel()
    d = np.linalg.norm(ref - got)
    return float("inf") if d == 0 else 20 * np.log10(np.linalg.norm(ref) / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", required=True)
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--name", default="")
    ap.add_argument("--dump", default="")
    ap.add_argument("--case", type=int, default=0)
    a = ap.parse_args()

    import onnx
    import onnxruntime as ort
    from layer_snr import sanitize

    suf = (a.name[6:] if a.name.startswith("mmdit_") else a.name) or "s{}f".format(a.stage)
    onnx_path = ND / "work/onnx/mmdit_{0}_env/mmdit_{0}_env_raw.onnx".format(suf)
    dump = Path(a.dump or W / "dblk_{}".format(suf)) / "Result_{}".format(a.case)
    io_dir = W / "mio_{}".format(suf)

    model = onnx.load(str(onnx_path), load_external_data=False)
    g = model.graph
    want = next((o for node in g.node for o in node.output
                 if sanitize(o) == a.tensor), None)
    if want is None:
        sys.exit("no ONNX tensor sanitises to {}".format(a.tensor))

    dims = {}
    for line in open(io_dir / "dims.txt"):
        m = re.search(r"--input_dim\s+(\S+)\s+([0-9,]+)", line)
        if m:
            dims[m.group(1)] = tuple(int(x) for x in m.group(2).split(","))
    feed = {nm: np.fromfile(io_dir / "{}_{:04d}.raw".format(nm, a.case),
                            np.float32).reshape(shp) for nm, shp in dims.items()}

    del g.output[:]
    g.output.extend([onnx.helper.make_empty_tensor_value_info(want)])
    tmp = onnx_path.parent / "_layout_probe_tmp.onnx"
    onnx.save(model, str(tmp))
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    ref = ort.InferenceSession(str(tmp), so, providers=["CPUExecutionProvider"]).run(
        None, feed)[0]
    tmp.unlink(missing_ok=True)

    flat = np.fromfile(dump / (a.tensor + ".raw"), np.float32)
    shape = tuple(ref.shape)
    print("tensor {}  onnx shape {}  ({} elems)".format(a.tensor, shape, flat.size))
    assert flat.size == ref.size, (flat.size, ref.size)

    print("")
    print("  {:<24} {:>10}".format("device raw read as", "SNR dB"))
    print("  " + "-" * 36)
    rows = []
    for perm in itertools.permutations(range(len(shape))):
        src = tuple(shape[i] for i in perm)          # the shape the device may have used
        inv = np.argsort(perm)
        got = flat.reshape(src).transpose(inv)
        rows.append((snr(ref, got), perm, src))
    rows.sort(reverse=True)
    for s, perm, src in rows:
        tag = "identity" if perm == tuple(range(len(shape))) else "perm {}".format(perm)
        print("  {:<24} {:>10.2f}".format("{} {}".format(tag, src), s))
    print("")
    print("  sorted values (any permutation at all): {:.2f} dB".format(
        snr(np.sort(ref.ravel()), np.sort(flat))))


if __name__ == "__main__":
    main()
