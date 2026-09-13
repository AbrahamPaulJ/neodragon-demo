"""Per-row SNR of one dumped [1, N, C] tensor, to see WHICH rows a broken op got wrong.

Stage 1's image-stream LayerNorm takes a 52.82 dB input to a 2.73 dB output while the
same node computes the 128-row text tensor at 72.37 dB. Whether the damage is spread
over all N rows or confined to a contiguous block decides the fix: a threshold effect
means "normalise in chunks", a uniform one means the op is wrong for this shape outright.

  usage: py -3.10 work/device/rowscan.py --tensor _norm_LayerNormalization_output_0 \
                  --stage 1 --dump work/device/dln_s1f
"""

import argparse
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
    ap.add_argument("--dump", required=True)
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--group", type=int, default=20, help="rows per printed bucket")
    a = ap.parse_args()

    import onnx
    import onnxruntime as ort
    from layer_snr import sanitize

    suf = (a.name[6:] if a.name.startswith("mmdit_") else a.name) or "s{}f".format(a.stage)
    onnx_path = ND / "work/onnx/mmdit_{0}_env/mmdit_{0}_env_raw.onnx".format(suf)
    io_dir = W / "mio_{}".format(suf)
    res = Path(a.dump) / "Result_{}".format(a.case)

    model = onnx.load(str(onnx_path), load_external_data=False)
    g = model.graph
    want = next((o for n in g.node for o in n.output if sanitize(o) == a.tensor), None)
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
    tmp = onnx_path.parent / "_rowscan_tmp.onnx"
    onnx.save(model, str(tmp))
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    ref = ort.InferenceSession(str(tmp), so, providers=["CPUExecutionProvider"]).run(
        None, feed)[0]
    tmp.unlink(missing_ok=True)

    dev = np.fromfile(res / (a.tensor + ".raw"), np.float32).reshape(ref.shape)
    r2 = ref.reshape(-1, ref.shape[-1])
    d2 = dev.reshape(-1, dev.shape[-1])
    n = r2.shape[0]
    print("{}  shape {}  -> {} rows of {}".format(a.tensor, tuple(ref.shape), n, r2.shape[1]))
    print("  whole tensor: {:.2f} dB".format(snr(ref, dev)))
    print("")
    print("  {:<14} {:>10} {:>12} {:>12}".format("rows", "SNR dB", "max|ref|", "max|dev|"))
    print("  " + "-" * 52)
    for s in range(0, n, a.group):
        e = min(s + a.group, n)
        print("  {:<14} {:>10.2f} {:>12.4f} {:>12.4f}".format(
            "{}:{}".format(s, e), snr(r2[s:e], d2[s:e]),
            float(np.abs(r2[s:e]).max()), float(np.abs(d2[s:e]).max())))


if __name__ == "__main__":
    main()
