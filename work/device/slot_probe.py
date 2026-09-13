"""Is a broken image stream broken everywhere, or only in one patchify slot?

Stage 1 is the first MMDiT graph whose latents have MIXED spatial sizes -- slot 0 is
5x10x16 while slots 1 and 2 are 1x20x32 -- and its image stream measures -1.5 dB from
block 1 onward while its text stream is a healthy 17 dB. Stage 0, whose three latents are
all 10x16, is fine. So the question is which of the three token groups is wrong.

Splits one dumped [1, S, C] tensor into its per-slot token ranges and scores each
separately against onnxruntime, and also reports whether the device values are a
PERMUTATION of the reference (sorted-value SNR) -- a scrambled token order scores ~0 dB
on the aligned comparison but near-perfect on the sorted one.

  usage: py -3.10 work/device/slot_probe.py --tensor _Add_23_output_0 --stage 1
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

TEXT_TOKENS = 128


def snr(ref, got):
    ref = np.asarray(ref, np.float64).ravel()
    got = np.asarray(got, np.float64).ravel()
    d = np.linalg.norm(ref - got)
    return float("inf") if d == 0 else 20 * np.log10(np.linalg.norm(ref) / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", default="_Add_23_output_0")
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--name", default="")
    ap.add_argument("--dump", default="")
    ap.add_argument("--case", type=int, default=0)
    a = ap.parse_args()

    import onnx
    import onnxruntime as ort
    from layer_snr import sanitize
    from export_mmdit_stage import envelope_for
    from mmdit_shapes import tokens

    suf = (a.name[6:] if a.name.startswith("mmdit_") else a.name) or "s{}f".format(a.stage)
    onnx_path = ND / "work/onnx/mmdit_{0}_env/mmdit_{0}_env_raw.onnx".format(suf)
    dump = Path(a.dump or W / "dblk_{}".format(suf)) / "Result_{}".format(a.case)
    io_dir = W / "mio_{}".format(suf)

    env = envelope_for(a.stage)
    slots = [tokens(s) for s in env]
    print("envelope {}  -> slot token counts {}".format(
        " ".join("{}x{}x{}".format(*s) for s in env), slots))

    model = onnx.load(str(onnx_path), load_external_data=False)
    g = model.graph
    want = None
    for node in g.node:
        for o in node.output:
            if sanitize(o) == a.tensor:
                want = o
                break
        if want:
            break
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
    tmp = onnx_path.parent / "_slot_probe_tmp.onnx"
    onnx.save(model, str(tmp))
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    ref = ort.InferenceSession(str(tmp), so, providers=["CPUExecutionProvider"]).run(
        None, feed)[0]
    tmp.unlink(missing_ok=True)

    dev = np.fromfile(dump / (a.tensor + ".raw"), np.float32).reshape(ref.shape)
    print("")
    print("tensor {}  shape {}".format(a.tensor, tuple(ref.shape)))
    print("  whole tensor              : {:7.2f} dB".format(snr(ref, dev)))
    print("  sorted values (permutation test): {:7.2f} dB".format(
        snr(np.sort(ref.ravel()), np.sort(dev.ravel()))))

    n_tok = ref.shape[1]
    is_text = n_tok == TEXT_TOKENS
    print("")
    if is_text:
        print("  (text stream -- no image slots)")
        return
    off = 0
    if n_tok == TEXT_TOKENS + sum(slots):
        print("  {:<22} {:>10} {:>12}".format("range", "SNR dB", "max|ref|"))
        print("  " + "-" * 46)
        print("  {:<22} {:>10.2f} {:>12.3f}".format(
            "text 0:{}".format(TEXT_TOKENS), snr(ref[:, :TEXT_TOKENS], dev[:, :TEXT_TOKENS]),
            float(np.abs(ref[:, :TEXT_TOKENS]).max())))
        off = TEXT_TOKENS
    else:
        print("  {:<22} {:>10} {:>12}".format("range", "SNR dB", "max|ref|"))
        print("  " + "-" * 46)
    for i, n in enumerate(slots):
        sl = slice(off, off + n)
        print("  {:<22} {:>10.2f} {:>12.3f}".format(
            "slot {} [{}:{}]".format(i, off, off + n),
            snr(ref[:, sl], dev[:, sl]), float(np.abs(ref[:, sl]).max())))
        off += n


if __name__ == "__main__":
    main()
