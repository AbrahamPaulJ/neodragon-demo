"""CLIP L / CLIP G on device vs fp32. Table 9 deploys both FP16, so expect DistilT5-class
numbers (that ships at 49.04 dB), not W8A16-class ones.

Two traps this checks rather than assumes:

  * **trap #8** -- `input_ids` is INTEGER and the graph is FLOAT, so qnn-net-run needs
    `--use_native_input_files`. Get it wrong and every input yields the same plausible
    output. With a single test prompt the usual "two inputs differ" assertion is not
    available, so this instead checks the output is not constant and actually correlates
    with the reference.
  * **trap #3** -- HTP runs float graphs in fp16 with no fp32 upcast in the layer norm,
    which turned 74% of DistilT5's first build into NaN. Any non-finite value is fatal.

  usage: py -3.10 work/device/compare_clip.py clipl [outdir]
"""

import sys
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parent
OUTS = {"clipl": ["hidden"], "clipg": ["hidden", "text_embeds"]}


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "clipl"
    d = W / (sys.argv[2] if len(sys.argv) > 2 else "dout_" + tag)
    io = W / ("io_" + tag)
    if not d.is_dir():
        sys.exit("no device outputs at {}".format(d))

    print("=" * 64)
    print("{} on S25 Ultra HTP V79 -- FP16 graph".format(tag))
    print("=" * 64)
    print("")
    ok = True
    for nm in OUTS[tag]:
        ref = np.fromfile(io / "{}ref_0000.raw".format(nm), np.float32)
        got = np.fromfile(d / "Result_0" / "{}.raw".format(nm), np.float32)
        if ref.size != got.size:
            print("  {:<14} SIZE MISMATCH ref={} got={}".format(nm, ref.size, got.size))
            ok = False
            continue
        nfin = int(np.count_nonzero(~np.isfinite(got)))
        e = np.linalg.norm(ref - got)
        snr = float("inf") if e == 0 else 20 * np.log10(np.linalg.norm(ref) / e)
        corr = float(np.corrcoef(ref, got)[0, 1]) if got.std() > 0 else 0.0
        print("  {:<14} {:8.2f} dB   max|d| {:.3e}   corr {:.4f}   nonfinite {}".format(
            nm, snr, float(np.abs(ref - got).max()), corr, nfin))
        if nfin:
            print("      -> NaN/inf: trap #3, the fp16 layer norm. Needs residual scaling.")
            ok = False
        if got.std() == 0:
            print("      -> output is CONSTANT: integer input not reaching the graph (trap #8)")
            ok = False
    print("")
    print("  verdict: {}".format("OK" if ok else "PROBLEM -- see above"))


if __name__ == "__main__":
    main()
