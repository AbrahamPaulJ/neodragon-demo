"""ContextAdapter on device vs the fp32 reference, on held-out prompts.

Paper Table 9 gives no row for the ContextAdapter -- it is one of the two modules the
phased plan never listed (the other was the VAE encoder, Phase 4b). It sits between
DistilT5 and the MMDiT and produces `encoder_hidden_states [1,128,1536]`, so its error
lands directly on the MMDiT's text stream. Treat the MMDiT's own text-stream quality as
the bar: stage 0 carries text at 20-27 dB, so anything comfortably above that is free.

  usage: py -3.10 work/device/compare_ctxadapt.py
"""

import sys
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parent


def main():
    io = W / "io_ctxadapt"
    d = W / (sys.argv[1] if len(sys.argv) > 1 else "dout_ctxadapt")
    refs = sorted(io.glob("cref_*.raw"))
    if not refs:
        sys.exit("no references in {}".format(io))
    if not d.is_dir():
        sys.exit("no device outputs at {}".format(d))

    print("=" * 60)
    print("ContextAdapter on S25 Ultra HTP V79 -- held-out prompts")
    print("=" * 60)
    print("")
    print("  {:>4} {:>10} {:>12}".format("case", "SNR dB", "max|d|"))
    print("  " + "-" * 28)
    rows = []
    for c in range(len(refs)):
        ref = np.fromfile(io / "cref_{:04d}.raw".format(c), np.float32)
        got = np.fromfile(d / "Result_{}".format(c) / "context.raw", np.float32)
        err = np.linalg.norm(ref - got)
        snr = float("inf") if err == 0 else 20 * np.log10(np.linalg.norm(ref) / err)
        rows.append(snr)
        print("  {:>4} {:>10.2f} {:>12.3e}".format(c, snr, float(np.abs(ref - got).max())))

    a = np.array(rows)
    print("")
    print("  mean {:.2f} dB   min {:.2f} dB".format(a.mean(), a.min()))
    g0 = np.fromfile(d / "Result_0" / "context.raw", np.float32)
    g1 = np.fromfile(d / "Result_1" / "context.raw", np.float32)
    md = float(np.abs(g0 - g1).max())
    print("  distinct outputs across cases 0/1: max|d| = {:.4f}  {}".format(
        md, "OK" if md > 1e-3 else "FAILED -- inputs not reaching graph (trap #8)"))


if __name__ == "__main__":
    main()
