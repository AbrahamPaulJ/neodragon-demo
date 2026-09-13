"""Rewrite the MemBlock state raws from NCHW to NHWC, in place.

For the `--preserve_io layout latent video` build. Leaving the 9 state inputs and
9 state outputs out of the preserve list lets them be channel-last, which removes
the last 18 boundary Transposes (55 of the 63 MB still moving in the vaedecs
build). The states are opaque round-trip data -- `nstate_k` feeds straight back
into `state_k` -- so nothing on the device ever has to interpret them, and only
this host-side raw generation needs to know the layout.

`latent` and `video` STAY NCHW: those the host does interpret, and trap #7 applies
to them exactly as before.

In place, because the full set is ~3 GB and duplicating it is most of the
remaining headroom on C:. A sentinel file guards against a double permute, which
would silently scramble the data rather than error.

  usage: py -3.10 work/device/permute_states_nhwc.py [--revert]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CALIB = ROOT / "work" / "calib" / "vae_dec_stream"
SENTINEL = CALIB / ".layout_nhwc"

# state_k index -> NCHW shape, from work/export/export_vae_decoder_stream.py
SHAPES = {0: (1, 256, 40, 64), 1: (1, 256, 40, 64), 2: (1, 256, 40, 64),
          3: (1, 128, 80, 128), 4: (1, 128, 80, 128), 5: (1, 128, 80, 128),
          6: (1, 64, 160, 256), 7: (1, 64, 160, 256), 8: (1, 64, 160, 256)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true", help="NHWC -> NCHW")
    args = ap.parse_args()

    if args.revert and not SENTINEL.exists():
        sys.exit("states are already NCHW -- nothing to revert")
    if not args.revert and SENTINEL.exists():
        sys.exit(f"states are already NHWC ({SENTINEL}).\n"
                 f"permuting twice would scramble them; use --revert first")

    files = sorted(CALIB.glob("state_*.raw"))
    if not files:
        sys.exit(f"no state raws in {CALIB}")
    print(f"{'NHWC -> NCHW' if args.revert else 'NCHW -> NHWC'}: {len(files)} files")

    moved = 0
    for f in files:
        k = int(f.name.split("_")[1])
        nchw = SHAPES[k]
        a = np.fromfile(f, dtype=np.float32)
        if args.revert:
            nhwc = (nchw[0], nchw[2], nchw[3], nchw[1])
            a.reshape(nhwc).transpose(0, 3, 1, 2).copy().tofile(f)
        else:
            a.reshape(nchw).transpose(0, 2, 3, 1).copy().tofile(f)
        moved += a.nbytes

    if args.revert:
        SENTINEL.unlink()
    else:
        SENTINEL.write_text("state_*.raw are NHWC (channel-last)\n")
    print(f"rewrote {moved/1e9:.2f} GB")
    print(f"sentinel: {'removed' if args.revert else SENTINEL}")


if __name__ == "__main__":
    main()
