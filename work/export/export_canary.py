"""A tiny FP16 canary graph, shipped INSIDE the APK, run before any 8 GB download.

Two failure modes this catches, both of which currently waste the user's bandwidth and
then surface as an unhelpful "Create From Binary failure" hours later:

1. **Wrong Hexagon architecture.** Every binary in this project is compiled for HTP
   **v79** / SM8750 (`soc_model 69`). On a v75 or earlier phone the context binary simply
   will not deserialize. Checking `Build.SOC_MODEL` against an allowlist would be a guess
   -- a canary that actually loads and executes proves the backend, the skel, and the
   arch all agree.

2. **FP16 arithmetic misbehaving.** HTP runs float graphs in fp16 with no fp32 upcast in
   the layer norm (trap #3 -- the unscaled DistilT5 build returned 74% NaN on device).
   The pipeline ships two float graphs (`ctxadaptfp16`, `distilt5f`), so a device or
   driver where fp16 is wrong produces plausible-looking garbage rather than an error.

The graph deliberately exercises the operations that matter: a Gemm, a residual add, a
LayerNorm and a Softmax -- the shapes trap #3 blew up on, in miniature. It is a few KB, so
it ships as an asset and costs nothing.

  usage: py -3.10 work/export/export_canary.py
"""

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "onnx" / "canary"
ASSETS = ROOT / "work" / "android" / "app" / "src" / "main" / "assets"

N = 64


class Canary(nn.Module):
    """Deterministic, weights baked in, no randomness at inference."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(N, N)
        self.ln = nn.LayerNorm(N)
        # Fixed, well-conditioned weights: a canary must fail because the DEVICE is
        # wrong, never because the weights happened to be numerically delicate.
        g = torch.Generator().manual_seed(7)
        with torch.no_grad():
            self.fc.weight.copy_(torch.randn(N, N, generator=g) * (1.0 / N ** 0.5))
            self.fc.bias.copy_(torch.linspace(-0.5, 0.5, N))
            self.ln.weight.fill_(1.0)
            self.ln.bias.zero_()

    def forward(self, x):
        h = self.fc(x)
        h = h + x                      # residual -- trap #3's shape
        h = self.ln(h)
        return torch.softmax(h, dim=-1)


def main():
    m = Canary().eval()
    # A fixed probe, spanning positive and negative, nothing near fp16's 65504 ceiling:
    # this tests CORRECTNESS, not overflow behaviour.
    x = torch.linspace(-2.0, 2.0, N).reshape(1, N)
    with torch.no_grad():
        y = m(x)

    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT / "canary_raw.onnx"
    torch.onnx.export(m, (x,), str(raw), input_names=["x"], output_names=["y"],
                      opset_version=17, do_constant_folding=True, dynamo=False)
    print(f"[export] {raw}  ({raw.stat().st_size} bytes)")

    ASSETS.mkdir(parents=True, exist_ok=True)
    x.numpy().astype(np.float32).tofile(ASSETS / "canary_in.raw")
    y.numpy().astype(np.float32).tofile(ASSETS / "canary_ref.raw")
    print(f"[assets] canary_in.raw / canary_ref.raw  ({N} floats each)")
    print(f"[ref]    sum={float(y.sum()):.6f} (softmax, so 1.0)  "
          f"max={float(y.max()):.6f}  min={float(y.min()):.6e}")

    dims = ROOT / "work" / "calib" / "canary"
    dims.mkdir(parents=True, exist_ok=True)
    (dims / "dims.txt").write_text(f"--input_dim x 1,{N}\n", newline="\n")
    print(f"[dims]   {dims / 'dims.txt'}")
    print("")
    print("next: convert as a FLOAT graph (no --input_list, no quantisation flags) so it")
    print("      runs in fp16 on the HTP exactly as ctxadaptfp16 and distilt5f do")


if __name__ == "__main__":
    main()
