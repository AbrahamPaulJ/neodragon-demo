"""How much arithmetic is the TAEHV decoder actually doing at T=1?

Context: the float build measures 10.88 s/inference on HTP V79 against the
paper's 248.9 ms. If the MAC count is large, the explanation is that HMX (the
matrix engine) only accelerates *quantised* convolution -- a float graph runs on
HVX vector units alone.
"""

import os
import sys
import types
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "src" / "neodragon"
MODEL = ROOT / "work" / "models" / "neodragon"
sys.path.insert(0, str(REPO))
_pkg = types.ModuleType("neodragon")
_pkg.__path__ = [str(REPO / "neodragon")]
sys.modules["neodragon"] = _pkg

from neodragon.asymmetric_causal_video_vae import AsymmetricCausalVideoVAE  # noqa: E402

T = 1


def main():
    vae = AsymmetricCausalVideoVAE.from_pretrained(
        MODEL / "causal_video_vae", torch_dtype=torch.float32).eval()

    class W(nn.Module):
        def __init__(s):
            super().__init__()
            s.post_quant_conv = vae.post_quant_conv
            s.decoder = vae.decoder

        def forward(s, z):
            return s.decoder(s.post_quant_conv(z))

    net = W().eval()

    macs = {}

    def hook(name, m):
        def f(_m, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                k = 1
                for d in m.kernel_size:
                    k *= d
                spatial = 1
                for d in o.shape[2:]:
                    spatial *= d
                n = o.shape[0] * o.shape[1] * spatial * k * (m.in_channels // m.groups)
                macs[name] = macs.get(name, 0) + n
        return f

    hs = [m.register_forward_hook(hook(n, m)) for n, m in net.named_modules()
          if isinstance(m, (nn.Conv2d, nn.Conv3d))]
    z = torch.randn(1, 16, T, 40, 64)
    with torch.no_grad():
        out = net(z)
    for h in hs:
        h.remove()

    total = sum(macs.values())
    print("=" * 66)
    print(f"TAEHV decoder MAC count, T={T} -> {tuple(out.shape)}")
    print("=" * 66)
    print(f"\n  total {total/1e9:.2f} GMAC  ({2*total/1e9:.2f} GFLOP)")

    print("\n  top convolutions:")
    for k in sorted(macs, key=lambda x: -macs[x])[:8]:
        print(f"    {macs[k]/1e9:>7.2f} GMAC  {k}")

    measured_s = 10.877
    paper_ms = 248.9
    print(f"\n  measured float on HTP V79 : {measured_s*1000:>9.1f} ms"
          f"  -> {total/1e9/measured_s:>8.2f} GMAC/s")
    print(f"  paper (W8A16)             : {paper_ms:>9.1f} ms"
          f"  -> {total/1e9/(paper_ms/1000):>8.2f} GMAC/s")
    print(f"  ratio                     : {measured_s*1000/paper_ms:>9.1f}x slower")
    print("\n  For scale, SM8750's NPU is quoted in the tens of TOPS for int8.")
    print("  Hitting only single-digit GMAC/s means HMX is not being used at all,")
    print("  which is consistent with float convolution running on HVX alone.")


if __name__ == "__main__":
    main()
