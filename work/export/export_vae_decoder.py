"""Phase 4b -- TinyAEHV (TAEHV) video VAE decoder -> ONNX.

The decoder is the light half of the "asymmetric" VAE: pure Conv2d, ReLU,
Upsample, Tanh and reshapes. The CausalConv3d / deque-cache machinery in
modeling_causal_ops.py belongs to the ENCODER and is not on this path. The only
3-D op here is post_quant_conv, a 1x1x1 CausalConv3d (time_pad = 0).

Two shapes are real per paper Fig 13:
  * first frame   [1,16,1,40,64] -> [1,3, 1,320,512]
  * video         [1,16,7,40,64] -> [1,3,49,320,512]
`frames_to_trim = 2**3 - 1 = 7`, so out_frames = 8*T - 7.
"""

import argparse
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
REPO = ROOT / "src" / "neodragon"
MODEL = ROOT / "work" / "models" / "neodragon"
OUT = ROOT / "work" / "onnx"
sys.path.insert(0, str(REPO))
_pkg = types.ModuleType("neodragon")
_pkg.__path__ = [str(REPO / "neodragon")]
sys.modules["neodragon"] = _pkg

from graph_fixes import load, op_histogram, save, scan_extreme  # noqa: E402
from neodragon.asymmetric_causal_video_vae import AsymmetricCausalVideoVAE  # noqa: E402

LAT_C, LAT_H, LAT_W = 16, 40, 64


class DecodeWrapper(nn.Module):
    """AsymmetricCausalVideoVAE.decode() without the dataclass wrapper."""

    def __init__(self, vae):
        super().__init__()
        self.post_quant_conv = vae.post_quant_conv
        self.decoder = vae.decoder

    def forward(self, z):
        z = self.post_quant_conv(z)
        return self.decoder(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=1, help="latent temporal length T")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    T = args.frames
    tag = args.tag or f"vaedec_t{T}"
    out_frames = 8 * T - 7

    print("=" * 76)
    print(f"PHASE 4b -- TAEHV decoder -> ONNX   T={T} -> {out_frames} frames")
    print("=" * 76)

    vae = AsymmetricCausalVideoVAE.from_pretrained(
        MODEL / "causal_video_vae", torch_dtype=torch.float32).eval()
    net = DecodeWrapper(vae).eval()

    n_par = sum(p.numel() for p in net.parameters())
    print(f"\n[model] decoder params {n_par/1e6:.2f} M")
    print(f"[model] frames_to_trim {vae.decoder.frames_to_trim}")

    torch.manual_seed(0)
    z = torch.randn(1, LAT_C, T, LAT_H, LAT_W)

    with torch.no_grad():
        ref = net(z)
    print(f"[shape] {tuple(z.shape)} -> {tuple(ref.shape)}")
    exp = (1, 3, out_frames, LAT_H * 8, LAT_W * 8)
    assert tuple(ref.shape) == exp, f"expected {exp}, got {tuple(ref.shape)}"
    print(f"[range] out min={ref.min():.4f} max={ref.max():.4f} "
          f"({ref.numel()*4/1024**2:.0f} MB fp32)")

    # ---- export -------------------------------------------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT / f"{tag}_raw.onnx"
    torch.onnx.export(
        net, (z,), str(raw),
        input_names=["latent"], output_names=["video"],
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    print(f"\n[export] {raw.name}  {raw.stat().st_size/1024**2:.1f} MB")

    # ---- fold ---------------------------------------------------------------
    import onnxruntime as ort
    dst = OUT / f"{tag}_qnn.onnx"
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = str(dst)
    ort.InferenceSession(str(raw), so, providers=["CPUExecutionProvider"])
    mm = load(dst)
    keep = [o for o in mm.opset_import if (o.domain or "") == ""]
    del mm.opset_import[:]
    mm.opset_import.extend(keep)
    save(mm, dst)

    h = op_histogram(mm)
    print(f"[fold]   {len(load(raw).graph.node)} -> {sum(h.values())} nodes")
    print("[ops]    " + ", ".join(f"{k}x{v}" for k, v in
                                  sorted(h.items(), key=lambda kv: -kv[1])))
    print(f"[safe]   unsafe constants: {len(scan_extreme(mm))}")

    # ---- verify -------------------------------------------------------------
    so2 = ort.SessionOptions()
    so2.log_severity_level = 3
    sess = ort.InferenceSession(str(dst), so2, providers=["CPUExecutionProvider"])
    got = torch.from_numpy(sess.run(None, {"latent": z.numpy()})[0])
    d = (got - ref)
    snr = 20 * torch.log10(ref.norm() / d.norm()).item()
    print(f"[verify] onnx vs torch: max|diff|={d.abs().max():.3e}  SNR={snr:.1f} dB")
    print(f"\nwrote {dst}  ({dst.stat().st_size/1024**2:.1f} MB)")


if __name__ == "__main__":
    main()
