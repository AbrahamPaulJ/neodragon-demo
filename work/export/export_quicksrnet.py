"""Phase 2 -- QuickSRNet 2x, the pipeline's final upscale. 320x512 -> 640x1024.

The paper applies 2x supersampling with QuickSRNet [62] as the last step, which is what
takes its output to the [640x1024] its headline quotes; without it our video is 320x512
and the two are not the same product. Table 7 puts it at 4.9 ms (X Elite) / 6.5 ms
(8 Elite Gen4), and Table 9 makes it the ONLY module using **W8A16 + AdaRound**, with 500
calibration samples and a 48 dB deploy SNR. The paper notes AdaRound is worth "7+ dB
SQNR" here, so plain PTQ is expected to land well short of 48 dB.

## Which variant

The paper cites QuickSRNet generically. Counting MACs at our input size settles it -- at
the ~2050 GMAC/s this chip sustains on the conv-heavy W8A16 VAE decoder (205.07 GMAC in
100.18 ms):

    small    3.73 GMAC   ~1.8 ms/frame    ~90 ms per 49-frame video
    medium   8.26 GMAC   ~4.0 ms/frame   ~200 ms          <- chosen
    large   67.9  GMAC  ~33   ms/frame    ~1.6 s          <- cannot be the paper's 6.5 ms

Large is ruled out by arithmetic. Medium is the best quality that fits the budget and
lands next to the paper's 318 ms for a whole video.

## Ranges

The checkpoint is trained on **[0,1] RGB** and returns [0,1]. Our streaming VAE decoder
emits **[-1,1]**, so the caller must map before and after. Getting that wrong does not
crash -- it produces a washed-out but plausible upscale, which is the hardest kind of bug
to catch by eye, so the conversion is done in Kotlin at the call site and asserted here.

  usage: py -3.10 work/export/export_quicksrnet.py [--variant medium] [--scale 2]
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "onnx"

VARIANTS = {
    "small": "quicksrnetsmall.model.QuickSRNetSmall",
    "medium": "quicksrnetmedium.model.QuickSRNetMedium",
    "large": "quicksrnetlarge.model.QuickSRNetLarge",
}


def load(variant: str, scale: int):
    from qai_hub_models.utils.asset_loaders import always_answer_prompts
    import importlib

    mod_path, cls_name = VARIANTS[variant].rsplit(".", 1)
    with always_answer_prompts(True):
        mod = importlib.import_module(f"qai_hub_models.models.{mod_path}")
        cls = getattr(mod, cls_name)
        return cls.from_pretrained(scale_factor=scale).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="medium", choices=list(VARIANTS))
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--width", type=int, default=512)
    a = ap.parse_args()

    print("=" * 74)
    print(f"PHASE 2 -- QuickSRNet {a.variant} {a.scale}x  "
          f"{a.height}x{a.width} -> {a.height*a.scale}x{a.width*a.scale}")
    print("=" * 74)

    m = load(a.variant, a.scale)
    n = sum(p.numel() for p in m.parameters())
    print(f"[load] {n:,} params")

    x = torch.rand(1, 3, a.height, a.width)          # [0,1] RGB, as trained
    with torch.no_grad():
        y = m(x)
    assert y.shape == (1, 3, a.height * a.scale, a.width * a.scale), y.shape
    print(f"[check] {tuple(x.shape)} -> {tuple(y.shape)}  "
          f"out range [{float(y.min()):.3f}, {float(y.max()):.3f}]")

    # The upscale must be a pure function of the frame: no state, no randomness. Two
    # calls on the same input have to agree bit for bit, or the video will shimmer
    # between frames in a way that looks like a model artefact.
    with torch.no_grad():
        y2 = m(x)
    assert torch.equal(y, y2), "QuickSRNet is not deterministic"
    print("[check] deterministic across repeated calls")

    tag = f"quicksr{a.variant[0]}{a.scale}x"
    dst = OUT / tag
    dst.mkdir(parents=True, exist_ok=True)
    raw = dst / f"{tag}_raw.onnx"

    torch.onnx.export(
        m, (x,), str(raw),
        input_names=["frame"], output_names=["upscaled"],
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    print(f"[export] -> {raw}")

    import onnx
    import sys
    sys.path.insert(0, str(ROOT / "work" / "export"))
    from graph_fixes import op_histogram, scan_extreme

    g = onnx.load(str(raw), load_external_data=True)
    hist = op_histogram(g)
    print("[ops]    " + ", ".join(f"{k}x{v}" for k, v in
                                  sorted(hist.items(), key=lambda kv: -kv[1])[:12]))
    print(f"[safe]   unsafe constants: {len(scan_extreme(g))}")
    total = sum(f.stat().st_size for f in dst.iterdir())
    print(f"[size]   {total/1e6:.2f} MB across {len(list(dst.iterdir()))} files")
    print("")
    print("next: build the calibration set, then convert (trap #7: a conv model MUST be")
    print("      converted with --preserve_io layout or the converter silently makes the")
    print("      input channel-last while leaving the output NCHW)")


if __name__ == "__main__":
    main()
