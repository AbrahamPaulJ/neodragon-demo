"""ContextAdapter -> ONNX + calibration + test set, in one pass.

The ContextAdapter is the module the phased plan never listed -- the same kind of gap
Phase 4b (the VAE encoder) turned out to be. It sits between DistilT5 and the MMDiT on
the video path: `generation_utils.py:501` does

    prompt_embeds = context_adapter(prompt_embeds)      # [1,128,4096] -> [1,128,1536]

and 1536 is exactly the MMDiT's `encoder_hidden_states` width (CAPTION_DIM in
export_mmdit_stage.py, which is NOT config.joint_attention_dim).

Structurally it is the easiest module in the pipeline: a SkipMLP, 4096 -> 4096 x4 with
concat skips -> 1536. No attention, no convolution, so none of traps #7/#11/#17 apply.
The one thing to watch is the skip `torch.cat([x, y], dim=-1)` -- concat on the LAST
axis of a rank-3 tensor. That is not the multi-input row concat that broke MMDiT stage 1
(this is 2 inputs on the innermost axis, and the token count is 128, far below the
~320-row threshold), but it is worth checking on device rather than assuming.

It runs ONCE per generation, not per AR unit, so its latency barely matters; it is
converted for completeness of the on-device pipeline.

  usage: py -3.10 work/export/export_context_adapter.py --prompts 300 --export
"""

import re
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
ROOT = HERE.parents[1]
REPO = ROOT / "src" / "neodragon"
MODEL = ROOT / "work" / "models" / "neodragon"
OUT = ROOT / "work" / "onnx"
CAL = ROOT / "work" / "calib"
DEV = ROOT / "work" / "device"
sys.path.insert(0, str(REPO))
_pkg = types.ModuleType("neodragon")
_pkg.__path__ = [str(REPO / "neodragon")]
sys.modules["neodragon"] = _pkg

DEVICE_ROOT = "/data/local/tmp/nd"
TAG = "ctxadapt"


class ClampGELU(nn.Module):
    """GELU with its input clamped below -- exact in fp32, far cheaper to quantise.

    MEASURED on device, with `--debug` on the 23-node graph: every GELU lost 13-33 dB.
    Comparing the device's GELU against an exact GELU of *the device's own input* isolates
    the op:

        layer 0: 19.59 dB     layer 3: 47.98 dB

    so layer 3's op is fine and its loss was inherited, while layer 0's is real. The error
    breakdown says why -- for layer 0, **89% of elements (469,489 of 524,288) sit below
    -8, where GELU is exactly 0, and the device returns ~3.6e-3**. GELU's tail is computed
    as a product with the input, so the absolute error scales with |x| and the huge
    negative outliers dominate. Layer 3 has only 11% of its mass there.

    Clamping the input is not an approximation: GELU(-6) underflows to exactly 0 in fp32,
    so the whole model is bit-identical (verified: SNR inf, max|diff| 0.000e+00 at -6, -8
    and -10; -5 gives 69 dB and -4 breaks it at 29.9 dB). It shrinks layer 0's GELU input
    range from [-31.95, 4.88] to [-6, 4.88] and layer 3's from [-79.93, 20.88] to
    [-6, 20.88], which shrinks both the op's error and the activation encoding step.
    """

    LO = -6.0

    def forward(self, z):
        return nn.functional.gelu(z.clamp(min=self.LO))


class Rank2CA(nn.Module):
    """The ContextAdapter with its token axis flattened away.

    MEASURED: exported from a rank-3 input, every `nn.Linear` lowers to **MatMul**, and
    QNN quantises a MatMul's second operand as a per-TENSOR 8-bit weight -- so
    `--use_per_channel_quantization` never applies. On this model's weights that is worth
    8-13 dB per layer:

        linear          max|w|   perTensor  perChannel
        layers.0.1       2.469     22.81      36.19
        layers.3.1      11.688     18.43      33.49
        layers.4.0       5.406     14.83      22.98

    and cascaded over five layers it put the device at **2.62 dB**.

    At rank 2 the same Linears lower to Gemm / FullyConnected, which the converter does
    quantise per output channel. Exactly the effect the MMDiT rank-2 rewrite had there
    (MatMul 268 -> 54, Gemm 428). The graph's input and output keep their rank-3 shapes;
    the reshapes are free.
    """

    def __init__(self, ca, clamp_gelu=True):
        super().__init__()
        if clamp_gelu:
            for m in ca.modules():
                for cn, cm in list(m.named_children()):
                    if isinstance(cm, nn.GELU):
                        setattr(m, cn, ClampGELU())
        self.ca = ca
        self.out_dims = ca.config.output_dims

    def forward(self, x):
        return self.ca(x.reshape(-1, x.shape[-1])).reshape(1, -1, self.out_dims)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=int, default=300)
    ap.add_argument("--test", type=int, default=16)
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse the captured prompt_embeds instead of re-running DistilT5")
    ap.add_argument("--no-clamp-gelu", action="store_true")
    a = ap.parse_args()

    from neodragon.context_adapter import ContextAdapter
    from neodragon.text_encoder_bundle import TextEncoderBundle

    print("[load] context_adapter ...")
    ca = ContextAdapter.from_pretrained(str(MODEL / "context_adapter"),
                                        torch_dtype=torch.float32).eval()
    n_par = sum(p.numel() for p in ca.parameters())
    print("       {:.1f} M params  {} -> {}".format(
        n_par / 1e6, ca.config.input_dims, ca.config.output_dims))

    total = a.prompts + a.test
    cache = CAL / TAG / "_embeds.npy"
    if a.reuse and cache.exists():
        arr = np.load(cache)
        assert arr.shape[0] >= total, (arr.shape, total)
        embeds = [torch.from_numpy(arr[i:i + 1]) for i in range(total)]
        print("[reuse] {} cached prompt_embeds from {}".format(total, cache))
    else:
        print("[load] text_encoder_bundle (DistilT5) ...")
        teb = TextEncoderBundle.from_pretrained(str(MODEL), torch_dtype=torch.float32)
        lines = [ln.strip() for ln in
                 (REPO / "prompts" / "vbench_prompts.txt").read_text(
                     encoding="utf-8").splitlines() if ln.strip()]
        print("[prompts] {} available, using {} calib + {} test".format(
            len(lines), a.prompts, a.test))
        embeds = []
        with torch.no_grad():
            for i in range(total):
                pe, _mask, _pooled = teb(lines[i % len(lines)], torch.device("cpu"))
                embeds.append(pe.float())
                if (i + 1) % 50 == 0:
                    print("  {}/{}".format(i + 1, total))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, torch.cat(embeds, 0).numpy())
        print("[cache] wrote {}".format(cache))
    x0 = embeds[0]
    print("[shape] prompt_embeds {}".format(tuple(x0.shape)))
    assert x0.shape[0] == 1, "batch must be 1 on the AR t2v path (no CFG)"

    # capture the stock output BEFORE building Rank2CA -- it swaps the GELUs in place,
    # so comparing against `ca` afterwards would compare the clamped model with itself.
    with torch.no_grad():
        exact = ca(x0).clone()
    net = Rank2CA(ca, clamp_gelu=not a.no_clamp_gelu).eval()
    with torch.no_grad():
        ref0 = net(x0)
    md = float((ref0 - exact).abs().max())
    print("[check] rank-2 + clamped GELU vs STOCK: max|diff| = {:.3e}".format(md))
    assert md == 0.0, "the rewrite must be bit-exact against the stock model"
    print("[shape] context_adapter out {}".format(tuple(ref0.shape)))
    assert ref0.shape[-1] == 1536, ref0.shape

    if not a.export:
        print("")
        print("pass --export to write the ONNX, calibration set and test set")
        return

    # ---- ONNX ------------------------------------------------------------
    dst = OUT / TAG
    dst.mkdir(parents=True, exist_ok=True)
    onnx_path = dst / (TAG + "_raw.onnx")
    torch.onnx.export(net, (x0,), str(onnx_path),
                      input_names=["prompt_embeds"], output_names=["context"],
                      opset_version=17, do_constant_folding=True, dynamo=False)
    import onnx
    sys.path.insert(0, str(HERE))
    from graph_fixes import op_histogram, scan_extreme
    g = onnx.load(str(onnx_path), load_external_data=True)
    print("[ops]    " + ", ".join("{}x{}".format(k, v) for k, v in
                                  sorted(op_histogram(g).items(), key=lambda kv: -kv[1])))
    print("[safe]   unsafe constants: {}".format(len(scan_extreme(g))))
    print("[onnx]   {:.0f} MB -> {}".format(
        onnx_path.stat().st_size / 1e6, onnx_path))

    # ---- calibration -----------------------------------------------------
    cal = CAL / TAG
    cal.mkdir(parents=True, exist_ok=True)
    for f in cal.glob("*.raw"):
        f.unlink()
    host = re.sub(r"^([A-Za-z]):", lambda m: "/mnt/" + m.group(1).lower(), str(cal).replace("\\", "/"))
    rows = []
    for i in range(a.prompts):
        p = cal / "prompt_embeds_{:04d}.raw".format(i)
        np.ascontiguousarray(embeds[i].numpy(), dtype=np.float32).tofile(p)
        rows.append("prompt_embeds:={}/{}".format(host, p.name))
    (cal / "calib_list_host.txt").write_text("\n".join(rows) + "\n", newline="\n")
    (cal / "dims.txt").write_text(
        "--input_dim prompt_embeds {}\n".format(
            ",".join(str(d) for d in x0.shape)), newline="\n")
    print("[calib]  {} samples -> {}".format(a.prompts, cal))

    # ---- held-out test set + fp32 reference ------------------------------
    io = DEV / ("io_" + TAG)
    io.mkdir(parents=True, exist_ok=True)
    for f in io.glob("*.raw"):
        f.unlink()
    rows = []
    for j in range(a.test):
        x = embeds[a.prompts + j]
        np.ascontiguousarray(x.numpy(), dtype=np.float32).tofile(
            io / "prompt_embeds_{:04d}.raw".format(j))
        with torch.no_grad():
            r = net(x)
        np.ascontiguousarray(r.numpy(), dtype=np.float32).tofile(
            io / "cref_{:04d}.raw".format(j))
        rows.append("prompt_embeds:={}/io_{}/prompt_embeds_{:04d}.raw".format(
            DEVICE_ROOT, TAG, j))
    (io / "input_list.txt").write_text("\n".join(rows) + "\n", newline="\n")
    print("[test]   {} held-out samples + fp32 refs -> {}".format(a.test, io))


if __name__ == "__main__":
    main()
