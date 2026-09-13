"""Phase 4b -- CausalVaeEncoder for the E2E first frame, collapsed to pure 2-D.

The encoder was never in the phase plan (found 2026-08-21). It IS on the
critical path: generation_utils.py:531 encodes the SSD1B first frame back into
latent space before the AR loop starts.

The scary part turns out not to apply. `modeling_causal_ops.CausalConv3d` keeps
a stateful `deque` feature cache -- but only on the `temporal_chunk=True`
branch. `vae.encode(image)` at generation_utils.py:531 takes the defaults
(`temporal_chunk=False`), and the tensor it is handed is a SINGLE frame:

    image = rearrange(image, "(b t) h w c -> b c t h w", b=1, t=1)

so the cache is never touched, and T == 1 throughout.

With T == 1 the whole thing collapses. Every CausalConv3d front-pads time by
`time_kernel_size - 1 == 2` with CONSTANT ZEROS (pad_mode is "constant", and the
`self.time_pad < x.shape[2]` guard forces "constant" anyway at T=1), giving
T_pad == 3, then convolves with a time-3 kernel and no time padding -> T == 1
out. Both stride cases land on the same identity:

    out[0] = w[0]*0 + w[1]*0 + w[2]*x_real  ==  w[2]*x_real

i.e. exactly a 2-D convolution with the LAST temporal slice of the weight.
Exact, not approximate: the discarded terms are multiplications by zero.

  * stride (1,1,1), k=3  -> Conv2d(stride=1, padding=1) on w[:, :, -1]
  * stride (1,2,2), k=3  -> Conv2d(stride=2, padding=1) on w[:, :, -1]
  * stride (2,1,1), k=3  -> Conv2d(stride=1, padding=1) on w[:, :, -1]
        (temporal downsample: (3-3)/2+1 == 1 output frame, same last-slice term)
  * k=1                  -> Conv2d 1x1, time_pad == 0, pointwise

`CausalGroupNorm` does `rearrange(x, "b c t h w -> (b t) c h w")`, which is
trap #11's shape -- a time-into-batch fold. At T == 1 it is a squeeze, so
dropping to a plain 2-D GroupNorm removes it outright instead of trusting the
converter to notice. Same for the mid block's attention rearranges.

Net effect: no 5-D tensor anywhere, no rank-5 layout transposes, and 3x fewer
MACs than a naive T=1 export would have spent convolving two planes of zeros.

  usage: py -3.10 work/export/export_vae_encoder_2d.py
"""

import os
import sys
import types
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

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

IMG_C, IMG_H, IMG_W = 3, 320, 512          # run_reference.py:192
LAT_C, LAT_H, LAT_W = 32, 40, 64           # 2*16 moments, /8 spatial


def conv2d_from_causal(cc):
    """CausalConv3d -> the exactly-equivalent Conv2d at T == 1.

    Asserts every assumption instead of trusting the config, because a silent
    mismatch here is a wrong latent that still looks plausible (trap #8).
    """
    w = cc.conv.weight
    kt, kh, kw = w.shape[2:]
    assert cc.pad_mode == "constant", cc.pad_mode
    assert kt in (1, 3), kt
    assert cc.time_pad == kt - 1, (cc.time_pad, kt)

    wpad, wpad2, hpad, hpad2, tpad, tpad2 = cc.time_causal_padding
    assert wpad == wpad2 == kw // 2, cc.time_causal_padding
    assert hpad == hpad2 == kh // 2, cc.time_causal_padding
    assert tpad == kt - 1 and tpad2 == 0, cc.time_causal_padding

    st, sh, sw = cc.conv.stride
    assert st == cc.temporal_stride
    if kt == 3:
        # T_pad == 3, kernel 3, no time padding -> exactly one output frame for
        # BOTH temporal strides, and only the last input plane is non-zero.
        assert st in (1, 2), st
    else:
        assert st == 1 and cc.time_pad == 0, (st, cc.time_pad)

    m = nn.Conv2d(w.shape[1], w.shape[0], (kh, kw), stride=(sh, sw),
                  padding=(hpad, wpad), bias=cc.conv.bias is not None)
    with torch.no_grad():
        m.weight.copy_(w[:, :, -1])          # the only slice that survives
        if cc.conv.bias is not None:
            m.bias.copy_(cc.conv.bias)
    return m


def groupnorm_from_causal(cg):
    """CausalGroupNorm -> nn.GroupNorm. The (b t) fold is a squeeze at T == 1."""
    m = nn.GroupNorm(cg.num_groups, cg.num_channels, eps=cg.eps,
                     affine=cg.affine)
    with torch.no_grad():
        if cg.affine:
            m.weight.copy_(cg.weight)
            m.bias.copy_(cg.bias)
    return m


class Resnet2D(nn.Module):
    """CausalResnetBlock3D at T == 1. dropout is identity in eval."""

    def __init__(self, r):
        super().__init__()
        assert r.time_embedding_norm == "default", r.time_embedding_norm
        self.norm1 = groupnorm_from_causal(r.norm1)
        self.conv1 = conv2d_from_causal(r.conv1)
        self.norm2 = groupnorm_from_causal(r.norm2)
        self.conv2 = conv2d_from_causal(r.conv2)
        self.shortcut = (conv2d_from_causal(r.conv_shortcut)
                         if r.conv_shortcut is not None else None)
        self.scale = float(r.output_scale_factor)
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        if self.shortcut is not None:
            x = self.shortcut(x)
        return (x + h) / self.scale


class Encoder2D(nn.Module):
    """[1, 3, 320, 512] pixels -> [1, 32, 40, 64] moments. 4-D end to end."""

    def __init__(self, vae):
        super().__init__()
        enc = vae.encoder
        self.conv_in = conv2d_from_causal(enc.conv_in)

        downs = []
        for blk in enc.down_blocks:
            sd = (conv2d_from_causal(blk.downsamplers[0].conv)
                  if blk.downsamplers is not None else None)
            td = (conv2d_from_causal(blk.temporal_downsamplers[0].conv)
                  if blk.temporal_downsamplers is not None else None)
            downs.append(nn.ModuleDict({
                "resnets": nn.ModuleList([Resnet2D(r) for r in blk.resnets]),
                "spatial": nn.ModuleList([sd] if sd is not None else []),
                "temporal": nn.ModuleList([td] if td is not None else []),
            }))
        self.down_blocks = nn.ModuleList(downs)

        mid = enc.mid_block
        self.mid_resnets = nn.ModuleList([Resnet2D(r) for r in mid.resnets])
        # diffusers Attention already handles a 4-D input natively; the module's
        # own rearranges only existed to fold t into batch.
        self.mid_attentions = nn.ModuleList(mid.attentions)

        self.conv_norm_out = groupnorm_from_causal(enc.conv_norm_out)
        self.conv_act = nn.SiLU()
        self.conv_out = conv2d_from_causal(enc.conv_out)
        self.quant_conv = conv2d_from_causal(vae.quant_conv)

    def forward(self, image):
        x = self.conv_in(image)
        for blk in self.down_blocks:
            for r in blk["resnets"]:
                x = r(x)
            for d in blk["spatial"]:
                x = d(x)
            for d in blk["temporal"]:
                x = d(x)

        x = self.mid_resnets[0](x)
        for attn, r in zip(self.mid_attentions, self.mid_resnets[1:]):
            if attn is not None:
                x = attn(x)
            x = r(x)

        x = self.conv_out(self.conv_act(self.conv_norm_out(x)))
        return self.quant_conv(x)


def reference_moments(vae, image5):
    with torch.no_grad():
        h = vae.encoder(image5, is_init_image=True, temporal_chunk=False)
        return vae.quant_conv(h, is_init_image=True, temporal_chunk=False)


def main():
    print("=" * 76)
    print("PHASE 4b -- CausalVaeEncoder, T=1 first frame, collapsed to 2-D -> ONNX")
    print("=" * 76)

    vae = AsymmetricCausalVideoVAE.from_pretrained(
        MODEL / "causal_video_vae", torch_dtype=torch.float32).eval()
    assert not vae.use_tiling, "tiling would change the graph shape"
    net = Encoder2D(vae).eval()

    n3d = sum(1 for m in vae.encoder.modules() if isinstance(m, nn.Conv3d))
    n2d = sum(1 for m in net.modules() if isinstance(m, nn.Conv2d))
    print("")
    print("[collapse] {} Conv3d in the encoder -> {} Conv2d (incl. quant_conv), "
          "0 five-D ops".format(n3d, n2d))

    # ---- correctness vs the stock encoder -----------------------------------
    torch.manual_seed(0)
    makers = [
        lambda: torch.rand(1, IMG_C, 1, IMG_H, IMG_W) * 2 - 1,
        lambda: torch.randn(1, IMG_C, 1, IMG_H, IMG_W) * 0.5,
    ]
    for trial, mk in enumerate(makers):
        img5 = mk()
        ref = reference_moments(vae, img5)
        with torch.no_grad():
            got = net(img5[:, :, 0])
        assert tuple(ref.shape) == (1, LAT_C, 1, LAT_H, LAT_W), ref.shape
        ref4 = ref[:, :, 0]
        d = (got - ref4).abs()
        snr = 20 * torch.log10(ref4.norm() / (got - ref4).norm()).item()
        print("[check] trial {}: {}  max|diff|={:.3e}  SNR={:.1f} dB".format(
            trial, tuple(ref4.shape), d.max().item(), snr))
        assert d.max() < 1e-4, "2-D collapse does not match the reference encoder"
    print("[check] PASS -- the T=1 collapse is exact")

    # ---- export -------------------------------------------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "vaeenc2d"
    raw = OUT / (tag + "_raw.onnx")
    args = (torch.rand(1, IMG_C, IMG_H, IMG_W) * 2 - 1,)

    torch.onnx.export(
        net, args, str(raw),
        input_names=["image"], output_names=["moments"],
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    print("")
    print("[export] {}  {:.1f} MB".format(raw.name, raw.stat().st_size / 1024 ** 2))

    # ---- fold ---------------------------------------------------------------
    import onnxruntime as ort
    dst = OUT / (tag + "_qnn.onnx")
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
    print("[fold]   {} -> {} nodes".format(len(load(raw).graph.node), sum(h.values())))
    print("[ops]    " + ", ".join("{}x{}".format(k, v) for k, v in
                                  sorted(h.items(), key=lambda kv: -kv[1])))
    print("[safe]   unsafe constants: {}".format(len(scan_extreme(mm))))

    # ---- verify the exported graph ------------------------------------------
    so2 = ort.SessionOptions()
    so2.log_severity_level = 3
    sess = ort.InferenceSession(str(dst), so2, providers=["CPUExecutionProvider"])
    onnx_out = torch.from_numpy(sess.run(None, {"image": args[0].numpy()})[0])
    with torch.no_grad():
        torch_out = net(*args)
    snr = 20 * torch.log10(
        torch_out.norm() / (onnx_out - torch_out).norm()).item()
    print("[verify] onnx vs torch: max|diff|={:.3e}  SNR={:.1f} dB".format(
        (onnx_out - torch_out).abs().max().item(), snr))

    print("")
    print("wrote {}  ({:.1f} MB)".format(dst, dst.stat().st_size / 1024 ** 2))
    print("")
    print("next: convert W8A16 and run analyze_net.py on the *_net.json.")


if __name__ == "__main__":
    main()
