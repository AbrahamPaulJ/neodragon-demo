"""Phase 4f -- TAEHV decoder as an explicit-state STREAMING graph.

Fixes the 24x latency gap diagnosed 2026-08-21 (docs/phase4-vae-decoder.md).

The T=1 `parallel=True` graph folds time into the BATCH axis (`TGrow` does
`x.reshape(-1, C, H, W)`). Batch is outermost in both layouts, so that reshape is
a free view in NCHW but a genuine scatter in NHWC -- and HTP runs convolution
channel-last. The converter therefore wraps every TGrow in a transpose pair on
the largest tensors in the graph: 779.8 MB of layout traffic per inference,
against 671.2 MB of actual convolution output.

This version never folds time into batch. It follows the upstream
`parallel=False` streaming order instead, where `TGrow` is a 1x1 conv followed by
a *channel chunk* -- contiguous on the innermost axis in NHWC, so it costs
nothing. Batch stays 1 throughout and the graph is unrolled over the 8 output
frames.

Two independent wins from the one rewrite:

  * no batch-fold reshapes  -> the layout transposes go away
  * 1 invocation per LATENT frame emitting 8 video frames, instead of 1
    invocation per VIDEO frame that computes 8 and keeps 1. A 49-frame decode
    becomes 7 invocations, not 49.

The 9 MemBlock previous-frame states become explicit graph inputs and outputs --
the same "fixed-size explicit state" pattern paper 4.1 describes for the MMDiT's
autoregressive history.

Frame trim: the graph always emits 8 frames. `frames_to_trim = 7` applies to the
whole video, so the CALLER drops the first 7 frames of invocation 1 only. That
reproduces 8*T - 7 exactly (T=7 -> 49 frames, matching Fig 13).

  usage: py -3.10 work/export/export_vae_decoder_stream.py
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
from neodragon.asymmetric_causal_video_vae.decoder import MemBlock, TGrow  # noqa: E402

LAT_C, LAT_H, LAT_W = 16, 40, 64
N_STATES = 9
FRAMES_PER_LATENT = 8


class StreamingTAEHVDecoder(nn.Module):
    """One latent frame in, 8 video frames out, 9 MemBlock states in and out.

    Input  latent   [1, 16, 40, 64]              (4-D: no 5-D op anywhere)
    Input  state_k  [1, C_k, H_k, W_k]           k = 0..8
    Output video    [1, 24, 320, 512]            8 frames packed on channels
    Output nstate_k [1, C_k, H_k, W_k]           k = 0..8
    """

    def __init__(self, vae):
        super().__init__()
        self.blocks = vae.decoder.blocks

        # post_quant_conv is a 1x1x1 CausalConv3d, so time_kernel_size == 1 and
        # every pad in time_causal_padding is 0 -- it is pointwise. Folding it to
        # a Conv2d removes the graph's only 5-D op, and with it the NCDHW <-> NDHWC
        # transposes on the input.
        pqc = vae.post_quant_conv
        assert pqc.time_kernel_size == 1, pqc.time_kernel_size
        assert all(p == 0 for p in pqc.time_causal_padding), pqc.time_causal_padding
        w = pqc.conv.weight
        assert tuple(w.shape[2:]) == (1, 1, 1), w.shape
        self.post_quant_conv = nn.Conv2d(w.shape[1], w.shape[0], 1,
                                         bias=pqc.conv.bias is not None)
        with torch.no_grad():
            self.post_quant_conv.weight.copy_(w.squeeze(2))
            if pqc.conv.bias is not None:
                self.post_quant_conv.bias.copy_(pqc.conv.bias)

    def forward(self, latent, *states):
        assert len(states) == N_STATES, len(states)
        xs = [self.post_quant_conv(latent)]
        new_states = []

        for block in self.blocks:
            if isinstance(block, MemBlock):
                # Streaming order: each element's `past` is the previous element's
                # INPUT at this block; the first one's past is the state carried in
                # from the previous invocation. What leaves is the last input seen.
                past = states[len(new_states)]
                out = []
                for x in xs:
                    out.append(block(x, past))
                    past = x
                new_states.append(past)
                xs = out
            elif isinstance(block, TGrow):
                # Upstream does conv -> reshape(-1, C, H, W) -> view back -> chunk on
                # channels. The two reshapes cancel, so this is conv + channel chunk.
                # Skipping the round trip is what keeps time out of the batch axis.
                out = []
                for x in xs:
                    c = x.shape[1]
                    y = block.conv(x)
                    for s in range(block.stride):
                        out.append(y[:, s * c:(s + 1) * c])
                xs = out
            else:
                xs = [block(x) for x in xs]

        assert len(xs) == FRAMES_PER_LATENT, len(xs)
        assert len(new_states) == N_STATES, len(new_states)
        return torch.cat(xs, dim=1), *new_states


def state_shapes(net):
    """Trace the per-MemBlock state shapes without hardcoding them."""
    shapes = []
    x = torch.zeros(1, LAT_C, LAT_H, LAT_W)
    xs = [net.post_quant_conv(x)]
    for block in net.blocks:
        if isinstance(block, MemBlock):
            shapes.append(tuple(xs[0].shape))
            xs = [block(x, torch.zeros_like(x)) for x in xs]
        elif isinstance(block, TGrow):
            out = []
            for x in xs:
                c = x.shape[1]
                y = block.conv(x)
                out.extend(y[:, s * c:(s + 1) * c] for s in range(block.stride))
            xs = out
        else:
            xs = [block(x) for x in xs]
    return shapes


def reference_video(vae, z):
    """Stock decoder: NCTHW latent -> NCTHW video, parallel=True, trimmed."""
    with torch.no_grad():
        return vae.decoder(vae.post_quant_conv(z))


def stream_video(net, z, shapes):
    """Run the streaming graph once per latent frame and stitch the result.

    Mirrors what the on-device caller must do: zero states, 8 frames per
    invocation, drop the first `frames_to_trim` frames of the whole video.
    """
    states = [torch.zeros(s) for s in shapes]
    frames = []
    with torch.no_grad():
        for t in range(z.shape[2]):
            out = net(z[:, :, t], *states)
            video, states = out[0], list(out[1:])
            # [1, 24, H, W] -> 8 x [1, 3, H, W]
            frames.extend(video[:, i * 3:(i + 1) * 3] for i in range(FRAMES_PER_LATENT))
    return torch.stack(frames, dim=2)  # NCTHW


def main():
    print("=" * 76)
    print("PHASE 4f -- TAEHV streaming decoder (explicit MemBlock state) -> ONNX")
    print("=" * 76)

    vae = AsymmetricCausalVideoVAE.from_pretrained(
        MODEL / "causal_video_vae", torch_dtype=torch.float32).eval()
    net = StreamingTAEHVDecoder(vae).eval()
    trim = vae.decoder.frames_to_trim

    shapes = state_shapes(net)
    print(f"\n[state] {len(shapes)} MemBlock states, frames_to_trim={trim}")
    total = 0
    for i, s in enumerate(shapes):
        n = 1
        for d in s:
            n *= d
        total += n
        print(f"        state_{i}  {str(s):24} {n*2/1e6:6.2f} MB @ fp16")
    print(f"        total state I/O: {total*2/1e6:.1f} MB @ fp16")

    # ---- correctness vs the stock parallel decoder ---------------------------
    T = 3
    torch.manual_seed(0)
    z = torch.randn(1, LAT_C, T, LAT_H, LAT_W)

    ref = reference_video(vae, z)
    got = stream_video(net, z, shapes)[:, :, trim:]
    print(f"\n[check] T={T}: reference {tuple(ref.shape)}  streaming {tuple(got.shape)}")
    assert ref.shape == got.shape, (ref.shape, got.shape)
    d = (got - ref).abs()
    snr = 20 * torch.log10(ref.norm() / (got - ref).norm()).item()
    print(f"[check] max|diff| = {d.max():.3e}   SNR = {snr:.1f} dB")
    assert d.max() < 1e-4, "streaming decoder does not match the reference"
    print("[check] PASS -- streaming order reproduces parallel=True")

    # ---- export -------------------------------------------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "vaedec_stream"
    raw = OUT / f"{tag}_raw.onnx"

    args = (torch.randn(1, LAT_C, LAT_H, LAT_W),
            *[torch.zeros(s) for s in shapes])
    in_names = ["latent"] + [f"state_{i}" for i in range(N_STATES)]
    out_names = ["video"] + [f"nstate_{i}" for i in range(N_STATES)]

    torch.onnx.export(
        net, args, str(raw),
        input_names=in_names, output_names=out_names,
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

    # ---- verify the exported graph ------------------------------------------
    so2 = ort.SessionOptions()
    so2.log_severity_level = 3
    sess = ort.InferenceSession(str(dst), so2, providers=["CPUExecutionProvider"])
    feed = {n: a.numpy() for n, a in zip(in_names, args)}
    onnx_out = sess.run(None, feed)
    with torch.no_grad():
        torch_out = net(*args)
    worst = max((torch.from_numpy(a) - b).abs().max().item()
                for a, b in zip(onnx_out, torch_out))
    ref0, got0 = torch_out[0], torch.from_numpy(onnx_out[0])
    snr = 20 * torch.log10(ref0.norm() / (got0 - ref0).norm()).item()
    print(f"[verify] onnx vs torch: max|diff|={worst:.3e}  video SNR={snr:.1f} dB")

    print(f"\nwrote {dst}  ({dst.stat().st_size/1024**2:.1f} MB)")
    print("\nnext: convert W8A16, then run analyze_net.py on the *_net.json and")
    print("      compare Transpose bytes against the 779.80 MB baseline.")


if __name__ == "__main__":
    main()
