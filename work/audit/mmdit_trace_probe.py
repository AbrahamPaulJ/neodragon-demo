"""Phase 5 probe: does the stock PyramidMMDiT forward actually run and trace at num_stages=1?

CLAUDE.md says "PyramidMMDiT.forward is not exportable as written". That claim was made
from reading the module. This tests it: build the real thing, feed it exactly what
generation_utils.py:328-334 feeds it, run it, then try torch.onnx.export on it unmodified
and record precisely what fails.

Knowing *which* op refuses to trace is what decides how much of the rewrite is real work
versus mechanical.

  usage: py -3.10 work/audit/mmdit_trace_probe.py [--stage 0] [--unit 6] [--export]
"""

import argparse
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "src" / "neodragon"
LOCAL = ROOT / "work" / "models" / "neodragon"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT / "work" / "audit"))
_pkg = types.ModuleType("neodragon")
_pkg.__path__ = [str(REPO / "neodragon")]
sys.modules["neodragon"] = _pkg

from mmdit_shapes import past_condition_shapes, pyramid_hw, tokens  # noqa: E402

TEXT_TOKENS = 128
# The DiT takes the CONTEXT ADAPTER's output, not the raw T5 embedding:
# generation_utils.py does prompt_embeds = context_adapter(prompt_embeds) first, and
# context_adapter/config.json maps 4096 -> 1536. config.joint_attention_dim=4096 is
# unused residue on this path -- the blocks LayerNorm at caption_projection_dim.
JOINT_DIM = 1536          # config.caption_projection_dim
POOLED_DIM = 2048         # config.pooled_projection_dim
LAT_C = 16


def build_sample(unit, stage, dtype):
    """Exactly `past_conditions[stage] + [latent_model_input]`, as a list of 5-D latents."""
    shapes = past_condition_shapes(unit, stage)
    h, w = pyramid_hw(stage)
    shapes = shapes + [(1, h, w)]
    return [torch.randn(1, LAT_C, t, hh, ww, dtype=dtype) for (t, hh, ww) in shapes], shapes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--unit", type=int, default=6)
    ap.add_argument("--export", action="store_true")
    args = ap.parse_args()

    from neodragon.pyramid_mmdit import PyramidMMDiT

    dtype = torch.float32
    print("=" * 78)
    print("PHASE 5 PROBE -- stock PyramidMMDiT, stage {} unit {}".format(args.stage, args.unit))
    print("=" * 78)

    print("")
    print("[load] diffusion_transformer_320p (fp32, CPU) ...")
    dit = PyramidMMDiT.from_pretrained(str(LOCAL / "diffusion_transformer_320p"),
                                       torch_dtype=dtype).eval()
    n = sum(p.numel() for p in dit.parameters())
    print("       {:.1f} M params, {} blocks".format(n / 1e6, len(dit.transformer_blocks)))

    sample, shapes = build_sample(args.unit, args.stage, dtype)
    n_tok = sum(tokens(s) for s in shapes)
    print("")
    print("[input] {} latents {}  -> {} image tokens + {} text = {}".format(
        len(shapes), " ".join("{}x{}x{}".format(*s) for s in shapes),
        n_tok, TEXT_TOKENS, n_tok + TEXT_TOKENS))

    ehs = torch.randn(1, TEXT_TOKENS, JOINT_DIM, dtype=dtype)
    eam = torch.ones(1, TEXT_TOKENS, dtype=dtype)
    pooled = torch.randn(1, POOLED_DIM, dtype=dtype)
    tstep = torch.tensor([0.5], dtype=dtype)

    # ---- does it run? -------------------------------------------------------
    with torch.no_grad():
        out = dit(sample=[sample], encoder_hidden_states=ehs,
                  encoder_attention_mask=eam, pooled_projections=pooled,
                  timestep_ratio=tstep)
    assert isinstance(out, list) and len(out) == 1, type(out)
    h, w = pyramid_hw(args.stage)
    print("[run]   OK -- output list of {}, tensor {}  (expected (1, {}, 1, {}, {}))".format(
        len(out), tuple(out[0].shape), LAT_C, h, w))
    assert tuple(out[0].shape) == (1, LAT_C, 1, h, w), out[0].shape
    print("[run]   num_stages == 1 confirmed: the stage loop ran once, output is the "
          "current latent only")

    if not args.export:
        print("")
        print("pass --export to attempt torch.onnx.export on the unmodified module")
        return

    # ---- does it trace? -----------------------------------------------------
    class Wrap(torch.nn.Module):
        """torch.onnx.export cannot take List[List[Tensor]] kwargs; flatten to positionals."""

        def __init__(self, dit, n_lat):
            super().__init__()
            self.dit = dit
            self.n_lat = n_lat

        def forward(self, ehs, eam, pooled, tstep, *lats):
            return self.dit(sample=[list(lats)], encoder_hidden_states=ehs,
                            encoder_attention_mask=eam, pooled_projections=pooled,
                            timestep_ratio=tstep)[0]

    wrap = Wrap(dit, len(sample)).eval()
    out_dir = ROOT / "work" / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "_mmdit_probe_s{}u{}.onnx".format(args.stage, args.unit)
    args_t = (ehs, eam, pooled, tstep, *sample)
    print("")
    print("[export] attempting torch.onnx.export (opset 17, TorchScript path) ...")
    try:
        torch.onnx.export(
            wrap, args_t, str(dst),
            input_names=["encoder_hidden_states", "encoder_attention_mask",
                         "pooled_projections", "timestep_ratio"]
            + ["latent_{}".format(i) for i in range(len(sample))],
            output_names=["noise_pred"],
            opset_version=17, do_constant_folding=True, dynamo=False,
        )
        print("[export] SUCCEEDED -- {:.1f} MB".format(dst.stat().st_size / 1024 ** 2))
        import onnx
        import collections
        g = onnx.load(str(dst))
        h = collections.Counter(n.op_type for n in g.graph.node)
        print("[ops]    " + ", ".join("{}x{}".format(k, v)
                                      for k, v in h.most_common(18)))
    except Exception as e:                                   # noqa: BLE001
        print("[export] FAILED: {}".format(type(e).__name__))
        msg = str(e)
        print("         " + msg[:1500].replace("\n", "\n         "))


if __name__ == "__main__":
    main()
