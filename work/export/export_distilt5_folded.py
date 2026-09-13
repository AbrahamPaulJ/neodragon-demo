"""Phase 1a v3 -- residual scaling folded into weights, no runtime multiplies.

v2 implemented residual scaling as an explicit `x * (1/S)` at each residual add.
On-device profiling showed those two multiplies cost 4.77 ms of an 18.66 ms
inference -- 25.6% of total runtime, for arithmetic that can be folded into
weights at build time.

Each branch ends in a linear layer with no bias:
  * T5LayerSelfAttention -> T5Attention.o
  * T5LayerFF            -> T5DenseGatedActDense.wo
so scaling those weight matrices by 1/S scales the branch output by 1/S
identically. The input embedding is scaled the same way. T5LayerNorm is
scale-invariant, and the encoder ends in a layer norm, so the output is
unchanged and no compensation is needed.

Net effect: same numerics as v2, minus two elementwise multiplies per block.
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

from graph_fixes import load, op_histogram, retarget_extreme_constants, save, scan_extreme  # noqa: E402
from neodragon.distil_t5 import T5EncoderWithProjection  # noqa: E402
from transformers import T5Tokenizer  # noqa: E402

SEQ = 128
MASK_FILL = -100.0
FP16_MAX = float(np.finfo(np.float16).max)


class Wrapper(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, input_ids, attention_mask):
        return self.enc(input_ids=input_ids, attention_mask=attention_mask,
                        return_dict=False)[0]


def load_enc():
    return T5EncoderWithProjection.from_pretrained(
        MODEL / "text_encoder_3", torch_dtype=torch.float32,
        attn_implementation="eager").eval()


@torch.no_grad()
def fold_residual_scaling(enc, s):
    """Divide the residual stream by S with zero runtime cost.

    T5LayerNorm is scale-invariant only while `variance >> variance_epsilon`.
    Scaling the stream by 1/S scales the variance by 1/S^2, so the epsilon must
    be scaled the same way or it stops being negligible -- at S=4096 with the
    stock 1e-6 the error jumps from 1.2e-06 to 5.6e-02.
    """
    inv = 1.0 / s
    stack = enc.encoder.encoder
    touched = []

    # layer-norm epsilons, so scale-invariance keeps holding
    n_eps = 0
    for mod in enc.modules():
        if hasattr(mod, "variance_epsilon"):
            mod.variance_epsilon = mod.variance_epsilon * inv * inv
            n_eps += 1
    touched.append(f"variance_epsilon x{n_eps} (scaled by 1/S^2)")

    # input embedding
    stack.embed_tokens.weight.mul_(inv)
    touched.append("embed_tokens.weight")

    for i, block in enumerate(stack.block):
        # self-attention branch ends in `o` (bias=False in T5)
        o = block.layer[0].SelfAttention.o
        assert o.bias is None, "T5 attention output projection unexpectedly has a bias"
        o.weight.mul_(inv)
        # feed-forward branch ends in `wo`
        wo = block.layer[1].DenseReluDense.wo
        assert wo.bias is None, "T5 FFN output projection unexpectedly has a bias"
        wo.weight.mul_(inv)
        if i == 0:
            touched += ["block.*.layer.0.SelfAttention.o.weight",
                        "block.*.layer.1.DenseReluDense.wo.weight"]
    return touched


def peak_residual(enc, ids, am):
    got = {}

    def mk(i):
        def f(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            got[i] = max(got.get(i, 0.0), t.abs().max().item())
        return f

    hs = [b.register_forward_hook(mk(i)) for i, b in enumerate(enc.encoder.encoder.block)]
    with torch.no_grad():
        enc(input_ids=ids, attention_mask=am)
    for h in hs:
        h.remove()
    return max(got.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=16.0)
    args = ap.parse_args()
    S = args.scale

    tok = T5Tokenizer.from_pretrained(MODEL / "tokenizer_3")
    texts = [ln.strip() for ln in
             (REPO / "prompts" / "showcase_prompts.txt").read_text(encoding="utf-8").splitlines()
             if ln.strip()][:16]
    b = tok(texts, padding="max_length", max_length=SEQ, truncation=True,
            add_special_tokens=True, return_tensors="pt")
    ids_all, am_all = b.input_ids, b.attention_mask
    ids, am = ids_all[:1], am_all[:1]

    print("=" * 78)
    print(f"PHASE 1a v3 -- residual scaling S={S:g} FOLDED INTO WEIGHTS")
    print("=" * 78)

    ref_enc = load_enc()
    with torch.no_grad():
        ref = Wrapper(ref_enc)(ids, am)
    pk0 = peak_residual(ref_enc, ids_all, am_all)
    print(f"\n[before] peak residual {pk0:,.0f} = {pk0/FP16_MAX:.2f}x fp16 max")

    enc = load_enc()
    touched = fold_residual_scaling(enc, S)
    print("[fold]   scaled weights:")
    for t in touched:
        print(f"           {t}")
    pk1 = peak_residual(enc, ids_all, am_all)
    with torch.no_grad():
        got = Wrapper(enc)(ids, am)
    rel = ((got - ref).norm() / ref.norm()).item()
    print(f"[after ] peak residual {pk1:,.0f} = {pk1/FP16_MAX:.2f}x fp16 max  "
          f"({'FITS' if pk1 < FP16_MAX else 'STILL OVERFLOWS'}), headroom {FP16_MAX/pk1:.1f}x")
    print(f"[exact?] rel_err vs unscaled fp32 reference = {rel:.2e}")

    # ---- export -------------------------------------------------------------
    raw = OUT / "distilt5_folded_raw.onnx"
    torch.onnx.export(
        Wrapper(enc), (ids, am), str(raw),
        input_names=["input_ids", "attention_mask"],
        output_names=["prompt_embeds"],
        opset_version=17, do_constant_folding=True, dynamo=False,
    )

    m = load(raw)
    n = retarget_extreme_constants(m, MASK_FILL, verbose=False)
    fixed = save(m, OUT / "distilt5_folded_maskfix.onnx")
    print(f"\n[fix]    rewrote {n} extreme constant(s); "
          f"remaining {len(scan_extreme(load(fixed)))}")

    import onnxruntime as ort
    dst = OUT / "distilt5_folded_qnn.onnx"
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = str(dst)
    ort.InferenceSession(str(fixed), so, providers=["CPUExecutionProvider"])

    mm = load(dst)
    keep = [o for o in mm.opset_import if (o.domain or "") == ""]
    del mm.opset_import[:]
    mm.opset_import.extend(keep)
    save(mm, dst)

    # ---- compare graph against v2 ------------------------------------------
    v2 = OUT / "distilt5_scaled_qnn.onnx"
    h3 = op_histogram(mm)
    print(f"[fold]   nodes: {sum(h3.values())}")
    if v2.exists():
        h2 = op_histogram(load(v2))
        print(f"         v2 (runtime Mul) had {sum(h2.values())} nodes, "
              f"Mul {h2.get('Mul', 0)} -> {h3.get('Mul', 0)}  "
              f"(removed {h2.get('Mul', 0) - h3.get('Mul', 0)})")

    so2 = ort.SessionOptions()
    so2.log_severity_level = 3
    sess = ort.InferenceSession(str(dst), so2, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input_ids": ids.numpy(), "attention_mask": am.numpy()})[0]
    out = torch.from_numpy(out)
    print(f"[verify] onnx vs torch reference: rel_err="
          f"{((out - ref).norm()/ref.norm()).item():.2e}  "
          f"max|diff|={(out - ref).abs().max().item():.3e}")
    print(f"\nwrote {dst}  ({dst.stat().st_size/1024**2:.1f} MB)")


if __name__ == "__main__":
    main()
