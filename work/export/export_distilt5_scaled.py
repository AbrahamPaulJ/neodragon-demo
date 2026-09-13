"""Phase 1a v2 -- DistilT5 -> ONNX with residual scaling applied.

Why: the float build saturated at fp16 range on device (+-65408, 74% NaN).
HTP executes float graphs in fp16, and DistilT5's residual stream peaks at
329,509 = 5.03x fp16 max. PyTorch hid this because T5LayerNorm computes its
variance in fp32; the device does not.

The transform: T5LayerNorm is RMS-style (no mean subtraction, no bias) and
therefore scale-invariant, so holding the residual stream S times smaller --
dividing the input embedding AND every branch contribution by S -- is exactly
equivalent. The encoder ends in a layer norm, so the output needs no
compensation. Verified rel_err 2.6e-06 at S=8 in work/audit/audit_t5_final.py.
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


class MulConst(nn.Module):
    """Multiply by 1/S. Multiplication, not division -- cheaper on HTP."""

    def __init__(self, inv_s):
        super().__init__()
        self.inv_s = inv_s

    def forward(self, x):
        return x * self.inv_s


class ScaledEmbedding(nn.Module):
    def __init__(self, emb, inv_s):
        super().__init__()
        self.emb, self.inv_s = emb, inv_s

    def forward(self, x):
        return self.emb(x) * self.inv_s


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


def scale_residual_stream(enc, s):
    inv = 1.0 / s
    stack = enc.encoder.encoder
    stack.embed_tokens = ScaledEmbedding(stack.embed_tokens, inv)
    for block in stack.block:
        for sublayer in block.layer:
            sublayer.dropout = MulConst(inv)


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
    print(f"PHASE 1a v2 -- DistilT5 -> ONNX with residual scaling S={S:g}")
    print("=" * 78)

    ref_enc = load_enc()
    with torch.no_grad():
        ref = Wrapper(ref_enc)(ids, am)
    pk0 = peak_residual(ref_enc, ids_all, am_all)
    print(f"\n[before] peak residual {pk0:,.0f} = {pk0/FP16_MAX:.2f}x fp16 max")

    enc = load_enc()
    scale_residual_stream(enc, S)
    pk1 = peak_residual(enc, ids_all, am_all)
    with torch.no_grad():
        got = Wrapper(enc)(ids, am)
    rel = ((got - ref).norm() / ref.norm()).item()
    print(f"[after ] peak residual {pk1:,.0f} = {pk1/FP16_MAX:.2f}x fp16 max  "
          f"({'FITS' if pk1 < FP16_MAX else 'STILL OVERFLOWS'})")
    print(f"[after ] headroom {FP16_MAX/pk1:.1f}x")
    print(f"[exact?] rel_err vs unscaled fp32 reference = {rel:.2e}")

    # ---- export -------------------------------------------------------------
    raw = OUT / "distilt5_scaled_raw.onnx"
    OUT.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        Wrapper(enc), (ids, am), str(raw),
        input_names=["input_ids", "attention_mask"],
        output_names=["prompt_embeds"],
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    print(f"\n[export] {raw.name}")

    # ---- mask constant ------------------------------------------------------
    m = load(raw)
    n = retarget_extreme_constants(m, MASK_FILL)
    fixed = save(m, OUT / "distilt5_scaled_maskfix.onnx")
    print(f"[fix]    rewrote {n} extreme constant(s); remaining {len(scan_extreme(load(fixed)))}")

    # ---- fold ---------------------------------------------------------------
    import onnxruntime as ort
    dst = OUT / "distilt5_scaled_qnn.onnx"
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = str(dst)
    ort.InferenceSession(str(fixed), so, providers=["CPUExecutionProvider"])

    # strip stray opset imports the optimiser adds
    mm = load(dst)
    doms = {nd.domain for nd in mm.graph.node}
    keep = [o for o in mm.opset_import if (o.domain or "") == ""]
    del mm.opset_import[:]
    mm.opset_import.extend(keep)
    save(mm, dst)
    print(f"[fold]   nodes {len(load(fixed).graph.node)} -> {len(mm.graph.node)}   "
          f"domains={doms}")

    # ---- verify -------------------------------------------------------------
    so2 = ort.SessionOptions()
    so2.log_severity_level = 3
    sess = ort.InferenceSession(str(dst), so2, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input_ids": ids.numpy(), "attention_mask": am.numpy()})[0]
    out = torch.from_numpy(out)
    print(f"[verify] onnx vs torch reference: rel_err="
          f"{((out - ref).norm()/ref.norm()).item():.2e}  "
          f"max|diff|={(out - ref).abs().max().item():.3e}")
    print(f"[verify] unsafe constants: {len(scan_extreme(load(dst)))}")
    print(f"\nwrote {dst}  ({dst.stat().st_size/1024**2:.1f} MB)")


if __name__ == "__main__":
    main()
