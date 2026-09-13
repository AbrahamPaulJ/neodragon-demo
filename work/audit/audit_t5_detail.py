"""Trap #3 follow-up.

Three questions the headline audit left open:
  1. Do the >fp16 peaks sit on padding, or on real tokens?
  2. Does fp16 fail loudly (inf/nan) or silently (wrong numbers)?
  3. Does residual scaling actually rescue it, and is it exact?

On (3): T5LayerNorm is scale-invariant (it is RMS-style, no mean subtraction,
no bias), so holding the residual stream S times smaller and dividing each
branch contribution by S is mathematically identical. The encoder ends in a
layer norm, so the output needs no rescaling at all. Concretely: every
`dropout` at a residual add (identity in eval) is replaced by a 1/S scale.
"""

import os
import sys
import types
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "src" / "neodragon"
MODEL = ROOT / "work" / "models" / "neodragon"
sys.path.insert(0, str(REPO))
_pkg = types.ModuleType("neodragon")
_pkg.__path__ = [str(REPO / "neodragon")]
sys.modules["neodragon"] = _pkg

from neodragon.distil_t5 import T5EncoderWithProjection  # noqa: E402
from transformers import T5Tokenizer  # noqa: E402

FP16_MAX = float(np.finfo(np.float16).max)
MAX_LEN = 128


class Scale(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.s = s

    def forward(self, x):
        return x / self.s


def load(dtype=torch.float32):
    return T5EncoderWithProjection.from_pretrained(
        MODEL / "text_encoder_3", torch_dtype=dtype).eval()


def apply_residual_scaling(enc, s):
    """Replace the residual-add dropouts with a 1/S scale."""
    n = 0
    for block in enc.encoder.encoder.block:
        for sublayer in block.layer:
            sublayer.dropout = Scale(s)
            n += 1
    return n


def peaks(enc, kw):
    got = {}

    def mk(i):
        def f(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            got[i] = max(got.get(i, 0.0), t.abs().max().item())
        return f

    hs = [b.register_forward_hook(mk(i)) for i, b in enumerate(enc.encoder.encoder.block)]
    with torch.no_grad():
        out = enc(**kw)[0]
    for h in hs:
        h.remove()
    return max(got.values()), out


def main():
    tok = T5Tokenizer.from_pretrained(MODEL / "tokenizer_3")
    texts = [ln.strip() for ln in
             (REPO / "prompts" / "showcase_prompts.txt").read_text(encoding="utf-8").splitlines()
             if ln.strip()][:8]
    b = tok(texts, padding="max_length", max_length=MAX_LEN, truncation=True,
            add_special_tokens=True, return_tensors="pt")
    ids, am = b.input_ids, b.attention_mask
    kw = {"input_ids": ids, "attention_mask": am}

    print("=" * 84)
    print("TRAP #3 detail -- padding vs real tokens; silent failure; the fix")
    print("=" * 84)
    print(f"\nreal tokens per prompt: {am.sum(1).tolist()} of {MAX_LEN}")

    # ---- Q1: where do the peaks live? --------------------------------------
    enc = load()
    caught = {}

    def hook(i):
        def f(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            caught[i] = t.detach()
        return f

    hs = [bl.register_forward_hook(hook(i)) for i, bl in enumerate(enc.encoder.encoder.block)]
    with torch.no_grad():
        out32 = enc(**kw)[0]
    for h in hs:
        h.remove()

    m = am.bool()
    print(f"\n{'block':>6}  {'max real tok':>14}  {'max padding':>14}  {'overflows':>12}")
    print("-" * 54)
    for i in sorted(caught):
        t = caught[i]
        real = t[m].abs().max().item()
        pad = t[~m].abs().max().item()
        which = [w for w, v in (("real", real), ("pad", pad)) if v > FP16_MAX]
        print(f"{i:>6}  {real:>14.1f}  {pad:>14.1f}  {','.join(which) or '-':>12}")

    # ---- Q2: loud or silent? ------------------------------------------------
    with torch.no_grad():
        out16 = load(torch.float16)(**kw)[0].float()
    cos = torch.nn.functional.cosine_similarity(out32.flatten(), out16.flatten(), dim=0).item()
    print("\n[fp16 end-to-end]")
    print(f"  inf={torch.isinf(out16).sum().item()}  nan={torch.isnan(out16).sum().item()}"
          f"  of {out16.numel()} elements")
    print(f"  cosine(fp16, fp32) = {cos:.4f}")
    print(f"  max|diff|={(out32 - out16).abs().max().item():.4f}  "
          f"signal max={out32.abs().max().item():.4f}")
    verdict = "SILENT corruption" if (torch.isfinite(out16).all() and cos < 0.99) else \
              ("clean" if cos > 0.99 else "loud failure")
    print(f"  verdict: {verdict}")

    # ---- Q3: does residual scaling fix it, exactly? -------------------------
    print("\n[residual scaling] replace each residual-add dropout with 1/S")
    print(f"{'S':>6}  {'peak residual':>15}  {'fits fp16':>10}  {'cos vs fp32 ref':>16}")
    print("-" * 56)
    for S in (1.0, 8.0, 16.0, 32.0, 64.0):
        e = load()
        if S != 1.0:
            apply_residual_scaling(e, S)
        pk, out = peaks(e, kw)
        c = torch.nn.functional.cosine_similarity(out32.flatten(), out.flatten(), dim=0).item()
        print(f"{S:>6.0f}  {pk:>15.1f}  {'yes' if pk < FP16_MAX else 'NO':>10}  {c:>16.6f}")

    # ---- and in actual fp16 -------------------------------------------------
    print("\n[residual scaling, executed in fp16]")
    print(f"{'S':>6}  {'cos vs fp32 ref':>16}  {'max|diff|':>12}")
    print("-" * 40)
    for S in (1.0, 16.0, 32.0, 64.0):
        e = load(torch.float16)
        if S != 1.0:
            apply_residual_scaling(e, S)
        with torch.no_grad():
            o = e(**kw)[0].float()
        c = torch.nn.functional.cosine_similarity(out32.flatten(), o.flatten(), dim=0).item()
        print(f"{S:>6.0f}  {c:>16.6f}  {(out32 - o).abs().max().item():>12.5f}")


if __name__ == "__main__":
    main()
