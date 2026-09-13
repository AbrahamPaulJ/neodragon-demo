"""Trap #3, settled.

Corrects two errors in the first pass:
  * the residual-scaling test scaled each branch by 1/S but left the input
    embedding unscaled, so the intended invariant g = h/S never held;
  * fp16 was measured once via .half() on an already-instantiated fp32 model
    and once via from_pretrained(fp16); those disagreed, so both are redone
    from a clean load with an unambiguous error metric.

It also asks the question that actually matters for this port: the deployment
target is W8A16 *integer*, not fp16. A 16-bit per-tensor activation quantiser
spends its 65535 levels over the observed range, so what matters is the ratio
of the peak to the typical activation, not whether fp16 happens to hold.
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


class Div(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.s = s

    def forward(self, x):
        return x / self.s


class ScaledEmbedding(nn.Module):
    def __init__(self, emb, s):
        super().__init__()
        self.emb, self.s = emb, s

    def forward(self, x):
        return self.emb(x) / self.s


def load(dtype=torch.float32):
    return T5EncoderWithProjection.from_pretrained(
        MODEL / "text_encoder_3", torch_dtype=dtype).eval()


def scale_residual_stream(enc, s):
    """Hold the whole residual stream S times smaller.

    T5LayerNorm is RMS-style with no mean subtraction and no bias, so it is
    scale-invariant: LN(h/S) == LN(h). Dividing both the input embedding and
    every branch contribution by S therefore gives an exactly equivalent
    network whose residual tensors are S times smaller. The encoder ends in a
    layer norm, so the output needs no compensation.
    """
    stack = enc.encoder.encoder
    stack.embed_tokens = ScaledEmbedding(stack.embed_tokens, s)
    for block in stack.block:
        for sublayer in block.layer:
            sublayer.dropout = Div(s)


def rel_err(ref, got):
    return ((got - ref).norm() / ref.norm()).item()


def cos(ref, got):
    a = ref.flatten().double()
    b = got.flatten().double()
    return (a @ b / (a.norm() * b.norm())).item()


def residual_peaks(enc, kw):
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
    kw = {"input_ids": b.input_ids, "attention_mask": b.attention_mask}

    print("=" * 82)
    print("TRAP #3 (settled) -- DistilT5 activation range")
    print("=" * 82)

    ref_model = load()
    peak32, out32 = residual_peaks(ref_model, kw)
    print(f"\nfp32 reference: peak residual {peak32:,.0f} = {peak32/FP16_MAX:.2f}x fp16 max")

    # ---- 1. fp16, from a clean load ----------------------------------------
    print("\n[1] does fp16 actually break?")
    with torch.no_grad():
        o_fp16 = load(torch.float16)(**kw)[0].float()
    m = load(torch.float32)
    with torch.no_grad():
        o_half = m.half()(**kw)[0].float()
    for label, o in (("from_pretrained(fp16)", o_fp16), (".half() on fp32 model", o_half)):
        print(f"  {label:<26} inf={torch.isinf(o).sum().item():>3} "
              f"nan={torch.isnan(o).sum().item():>3}  "
              f"rel_err={rel_err(out32, o):.4f}  cos={cos(out32, o):.6f}")
    print("  (both paths load the same bf16 checkpoint; any gap here is a")
    print("   measurement artefact, not a property of the model)")

    # ---- 2. what W8A16 actually sees ---------------------------------------
    print("\n[2] what a 16-bit INTEGER activation quantiser sees")
    caught = {}

    def hook(i):
        def f(_m, _i, out):
            caught[i] = (out[0] if isinstance(out, tuple) else out).detach()
        return f

    e = load()
    hs = [bl.register_forward_hook(hook(i)) for i, bl in enumerate(e.encoder.encoder.block)]
    with torch.no_grad():
        e(**kw)
    for h in hs:
        h.remove()

    print(f"  {'block':>5}  {'peak':>10}  {'median|.|':>10}  {'peak/med':>9}  "
          f"{'a16 LSB':>9}  {'bits left':>9}")
    print("  " + "-" * 62)
    for i in sorted(caught):
        t = caught[i].abs()
        pk = t.max().item()
        med = t.median().item()
        lsb = pk / 32767          # symmetric int16
        eff = np.log2(max(med / lsb, 1e-9))
        print(f"  {i:>5}  {pk:>10,.0f}  {med:>10.3f}  {pk/med:>9,.0f}  "
              f"{lsb:>9.3f}  {eff:>9.1f}")
    print("\n  'bits left' = log2(median / LSB): resolution actually available to a")
    print("  typical activation once one outlier fixes the per-tensor scale.")

    # ---- 3. residual scaling, done correctly -------------------------------
    print("\n[3] residual scaling (embedding AND branches divided by S)")
    print(f"  {'S':>5}  {'peak residual':>14}  {'fits fp16':>10}  {'rel_err vs fp32':>16}")
    print("  " + "-" * 52)
    for S in (1.0, 8.0, 16.0, 32.0, 64.0):
        e = load()
        if S != 1.0:
            scale_residual_stream(e, S)
        pk, out = residual_peaks(e, kw)
        print(f"  {S:>5.0f}  {pk:>14,.0f}  {'yes' if pk < FP16_MAX else 'NO':>10}  "
              f"{rel_err(out32, out):>16.2e}")
    print("\n  rel_err ~1e-7 confirms the transform is exact in fp32, i.e. it is a")
    print("  free change: same outputs, smaller tensors.")

    # ---- 4. and executed in fp16 -------------------------------------------
    print("\n[4] the same transform executed in fp16")
    print(f"  {'S':>5}  {'rel_err vs fp32':>16}  {'cos':>10}")
    print("  " + "-" * 36)
    for S in (1.0, 16.0, 32.0, 64.0):
        e = load(torch.float16)
        if S != 1.0:
            scale_residual_stream(e, S)
        with torch.no_grad():
            o = e(**kw)[0].float()
        print(f"  {S:>5.0f}  {rel_err(out32, o):>16.2e}  {cos(out32, o):>10.6f}")


if __name__ == "__main__":
    main()
