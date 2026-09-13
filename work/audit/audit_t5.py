"""Trap #3 audit: DistilT5 residual-add / FFN-add magnitudes vs FP16 range.

Loads the real text_encoder_3 weights and measures the actual activation
magnitude at every module output, in fp32, then re-runs in fp16 to see what
actually breaks. Also reports the relative-position-bias constant (trap #2's
sibling on the T5 side).
"""

import os
import sys
import types
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "src" / "neodragon"
MODEL = ROOT / "work" / "models" / "neodragon"
sys.path.insert(0, str(REPO))

_pkg = types.ModuleType("neodragon")
_pkg.__path__ = [str(REPO / "neodragon")]
sys.modules["neodragon"] = _pkg

from neodragon.distil_t5 import T5EncoderWithProjection  # noqa: E402
from transformers import T5Tokenizer  # noqa: E402

FP16_MAX = float(np.finfo(np.float16).max)  # 65504
MAX_LEN = 128  # text_encoder_bundle.MAX_SEQUENCE_LENGTH


def load():
    tok = T5Tokenizer.from_pretrained(MODEL / "tokenizer_3")
    enc = T5EncoderWithProjection.from_pretrained(MODEL / "text_encoder_3", torch_dtype=torch.float32)
    enc.eval()
    return tok, enc


def prompts():
    p = REPO / "prompts" / "showcase_prompts.txt"
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return lines[:8]


def run_hooked(enc, batch):
    """Record max|out| for every leaf-ish module."""
    stats = {}

    def mk(name):
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(t) and t.dtype.is_floating_point:
                v = t.abs().max().item()
                stats[name] = max(stats.get(name, 0.0), v)
        return hook

    handles = [m.register_forward_hook(mk(n)) for n, m in enc.named_modules() if n]
    with torch.no_grad():
        enc(**batch)
    for h in handles:
        h.remove()
    return stats


def classify(name):
    """Which of the trap-#3 categories a module output belongs to."""
    if name.endswith("layer.0"):
        return "residual-add (self-attn)"
    if name.endswith("layer.1"):
        return "residual-add (FFN)"
    if "DenseReluDense" in name and name.endswith("DenseReluDense"):
        return "FFN inner"
    if name.endswith("wo"):
        return "FFN out proj"
    return None


def main():
    print("=" * 86)
    print("TRAP #3 -- DistilT5 residual / FFN activation magnitudes vs FP16")
    print("=" * 86)

    tok, enc = load()
    texts = prompts()
    print(f"\n[input] {len(texts)} showcase prompts, padded to max_length={MAX_LEN}")

    batch = tok(texts, padding="max_length", max_length=MAX_LEN, truncation=True,
                add_special_tokens=True, return_tensors="pt")
    batch = {"input_ids": batch.input_ids, "attention_mask": batch.attention_mask}

    stats = run_hooked(enc, batch)

    # --- residual stream, layer by layer -------------------------------------
    print(f"\n[residual stream] max|activation| per block  (fp16 max = {FP16_MAX:.0f})")
    print(f"{'block':>6}  {'after self-attn add':>20}  {'after FFN add':>16}  {'FFN inner':>12}")
    print("-" * 66)
    n_layers = enc.config.num_layers
    worst = 0.0
    for i in range(n_layers):
        base = f"encoder.encoder.block.{i}"
        a = stats.get(f"{base}.layer.0", float("nan"))
        f = stats.get(f"{base}.layer.1", float("nan"))
        inner = stats.get(f"{base}.layer.1.DenseReluDense", float("nan"))
        worst = max(worst, *(x for x in (a, f, inner) if x == x))
        flag = "  <-- OVERFLOWS FP16" if max(x for x in (a, f, inner) if x == x) > FP16_MAX else ""
        print(f"{i:>6}  {a:>20.1f}  {f:>16.1f}  {inner:>12.1f}{flag}")

    print(f"\n  peak residual magnitude: {worst:.1f}  "
          f"({worst / FP16_MAX:.2f}x fp16 max)")

    # --- everything above a fraction of fp16 range ---------------------------
    print(f"\n[modules above 1% of fp16 max ({FP16_MAX*0.01:.0f})]")
    hot = sorted((v, k) for k, v in stats.items() if v > FP16_MAX * 0.01)
    if not hot:
        print("  none")
    for v, k in hot[-15:]:
        mark = " OVERFLOW" if v > FP16_MAX else ""
        print(f"  {v:>14.1f}  {k}{mark}")

    # --- final outputs -------------------------------------------------------
    with torch.no_grad():
        out32 = enc(**batch)[0]
    print(f"\n[output] final_projection out: shape {tuple(out32.shape)}  "
          f"max|.|={out32.abs().max().item():.3f}  std={out32.std().item():.3f}")

    # --- what actually breaks in fp16 ---------------------------------------
    print("\n[fp16 rerun] same weights/inputs cast to fp16")
    enc16 = enc.half()
    with torch.no_grad():
        out16 = enc16(**batch)[0]
    n_inf = torch.isinf(out16).sum().item()
    n_nan = torch.isnan(out16).sum().item()
    print(f"  inf in output: {n_inf}   nan in output: {n_nan}   of {out16.numel()} elems")
    if n_inf == 0 and n_nan == 0:
        err = (out16.float() - out32).abs().max().item()
        rel = err / out32.abs().max().item()
        print(f"  max|fp16-fp32| = {err:.4f}  (relative {rel:.2%})")

    # --- relative position bias (T5's mask-adjacent constant) ---------------
    enc = enc.float()
    rpb = enc.encoder.encoder.block[0].layer[0].SelfAttention.relative_attention_bias
    print(f"\n[rel-pos bias] {tuple(rpb.weight.shape)}  max|.|={rpb.weight.abs().max().item():.2f}")
    print("  computed once at block 0 and reused; with seq fixed at 128 it folds")
    print("  to a constant [1, n_heads, 128, 128] and leaves the graph.")

    # --- extended attention mask, T5's own -inf equivalent ------------------
    am = batch["attention_mask"]
    ext = enc.encoder.encoder.get_extended_attention_mask(am, am.shape)
    print(f"\n[extended mask] dtype={ext.dtype}  min={ext.min().item():.6e}")
    print(f"  fp32 finfo.min = {torch.finfo(torch.float32).min:.6e}")
    print("  this additive mask is created INSIDE transformers, not the neodragon")
    print("  source, and is the T5-side instance of the trap-#2 constant.")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
