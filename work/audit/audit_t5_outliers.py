"""Trap #3, part 2: are the residual-stream outliers concentrated in a few
channels?

This decides the mitigation. Residual scaling divides peak and median alike,
so it fixes fp16 overflow but leaves peak/median -- and therefore the effective
bit depth under per-tensor a16 -- completely unchanged. If instead the mass sits
in a handful of d_model channels (the classic transformer outlier pattern),
then per-channel handling or a SmoothQuant-style scale migration recovers the
bits, and that is what belongs in the Phase-1 quantisation config.
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

MAX_LEN = 128


def main():
    tok = T5Tokenizer.from_pretrained(MODEL / "tokenizer_3")
    texts = [ln.strip() for ln in
             (REPO / "prompts" / "showcase_prompts.txt").read_text(encoding="utf-8").splitlines()
             if ln.strip()][:32]
    b = tok(texts, padding="max_length", max_length=MAX_LEN, truncation=True,
            add_special_tokens=True, return_tensors="pt")
    kw = {"input_ids": b.input_ids, "attention_mask": b.attention_mask}

    enc = T5EncoderWithProjection.from_pretrained(
        MODEL / "text_encoder_3", torch_dtype=torch.float32).eval()

    caught = {}

    def hook(i):
        def f(_m, _i, out):
            caught[i] = (out[0] if isinstance(out, tuple) else out).detach()
        return f

    hs = [bl.register_forward_hook(hook(i)) for i, bl in enumerate(enc.encoder.encoder.block)]
    with torch.no_grad():
        enc(**kw)
    for h in hs:
        h.remove()

    print("=" * 84)
    print(f"TRAP #3 part 2 -- outlier channel concentration ({len(texts)} prompts, d_model=768)")
    print("=" * 84)
    print(f"\n{'block':>5}  {'peak':>10}  {'top-1 ch':>9}  {'#ch >1% pk':>11}  "
          f"{'per-tensor':>10}  {'per-channel':>11}  {'gain':>6}")
    print("-" * 76)

    for i in sorted(caught):
        t = caught[i]                       # [B, T, C]
        flat = t.reshape(-1, t.shape[-1])
        ch_max = flat.abs().amax(0)         # [C]
        peak = ch_max.max().item()
        top1 = int(ch_max.argmax())
        n_hot = int((ch_max > 0.01 * peak).sum())

        # effective bits for a typical activation, per-tensor vs per-channel a16
        med = flat.abs().median().item()
        lsb_tensor = peak / 32767
        # per-channel: each channel gets its own scale
        lsb_ch = (ch_max / 32767).clamp_min(1e-12)
        med_ch = flat.abs().median(0).values
        bits_tensor = np.log2(max(med / lsb_tensor, 1e-9))
        bits_ch = torch.log2((med_ch / lsb_ch).clamp_min(1e-9)).median().item()
        print(f"{i:>5}  {peak:>10,.0f}  {top1:>9}  {n_hot:>11}  "
              f"{bits_tensor:>10.1f}  {bits_ch:>11.1f}  {bits_ch - bits_tensor:>+6.1f}")

    # which channels, and are they stable across blocks?
    print("\n[channel identity] top-5 channels by max magnitude")
    tops = {}
    for i in sorted(caught):
        ch_max = caught[i].reshape(-1, caught[i].shape[-1]).abs().amax(0)
        tops[i] = torch.topk(ch_max, 5).indices.tolist()
        print(f"  block {i:>2}: {tops[i]}")

    common = set.intersection(*(set(v) for v in tops.values()))
    print(f"\n  channels in the top-5 of EVERY block: {sorted(common)}")
    print("  a fixed small channel set => the outliers are structural, not")
    print("  input-dependent, so a static per-channel scale handles them.")

    # how much of the energy is in those channels?
    last = caught[max(caught)]
    flat = last.reshape(-1, last.shape[-1])
    ch_max = flat.abs().amax(0)
    order = torch.argsort(ch_max, descending=True)
    for k in (1, 2, 4, 8, 16):
        kept = ch_max[order[k:]].max().item()
        print(f"  drop top-{k:>2} channels of the last block -> peak falls "
              f"{ch_max.max().item():,.0f} -> {kept:,.0f} "
              f"({ch_max.max().item()/kept:.0f}x)")


if __name__ == "__main__":
    main()
