"""Trap #1 audit: RoPE tensor rank / dtype / element counts, and an equivalence
check for a rank-reduced replacement.

Runs against the upstream source with no weights loaded -- rope() and
EmbedNDRoPE are pure functions of position ids.
"""

import os
import sys
import types
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch

REPO = Path(__file__).resolve().parents[2] / "src" / "neodragon"
sys.path.insert(0, str(REPO))

# neodragon/__init__.py drags in the VAE (and timm) which we do not need here.
# Stub the parent package so the submodule loads on its own.
_pkg = types.ModuleType("neodragon")
_pkg.__path__ = [str(REPO / "neodragon")]
sys.modules["neodragon"] = _pkg

from neodragon.pyramid_mmdit import EmbedNDRoPE  # noqa: E402
from neodragon.pyramid_mmdit.modeling_embedding import rope  # noqa: E402

# From PyramidMMDiT.__init__ defaults
NUM_HEADS = 24
HEAD_DIM = 64
TEXT_LEN = 128  # text_encoder_bundle.MAX_SEQUENCE_LENGTH
THETA = 10000

# Stage shapes (T, H, W) in patch units, from the paper / research brief
STAGES = {
    "dit-low  [7x10x16]": (7, 10, 16),
    "dit-mid  [7x20x32]": (7, 20, 32),
    "dit-high [7x40x64]": (7, 40, 64),
}


def build_ids(temp, height, width, text_len=TEXT_LEN):
    """Mirror _prepare_temporal_rope_ids + the text_ids prepend in merge_input."""
    img = torch.arange(temp, dtype=torch.float32).repeat_interleave(height * width)
    ids = torch.cat([torch.zeros(text_len), img]).reshape(1, -1, 1)
    return ids


def dtype_probe():
    """rope() computes in float64 internally before a final .float()."""
    pos = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    scale = torch.arange(0, HEAD_DIM, 2, dtype=torch.float64) / HEAD_DIM
    omega = 1.0 / (THETA**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    return omega.dtype, out.dtype, torch.cos(out).dtype, rope(pos, HEAD_DIM, THETA).dtype


def apply_rope_orig(xq, freqs_cis):
    """Verbatim from modeling_block.VarlenSelfAttentionWithT5Mask.apply_rope (q only)."""
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq)


def apply_rope_reduced(xq, cos, sin):
    """Rank-reduced equivalent: everything stays 4-D."""
    x0 = xq[..., 0::2]
    x1 = xq[..., 1::2]
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos
    return torch.stack([o0, o1], dim=-1).reshape(*xq.shape)


def split_freqs(freqs_cis):
    """Pull the 2 distinct values out of the 2x2 rotation block. (B,N,1,d/2) each."""
    cos = freqs_cis[..., 0, 0]
    sin = freqs_cis[..., 1, 0]
    return cos, sin


def main():
    print("=" * 78)
    print("TRAP #1 -- RoPE tensor rank audit")
    print("=" * 78)

    o_dt, e_dt, c_dt, r_dt = dtype_probe()
    print(f"\n[dtype] omega={o_dt}  einsum={e_dt}  cos/sin={c_dt}  -> rope() returns {r_dt}")
    print("        => a float64 subgraph feeds every RoPE value before the final cast.")

    embed = EmbedNDRoPE(NUM_HEADS * HEAD_DIM, THETA, axes_dim=[HEAD_DIM])

    print(f"\n{'stage':<20} {'seq':>7} {'freqs shape':<30} {'rank':>4} {'MB fp32':>9}")
    print("-" * 78)
    rows = {}
    for name, (t, h, w) in STAGES.items():
        ids = build_ids(t, h, w)
        freqs = embed(ids)
        mb = freqs.numel() * 4 / 1024**2
        rows[name] = (ids.shape[1], freqs)
        print(f"{name:<20} {ids.shape[1]:>7} {str(tuple(freqs.shape)):<30} "
              f"{freqs.ndim:>4} {mb:>9.2f}")

    # --- redundancy inside the 2x2 block -------------------------------------
    name0 = list(STAGES)[0]
    _, f0 = rows[name0]
    cos, sin = split_freqs(f0)
    print(f"\n[2x2 block] freqs[...,0,0] == freqs[...,1,1] (cos duplicated): "
          f"{torch.equal(f0[..., 0, 0], f0[..., 1, 1])}")
    print(f"[2x2 block] freqs[...,0,1] == -freqs[...,1,0] (sin duplicated): "
          f"{torch.equal(f0[..., 0, 1], -f0[..., 1, 0])}")
    print("            => the 4-slot 2x2 block stores only 2 distinct tensors (2x waste).")

    # --- numerical equivalence of the reduced form ---------------------------
    print("\n[equivalence] reduced 4-D form vs upstream apply_rope")
    torch.manual_seed(0)
    for name in STAGES:
        seq, freqs = rows[name]
        if seq * NUM_HEADS * HEAD_DIM * 4 > 4e9:   # keep peak RAM sane
            print(f"  {name:<20} skipped (seq={seq}, would need ~{seq*NUM_HEADS*64*4/1e9:.1f} GB)")
            continue
        xq = torch.randn(1, seq, NUM_HEADS, HEAD_DIM)
        ref = apply_rope_orig(xq, freqs)
        cos, sin = split_freqs(freqs)
        got = apply_rope_reduced(xq, cos, sin)
        print(f"  {name:<20} max|diff| = {(ref - got).abs().max().item():.3e}  "
              f"shapes {tuple(ref.shape)} == {tuple(got.shape)}")

    # --- rank / intermediate accounting --------------------------------------
    print("\n[intermediate ranks] per apply_rope call, dit-high, batch 1")
    seq_hi = rows["dit-high [7x40x64]"][0]
    b, s, h, d = 1, seq_hi, NUM_HEADS, HEAD_DIM

    def mb(n):
        return n * 4 / 1024**2

    print("  upstream:")
    print(f"    xq_          {(b,s,h,d//2,1,2)}  rank 6   {mb(b*s*h*d):>9.1f} MB")
    print(f"    freqs[...,i] {(b,s,1,d//2,2)}  rank 5   {mb(b*s*1*(d//2)*2):>9.1f} MB")
    print(f"    product      {(b,s,h,d//2,2)}  rank 5   {mb(b*s*h*d):>9.1f} MB  x2 terms")
    print("  reduced:")
    print(f"    x0/x1        {(b,s,h,d//2)}     rank 4   {mb(b*s*h*d//2):>9.1f} MB")
    print(f"    cos/sin      {(b,s,1,d//2)}     rank 4   {mb(b*s*1*d//2):>9.1f} MB")
    print(f"    product      {(b,s,h,d//2)}     rank 4   {mb(b*s*h*d//2):>9.1f} MB  x4 terms")
    print("\n  multiply count is identical (4 x B*S*H*d/2); the win is RANK 6->4,")
    print("  which is what the HTP tiler chokes on, plus 2x smaller constants.")

    # --- constant-foldability ------------------------------------------------
    print("\n[constant-foldability] rope() depends only on (pos, dim, theta).")
    ids = build_ids(*STAGES[name0])
    a = embed(ids)
    b_ = embed(ids.clone())
    print(f"  deterministic across calls: {torch.equal(a, b_)}")
    print("  pos is fixed once the stage shape is fixed => the whole tensor is a")
    print("  compile-time constant and need not appear in the graph at all.")


if __name__ == "__main__":
    main()
