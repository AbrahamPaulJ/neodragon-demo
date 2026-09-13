"""Phase 5c -- PyramidMMDiT as a single-stage static graph with a host-hoisted preamble.

Read docs/phase5-mmdit-scope.md first. The three findings this file acts on:

  1. `num_stages == 1` on every AR t2v invocation (`dit(sample=[latent_model_input])`),
     so every Python stage loop degenerates to a straight line.
  2. The module DOES trace. What pollutes the graph is `merge_input`'s mask construction
     (Shape x186, Where x196, ConstantOfShape x201), which survives constant folding
     because it descends from the `encoder_attention_mask` INPUT.
  3. `AdaLayerNorm*.forward_with_pad` materialises `zeros_like(x).repeat(1,1,6)` and
     scatters into it once per block. At num_stages == 1 that is a plain broadcast --
     and 18 blocks x ~11 nodes is essentially all 201 ConstantOfShape nodes.

Strategy: patch the PLUMBING, keep the MATH. Every weight-bearing module -- the blocks,
the feed-forwards, the norms, the projections -- is the stock object, untouched. Only four
things are replaced, and each one is verified against the stock model at every shape:

  * `merge_input`            -> host-side `stage_conditioning()`, returning `attn_mask`,
                                `rope_cos`, `rope_sin` as ordinary tensors
  * `PatchEmbed3D.forward`   -> precomputed per-latent `pos_embed` crops held as buffers,
                                so the 226 MB `pos_embed` table never reaches the graph
  * `AdaLayerNorm*.forward`  -> the plain broadcast path (`hidden_length` ignored)
  * `apply_rope`             -> RANK 4, trap #1. `freqs_cis` is [b, s, 1, 32, 2, 2]; the
                                replacement carries `cos` and `sin` as [1, s, 1, 32] and
                                computes the rotation directly. Same multiply count --
                                the win is the rank, exactly as in the Phase 1 audit.

Padding is NOT applied here. This exports one exact shape; `mmdit_shapes.py` says the 18
shapes collapse into 3 padded envelopes, and padding is layered on top once the exact
shapes are proven equivalent.

  usage: py -3.10 work/export/export_mmdit_stage.py --stage 0 --unit 6 [--export]
"""

import argparse
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
sys.path.insert(0, str(ROOT / "work" / "audit"))
_pkg = types.ModuleType("neodragon")
_pkg.__path__ = [str(REPO / "neodragon")]
sys.modules["neodragon"] = _pkg

from mmdit_shapes import past_condition_shapes, pyramid_hw, tokens  # noqa: E402

TEXT_TOKENS = 128
CAPTION_DIM = 1536      # context_adapter output; NOT config.joint_attention_dim
POOLED_DIM = 2048
LAT_C = 16
MASK_FILL = -100.0      # trap #2: never -inf. Phase 1 measured 2.8e-42 leakage.


# --------------------------------------------------------------------------- #
# host-side preamble: everything merge_input() computes that is not latent data
# --------------------------------------------------------------------------- #
def stage_conditioning(dit, shapes, encoder_attention_mask, dtype=torch.float32):
    """Reproduce merge_input()'s mask and RoPE, on the host, as plain tensors.

    Returns (attn_mask [1,1,S,S] additive float, rope_cos [1,S,1,D/2],
             rope_sin [1,S,1,D/2]) where S = TEXT_TOKENS + sum(image tokens).

    Depends only on the token layout and the prompt mask -- never on latent values.
    """
    device = encoder_attention_mask.device
    bs = encoder_attention_mask.shape[0]
    patch = dit.patch_size

    # --- temporal position ids, exactly _prepare_pyramid_temporal_rope_ids ---
    ids, start = [], 0
    for (t, h, w) in shapes:
        ids.append(dit._prepare_temporal_rope_ids(
            bs, t, h // patch, w // patch, device, start_time_stamp=start))
        start += t
    image_ids = torch.cat(ids, dim=1)
    text_ids = torch.zeros(bs, encoder_attention_mask.shape[1], 1, device=device)
    input_ids = torch.cat([text_ids, image_ids], dim=1)             # [b, S, 1]

    # --- RoPE tables. Stock: temp_rope_embed(input_ids) -> [b,S,1,D/2,2,2] ----
    # rope() builds stack([cos, -sin, sin, cos]).view(..., 2, 2), so column 0 of
    # each 2x2 is (cos, sin) and column 1 is (-sin, cos). Recover cos and sin and
    # ship them at rank 4 (trap #1: the win is the rank, not the multiply count).
    freqs = dit.temp_rope_embed(input_ids)                          # [b,S,1,D/2,2,2]
    assert freqs.dim() == 6, freqs.shape
    # RANK 3, not 4. A rank-4 tensor entering the attention chain is read as image
    # layout by the converter and permuted -- the rank-4 build measured a
    # transpose/compute ratio of 1.35, worse than the VAE decoder's catastrophic 1.16.
    rope_cos = freqs[0, ..., 0, 0].to(dtype)                        # [S,1,D/2]
    rope_sin = freqs[0, ..., 1, 0].to(dtype)                        # [S,1,D/2]
    assert torch.allclose(freqs[..., 1, 1], freqs[..., 0, 0]), "cos not on the diagonal"
    assert torch.allclose(freqs[..., 0, 1], -freqs[..., 1, 0]), "sin not antisymmetric"

    # --- attention mask, exactly merge_input()'s construction ----------------
    n_img = sum(tokens(s) for s in shapes)
    tid = torch.arange(1, bs + 1, dtype=encoder_attention_mask.dtype,
                       device=device).unsqueeze(1).repeat(1, encoder_attention_mask.shape[1])
    tid[encoder_attention_mask == 0] = 0                            # the "boolean scatter"
    iid = torch.arange(1, bs + 1, dtype=encoder_attention_mask.dtype,
                       device=device).unsqueeze(1).repeat(1, n_img)
    token_ids = torch.cat([tid, iid], dim=1)
    mask = (token_ids[:, None, :, None] == token_ids[:, None, None, :])
    if dit.use_temporal_causal:
        order = input_ids.squeeze(2)
        mask = mask & (order[:, None, :, None] >= order[:, None, None, :])

    # additive form with a FINITE fill -- SDPA would lower a bool mask to -inf
    add_mask = torch.zeros(mask.shape, dtype=dtype, device=device)
    add_mask.masked_fill_(~mask, MASK_FILL)
    return add_mask.squeeze(1), rope_cos, rope_sin                  # [1,S,S], rank 3


def envelope_for(stage):
    """The per-stage padding envelope: the largest shape list the stage ever sees."""
    shapes = past_condition_shapes(6, stage)
    h, w = pyramid_hw(stage)
    return shapes + [(1, h, w)]


def align_to_envelope(real_shapes, env_shapes):
    """Map a short unit's latents onto the envelope's slots, padding at the FRONT.

    `split_output` slices the TRAILING tokens, so the current latent must stay last;
    padding therefore goes on the history side. Slots are matched back-to-front by
    spatial size, the frontmost matched slot absorbs the temporal padding, and any
    envelope slot left over in front is entirely dummy.

    Returns a list, one entry per envelope slot:
        (real_index or None, n_pad_frames)
    """
    plan = [None] * len(env_shapes)
    i, j = len(real_shapes) - 1, len(env_shapes) - 1
    while i >= 0 and j >= 0:
        (rt, rh, rw), (et, eh, ew) = real_shapes[i], env_shapes[j]
        if (rh, rw) != (eh, ew):
            break                       # spatial mismatch: this slot is pure padding
        assert rt <= et, (real_shapes[i], env_shapes[j])
        plan[j] = (i, et - rt)
        i, j = i - 1, j - 1
    assert i < 0, "real shapes do not fit the envelope: {} vs {}".format(
        real_shapes, env_shapes)
    for k in range(j + 1):
        plan[k] = (None, env_shapes[k][0])
    return plan


def pad_latents(lats, real_shapes, env_shapes):
    """Zero-pad each latent's temporal axis at the front to the envelope's shape."""
    plan = align_to_envelope(real_shapes, env_shapes)
    out = []
    for slot, (t, h, w) in zip(plan, env_shapes):
        ri, n_pad = slot
        if ri is None:
            out.append(torch.zeros(1, LAT_C, t, h, w, dtype=lats[0].dtype))
            continue
        x = lats[ri]
        if n_pad:
            x = torch.cat([torch.zeros(1, LAT_C, n_pad, h, w, dtype=x.dtype), x], dim=2)
        out.append(x)
    return out


def stage_conditioning_padded(dit, real_shapes, env_shapes, encoder_attention_mask,
                              dtype=torch.float32):
    """Conditioning for a short unit run through the envelope graph.

    The one thing that makes front-padding exact here: because RoPE was hoisted to the
    host, we choose the position id of every slot. Real tokens keep the ids they would
    have had at their true shape, so every real-to-real relative position -- including
    text-to-image, where the text sits at id 0 and is NOT shifted -- is unchanged.
    Naively prepending frames would have shifted every image token and silently altered
    every text-image score.

    Padded slots are hidden by the mask. A fully-masked query row is finite, not NaN,
    because MASK_FILL is -100 rather than -inf (trap #2); such rows belong to padding
    and are discarded by the output slice anyway.
    """
    device = encoder_attention_mask.device
    bs = encoder_attention_mask.shape[0]
    patch = dit.patch_size
    plan = align_to_envelope(real_shapes, env_shapes)

    # true temporal ids for the REAL shapes, in real order
    real_ids, start = [], 0
    for (t, h, w) in real_shapes:
        real_ids.append(dit._prepare_temporal_rope_ids(
            bs, t, h // patch, w // patch, device, start_time_stamp=start))
        start += t

    ids, valid = [], []
    for slot, (t, h, w) in zip(plan, env_shapes):
        ri, n_pad = slot
        n_spatial = (h // patch) * (w // patch)
        if ri is None:
            ids.append(torch.zeros(bs, t * n_spatial, 1, device=device))
            valid.append(torch.zeros(t * n_spatial, dtype=torch.bool, device=device))
            continue
        pad_tok = n_pad * n_spatial
        if pad_tok:
            ids.append(torch.zeros(bs, pad_tok, 1, device=device))
            valid.append(torch.zeros(pad_tok, dtype=torch.bool, device=device))
        ids.append(real_ids[ri])
        valid.append(torch.ones(real_ids[ri].shape[1], dtype=torch.bool, device=device))

    image_ids = torch.cat(ids, dim=1)
    image_valid = torch.cat(valid, dim=0)
    text_ids = torch.zeros(bs, encoder_attention_mask.shape[1], 1, device=device)
    input_ids = torch.cat([text_ids, image_ids], dim=1)

    freqs = dit.temp_rope_embed(input_ids)
    rope_cos = freqs[0, ..., 0, 0].to(dtype)                        # [S,1,D/2]
    rope_sin = freqs[0, ..., 1, 0].to(dtype)                        # [S,1,D/2]

    n_img = image_ids.shape[1]
    tid = torch.arange(1, bs + 1, dtype=encoder_attention_mask.dtype,
                       device=device).unsqueeze(1).repeat(1, encoder_attention_mask.shape[1])
    tid[encoder_attention_mask == 0] = 0
    iid = torch.arange(1, bs + 1, dtype=encoder_attention_mask.dtype,
                       device=device).unsqueeze(1).repeat(1, n_img)
    token_ids = torch.cat([tid, iid], dim=1)
    mask = (token_ids[:, None, :, None] == token_ids[:, None, None, :])
    if dit.use_temporal_causal:
        order = input_ids.squeeze(2)
        mask = mask & (order[:, None, :, None] >= order[:, None, None, :])

    # hide padded KEYS from every query
    key_valid = torch.cat([torch.ones(encoder_attention_mask.shape[1], dtype=torch.bool,
                                      device=device), image_valid])
    mask = mask & key_valid[None, None, None, :]

    add_mask = torch.zeros(mask.shape, dtype=dtype, device=device)
    add_mask.masked_fill_(~mask, MASK_FILL)
    return add_mask.squeeze(1), rope_cos, rope_sin                  # [1,S,S], rank 3


# --------------------------------------------------------------------------- #
# the graph
# --------------------------------------------------------------------------- #
def apply_rope_rank3(x, cos, sin):
    """RoPE on [S, h, D], returning the even and odd halves SEPARATELY.

    Stock `apply_rope` reshapes x to [b,s,h,D/2,1,2] and contracts against a rank-6
    freqs_cis -- trap #1. Same arithmetic, same multiply count, at rank 3.

    This used to end with `torch.stack([out_even, out_odd], dim=-1).reshape(...)` to
    put the rotated pairs back into one [S,h,D] tensor. That single line measured
    **28.7% of the entire stage-0 graph** -- 36 Concat ops, 108.6M cycles, 74.9 ms
    (docs/phase5-mmdit-latency.md). `torch.stack(..., dim=-1)` lowers to
    Unsqueeze/Unsqueeze/Concat on the INNERMOST axis, i.e. an element-by-element
    interleave; for comparison the 54 concats that join the text and image streams on
    axis 0 -- twice the volume -- cost exactly ZERO cycles, because QNN folds an
    outermost-axis concat into its producers. The cost is granularity, not bytes.

    The halves never have to be reassembled. The attention score is a sum over pairs:

        <q', k'>  =  <q'_even, k'_even> + <q'_odd, k'_odd>

    so `_attention` contracts the two halves separately and adds. Identical FLOPs,
    identical result, no interleave. Trap #26, same family as #1/#11/#16/#23.

    Shapes are no longer passed in because nothing here needs them (trap #19 is moot
    once the reshape is gone).
    """
    x_even, x_odd = x[..., 0::2], x[..., 1::2]                      # [S,h,D/2]
    return cos * x_even - sin * x_odd, sin * x_even + cos * x_odd


class StageMMDiT(nn.Module):
    """One pyramid stage, one static shape. All weights are the stock modules."""

    def __init__(self, dit, shapes):
        super().__init__()
        self.dit = dit
        self.shapes = list(shapes)
        self.patch = dit.patch_size
        self.heads = dit.config.num_attention_heads
        self.head_dim = dit.inner_dim // self.heads
        self.n_img = sum(tokens(s) for s in self.shapes)

        # --- precompute the pos_embed crops (constant per shape) -------------
        # PatchEmbed3D.cropped_pos_embed() crops the 192x192 table to the FINEST
        # latent's grid and bilinearly resizes it to each condition's own grid.
        # Both depend only on shapes, so they are constants: only the crops ship,
        # never the 226 MB table.
        pe = dit.pos_embed
        oh, ow = self.shapes[-1][1], self.shapes[-1][2]
        for i, (_, h, w) in enumerate(self.shapes):
            with torch.no_grad():
                crop = pe.cropped_pos_embed(h, w, oh, ow)           # [1, h*w/patch^2, C]
            self.register_buffer("pos_embed_{}".format(i), crop.clone(), persistent=False)

    def patchify(self, lats):
        """PatchEmbed3D.forward for a list of latents, with pos_embed precomputed.

        Every dimension here is known from self.shapes, so nothing queries the tensor.
        Asking `lat.shape[0]` inside a traced graph emits Shape+Gather that constant
        folding cannot remove -- 112 of them survived the first export.
        """
        pe = self.dit.pos_embed
        c, dim = LAT_C, self.dit.inner_dim
        out = []
        for i, (lat, (t, hh, ww)) in enumerate(zip(lats, self.shapes)):
            n = (hh // self.patch) * (ww // self.patch)
            x = lat.permute(0, 2, 1, 3, 4).reshape(t, c, hh, ww)
            x = pe.proj(x).reshape(t, dim, n).transpose(1, 2)       # [t, n, C]
            x = x + getattr(self, "pos_embed_{}".format(i))
            out.append(x.reshape(1, t * n, dim))                    # b=1
        return torch.cat(out, dim=1)

    def _attention(self, block, hidden, enc_hidden, cos, sin, mask):
        """JointAttention.forward + the num_stages==1 body of VarlenSelfAttentionWithT5Mask.

        Everything inside the attention is RANK 3. The rank-4 version of this measured a
        transpose/compute ratio of 1.35 -- worse than the VAE decoder's catastrophic 1.16
        -- because the converter reads [1, heads, S, d] as image layout and permutes it.
        The score matrix alone came back as [1, 408, 408, 24] at 7.99 MB, 36 times over.
        At rank 3 the converter leaves the axes alone. Trap #11's family: the axis
        meanings are ours, the layout rules are the backend's, and RANK decides which apply.
        """
        a = block.attn
        h, d = self.heads, self.head_dim
        enc_len = TEXT_TOKENS

        seq = enc_len + self.n_img

        def split(t, n):
            return t.reshape(n, h, d)                              # [n, h, d], no batch

        q = split(a.to_q(hidden), self.n_img)
        k = split(a.to_k(hidden), self.n_img)
        v = split(a.to_v(hidden), self.n_img)
        if a.norm_q is not None:
            q = a.norm_q(q)                                        # RMSNorm over the last axis
        if a.norm_k is not None:
            k = a.norm_k(k)

        eq = split(a.add_q_proj(enc_hidden), enc_len)
        ek = split(a.add_k_proj(enc_hidden), enc_len)
        ev = split(a.add_v_proj(enc_hidden), enc_len)
        if a.norm_add_q is not None:
            eq = a.norm_add_q(eq)
        if a.norm_add_k is not None:
            ek = a.norm_add_k(ek)

        # text first, then image -- the stock concat order, now on the token axis (0)
        q = torch.cat([eq, q], dim=0)
        k = torch.cat([ek, k], dim=0)
        v = torch.cat([ev, v], dim=0)

        # SDPA scales the QUERY by 1/sqrt(head_dim) before the contraction, and that is
        # deliberately kept here rather than scaling the score afterwards: q is
        # [S,h,64] = 1.25 MB, the score is [24,408,408] = 8 MB, and in a W8A16 graph the
        # bigger tensor also means one more 16-bit encoding to burn. Note the scale is
        # over the FULL head_dim -- taking it from a half below would be sqrt(2) off.
        q = q * (d ** -0.5)

        qe, qo = apply_rope_rank3(q, cos, sin)                      # [S, h, d/2] each
        ke, ko = apply_rope_rank3(k, cos, sin)

        qe, qo, ke, ko, v = (t.transpose(0, 1)
                             for t in (qe, qo, ke, ko, v))          # [h, S, *]

        # scaled_dot_product_attention, written out, with the score contracted as two
        # half-width MatMuls so the rotated halves never need interleaving back
        # together (see apply_rope_rank3). Identical FLOPs: 2 x (d/2) == 1 x d.
        o = (torch.matmul(qe, ke.transpose(-1, -2))
             + torch.matmul(qo, ko.transpose(-1, -2)))
        o = torch.matmul(torch.softmax(o + mask, dim=-1), v)        # [h, S, d]
        o = o.transpose(0, 1).reshape(1, seq, h * d)                # [1, S, h*d]

        enc_out = a.to_add_out(o[:, :enc_len]) if not a.context_pre_only else None
        return a.to_out[0](o[:, enc_len:]), enc_out

    def _block(self, block, hidden, enc_hidden, temb, cos, sin, mask):
        """JointTransformerBlock.forward, with hidden_length dropped (see module docstring)."""
        norm_h, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.norm1(hidden, emb=temb)

        if block.context_pre_only:
            norm_e = block.norm1_context(enc_hidden, temb)
            c_gate_msa = c_shift_mlp = c_scale_mlp = c_gate_mlp = None
        else:
            (norm_e, c_gate_msa, c_shift_mlp,
             c_scale_mlp, c_gate_mlp) = block.norm1_context(enc_hidden, emb=temb)

        attn_out, ctx_attn_out = self._attention(block, norm_h, norm_e, cos, sin, mask)

        hidden = hidden + gate_msa[:, None] * attn_out
        norm_h = block.norm2(hidden) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        hidden = hidden + gate_mlp[:, None] * block.ff(norm_h)

        if block.context_pre_only:
            return None, hidden

        enc_hidden = enc_hidden + c_gate_msa[:, None] * ctx_attn_out
        norm_e = block.norm2_context(enc_hidden) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        enc_hidden = enc_hidden + c_gate_mlp[:, None] * block.ff_context(norm_e)
        return enc_hidden, hidden

    def forward(self, encoder_hidden_states, pooled_projections, timestep_ratio,
                attn_mask, rope_cos, rope_sin, *lats):
        dit = self.dit
        temb = dit.time_text_embed(timestep_ratio, pooled_projections)
        hidden = self.patchify(lats)
        enc = encoder_hidden_states

        for block in dit.transformer_blocks:
            enc, hidden = self._block(block, hidden, enc, temb, rope_cos, rope_sin, attn_mask)

        # norm_out is AdaLayerNormContinuous; the plain path is the num_stages==1 path
        hidden = dit.norm_out(hidden, temb)
        hidden = dit.proj_out(hidden)

        # split_output for one stage: keep only the current latent's trailing tokens
        t, h, w = self.shapes[-1]
        h, w, p = h // self.patch, w // self.patch, self.patch
        hidden = hidden[:, -(t * h * w):]
        hidden = hidden.reshape(1, t, h, w, p, p, dit.out_channels)
        hidden = hidden.permute(0, 6, 1, 2, 4, 3, 5)                # b c t h p1 w p2
        return hidden.reshape(1, dit.out_channels, t, h * p, w * p)


# --------------------------------------------------------------------------- #
def build_latents(unit, stage, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    shapes = past_condition_shapes(unit, stage)
    h, w = pyramid_hw(stage)
    shapes = shapes + [(1, h, w)]
    lats = [torch.randn(1, LAT_C, t, hh, ww, generator=g, dtype=dtype)
            for (t, hh, ww) in shapes]
    return lats, shapes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--unit", type=int, default=6)
    ap.add_argument("--all-shapes", action="store_true",
                    help="verify every (unit, stage) shape, not just one")
    ap.add_argument("--envelope", action="store_true",
                    help="export the stage's PADDING ENVELOPE graph (the shipping one)")
    ap.add_argument("--pad-check", action="store_true",
                    help="run every short unit through its stage's envelope graph")
    ap.add_argument("--export", action="store_true")
    args = ap.parse_args()

    from neodragon.pyramid_mmdit import PyramidMMDiT

    dtype = torch.float32
    print("=" * 78)
    print("PHASE 5c -- single-stage MMDiT, host-hoisted mask/RoPE, rank-4 RoPE")
    print("=" * 78)
    print("")
    print("[load] diffusion_transformer_320p (fp32, CPU) ...")
    dit = PyramidMMDiT.from_pretrained(str(MODEL / "diffusion_transformer_320p"),
                                       torch_dtype=dtype).eval()
    print("       {:.1f} M params, {} blocks".format(
        sum(p.numel() for p in dit.parameters()) / 1e6, len(dit.transformer_blocks)))

    cases = ([(u, s) for u in range(1, 7) for s in range(3)]
             if args.all_shapes else [(args.unit, args.stage)])

    torch.manual_seed(0)
    ehs = torch.randn(1, TEXT_TOKENS, CAPTION_DIM, dtype=dtype)
    pooled = torch.randn(1, POOLED_DIM, dtype=dtype)
    tstep = torch.tensor([0.5], dtype=dtype)
    eam = torch.ones(1, TEXT_TOKENS, dtype=dtype)
    eam[:, 100:] = 0                      # a realistic padded prompt, not all-ones

    worst = 0.0
    net = None
    for (u, s) in cases:
        lats, shapes = build_latents(u, s, dtype, seed=u * 10 + s)
        net = StageMMDiT(dit, shapes).eval()
        mask, cos, sin = stage_conditioning(dit, shapes, eam, dtype)

        with torch.no_grad():
            ref = dit(sample=[lats], encoder_hidden_states=ehs,
                      encoder_attention_mask=eam, pooled_projections=pooled,
                      timestep_ratio=tstep)[0]
            got = net(ehs, pooled, tstep, mask, cos, sin, *lats)

        assert ref.shape == got.shape, (ref.shape, got.shape)
        snr = 20 * torch.log10(ref.norm() / (got - ref).norm()).item()
        md = (got - ref).abs().max().item()
        worst = max(worst, md)
        print("[check] unit {} stage {}  {:>5} img tok  {:<20} "
              "max|diff|={:.3e}  SNR={:.1f} dB".format(
                  u, s, net.n_img, str(tuple(ref.shape)), md, snr))
        assert md < 2e-4, "single-stage rewrite does not match the stock model"

    print("[check] PASS -- worst max|diff| over {} shape(s): {:.3e}".format(
        len(cases), worst))

    # ---- padding: run short units through the per-stage envelope ------------
    if args.pad_check:
        print("")
        print("=" * 78)
        print("PADDING CHECK -- short units through the per-stage envelope graph")
        print("=" * 78)
        for stage in range(3):
            env = envelope_for(stage)
            net_env = StageMMDiT(dit, env).eval()
            print("")
            print("  stage {} envelope {}  ({} image tokens)".format(
                stage, " ".join("{}x{}x{}".format(*x) for x in env), net_env.n_img))
            for u in range(1, 7):
                lats, shapes = build_latents(u, stage, dtype, seed=u * 10 + stage)
                mask_e, cos_e, sin_e = stage_conditioning_padded(
                    dit, shapes, env, eam, dtype)
                lats_e = pad_latents(lats, shapes, env)
                with torch.no_grad():
                    ref = dit(sample=[lats], encoder_hidden_states=ehs,
                              encoder_attention_mask=eam, pooled_projections=pooled,
                              timestep_ratio=tstep)[0]
                    got = net_env(ehs, pooled, tstep, mask_e, cos_e, sin_e, *lats_e)
                md = (got - ref).abs().max().item()
                snr = 20 * torch.log10(ref.norm() / (got - ref).norm()).item()
                pad = net_env.n_img - sum(tokens(x) for x in shapes)
                print("    unit {}  +{:>4} padded tokens  max|diff|={:.3e}  "
                      "SNR={:.1f} dB".format(u, pad, md, snr))
                assert md < 1e-3, "padding is not transparent at unit {}".format(u)
        print("")
        print("[pad]   PASS -- the envelope graph reproduces every short unit")

    if not args.export:
        print("")
        print("pass --export to write the ONNX for the last case")
        return

    # ---- export -------------------------------------------------------------
    u, s = cases[-1]
    if args.envelope:
        # the shipping graph: the stage's envelope, which is unit 6's shape. Its
        # conditioning comes from stage_conditioning_padded so that the exported
        # graph is exactly what every unit 1..6 will be fed.
        shapes = envelope_for(args.stage)
        lats, real_shapes = build_latents(6, args.stage, dtype, seed=60 + args.stage)
        mask, cos, sin = stage_conditioning_padded(dit, real_shapes, shapes, eam, dtype)
        lats = pad_latents(lats, real_shapes, shapes)
        net = StageMMDiT(dit, shapes).eval()
        s, tag = args.stage, "mmdit_s{}_env".format(args.stage)
        print("")
        print("[envelope] stage {}: {}  -> {} image tokens, {} inputs".format(
            args.stage, " ".join("{}x{}x{}".format(*x) for x in shapes),
            net.n_img, 6 + len(lats)))
    else:
        lats, shapes = build_latents(u, s, dtype, seed=u * 10 + s)
        mask, cos, sin = stage_conditioning(dit, shapes, eam, dtype)
        tag = "mmdit_s{}u{}".format(s, u)
    dst_dir = OUT / tag                   # own directory: >2 GB means external data
    dst_dir.mkdir(parents=True, exist_ok=True)
    raw = dst_dir / (tag + "_raw.onnx")

    in_names = (["encoder_hidden_states", "pooled_projections", "timestep_ratio",
                 "attn_mask", "rope_cos", "rope_sin"]
                + ["latent_{}".format(i) for i in range(len(lats))])
    args_t = (ehs, pooled, tstep, mask, cos, sin, *lats)

    print("")
    print("[export] {} inputs -> {}".format(len(in_names), raw))
    torch.onnx.export(
        net, args_t, str(raw),
        input_names=in_names, output_names=["noise_pred"],
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    import onnx
    from graph_fixes import op_histogram, scan_extreme
    g = onnx.load(str(raw), load_external_data=True)

    hist = op_histogram(g)
    print("[ops]    " + ", ".join("{}x{}".format(k, v) for k, v in
                                  sorted(hist.items(), key=lambda kv: -kv[1])[:18]))
    # trap #2: scan BEFORE externalising. Once save_model has moved the initializers
    # out to a weights blob, numpy_helper.to_array cannot read them back in place.
    print("[safe]   unsafe constants: {}".format(len(scan_extreme(g))))

    # Consolidate the external data. torch writes ONE FILE PER TENSOR -- 659 of them
    # for this model -- and the converter reads the ONNX from /mnt/c over 9p, where
    # per-file overhead dominates. One weights blob turns that into a single big read.
    for f in dst_dir.iterdir():
        if f.name != raw.name:
            f.unlink()
    onnx.save_model(g, str(raw), save_as_external_data=True,
                    all_tensors_to_one_file=True, location=tag + "_weights.bin",
                    size_threshold=1024)
    print("[fold]   external data consolidated -> {}_weights.bin".format(tag))
    total = sum(f.stat().st_size for f in dst_dir.iterdir())
    print("[size]   {:.2f} GB across {} files".format(total / 1024 ** 3,
                                                      len(list(dst_dir.iterdir()))))
    print("")
    print("compare against the stock trace (docs/phase5-mmdit-scope.md, Correction 3):")
    print("  ConstantOfShape 201, Where 196, Shape 186, Gather 184, Equal 179")


if __name__ == "__main__":
    main()
