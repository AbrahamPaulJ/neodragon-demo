# Phase 5 — where the MMDiT's 248 ms actually goes

**Measured 2026-08-21 on SM-S938B / HTP V79, stage 0, `mmdit_s0_v79.bin`.**
Detailed per-op profile, `--profiling_level detailed`, 3 inferences, last (warmed) one.

The handoff left this gap "uninvestigated" and named `Reshape ×778` and
`StridedSlice ×324` as the suspects. **Both guesses were wrong.** The profile names a
single culprit that neither `analyze_net.py` nor `analyze_encodings.py` could ever have
found, because it is not a transpose, not a float fallback, and not a range problem.

> ## RESULT, measured on device 2026-08-21
>
> | | baseline | RoPE rewrite | Δ |
> |---|---|---|---|
> | **latency** (min-of-3 rounds) | 248.3 ms | **172.2 ms** | **−30.6%** |
> | deploy SNR | 19.47 dB | 19.48 dB | unchanged, as intended |
> | `Concat` cycles | 108,710,352 (28.7%) | **103,527 (0.0%)** | **−1050×** |
> | converter wall | 35:04 | 35:04 | unchanged |
>
> The rewrite is exact in fp32 (123–127 dB at all 18 shapes), so accuracy moving by
> 0.01 dB is the correct outcome — this was a pure layout change. Paper target is
> 104.7 ms; we are now 1.64× over it, down from 2.37×.

## Headline

> **RoPE — a positional embedding — is 43.3% of the graph (113.0 ms of 261.3 ms).**
> That is **6.6× the cost of all 255 FullyConnected layers combined** (17.1 ms) and
> **6.4× the cost of both attention MatMuls** (17.7 ms).
>
> 28.7% of the *entire model* is spent in 36 `Concat` ops that do nothing but
> **interleave two tensors element by element**.

Reproduce:

```
py -3.10 work/device/prof_by_optype.py \
         work/device/dprof_s0/prof_s0_base.csv work/device/mmdit_s0_net.BASELINE.json
```

## The op census

2446 nodes, 14 types, 379,127,825 accelerator cycles = 261.3 ms.

| ops | cycles | share | op type |
|---:|---:|---:|---|
| 93 | 108,710,352 | **28.7%** | Concat |
| 634 | 62,087,456 | 16.4% | Eltwise_Binary |
| 324 | 27,124,982 | 7.2% | StridedSlice |
| 18 | 26,761,363 | 7.1% | Softmax |
| 778 | 26,196,565 | 6.9% | Reshape |
| 72 | 25,968,980 | 6.8% | RmsNorm |
| 36 | 25,745,824 | 6.8% | MatMul |
| 255 | 24,795,576 | 6.5% | FullyConnected |
| 72 | 20,738,002 | 5.5% | LayerNorm |
| 38 | 18,906,609 | 5.0% | ElementWiseNeuron |
| 85 | 8,272,959 | 2.2% | Transpose |
| 36 | 3,673,815 | 1.0% | Convert |
| 3 | 45,960 | 0.0% | Conv2d |

**Real arithmetic — FullyConnected + MatMul + Conv2d — is 13.3% of the time.**
The other 86.7% is data movement, normalisation and elementwise.

Note the transposes really are clean, exactly as trap #23 promised: 2.2%. The
NONTRIVIAL layout work from the last session was correct and is not the problem.

## Which Concats

Splitting the 93 Concats by shape is what breaks the case open:

| n | out dims | inputs | cycles | share |
|---:|---|---|---:|---:|
| 36 | `[408,24,32,2]` | `[408,24,32,1] + [408,24,32,1]` | 108,626,671 | **28.7%** |
| 1 | `[1,280,1536]` | 3-way, the latent stages | 83,681 | 0.0% |
| 54 | `[408,24,64]` | `[128,24,64] + [280,24,64]` | **0** | **0.0%** |
| 2 | `[1,256]` | `[1,128] + [1,128]` | 0 | 0.0% |

The **54 joint-attention concats cost literally zero cycles.** Concatenating the text
stream (128 tokens) onto the image stream (280 tokens) is a concat on axis 0 — the
outermost axis — so QNN folds it into the producers' output allocation. It is free.

Every cycle is in the **36 concats that build a trailing axis of size 2 out of two
size-1 slices**: 18 blocks × 2 (one for Q, one for K).

## What those 36 concats are

Tracing one of them through the graph:

```
Reshape _Unsqueeze     [408,24,32] -> [408,24,32,1]   \
                                                       > Concat _Concat_4 (axis -1)
Reshape _Unsqueeze_1   [408,24,32] -> [408,24,32,1]   /        [408,24,32,2]
                                          -> Reshape _Reshape_15  [408,24,64]
```

That is the ONNX lowering of exactly one line, `work/export/export_mmdit_stage.py:263`:

```python
return torch.stack([out_even, out_odd], dim=-1).reshape(seq, heads, head_dim)
```

`torch.stack(..., dim=-1)` is `unsqueeze, unsqueeze, concat` — and a concat on the
**innermost** axis where both inputs have extent 1 is an **element-by-element
interleave**. The output alternates one 2-byte element from buffer A, one from B, for
1.25 MB, 36 times per inference. Compare with the axis-0 concat above, which copies
whole contiguous rows and costs nothing.

**Same bytes. 108.6M cycles versus 0.** The cost is entirely granularity.

This is the interleaved (GPT-J style) rotary layout: pairs are
`(x[0],x[1]), (x[2],x[3]), …`, so the even and odd components must be gathered with
stride-2 slices and scattered back with a stack.

## The full RoPE bill

The interleave is the biggest piece but not the only one. Everything below is RoPE
and nothing else:

| op | shape | n | cycles | share | ms |
|---|---|---:|---:|---:|---:|
| Concat | `[408,24,32,2]` | 36 | 108,626,671 | 28.7% | 74.9 |
| Eltwise_Binary | `[408,24,32]` | 216 | 21,385,874 | 5.6% | 14.7 |
| StridedSlice | `[408,24,32]` | 72 | 17,535,994 | 4.6% | 12.1 |
| Reshape | `[408,24,32,1]` | 72 | 8,389,448 | 2.2% | 5.8 |
| Reshape | `[408,24,64]` | 36 | 8,035,227 | 2.1% | 5.5 |
| | | | **163,973,214** | **43.3%** | **113.0** |

- the 216 `Eltwise_Binary` are the rotation itself, 12 per block:
  `cos*x_even - sin*x_odd` and `sin*x_even + cos*x_odd`, for Q and K. This is the only
  part that is real work.
- the 72 `StridedSlice` are `x[..., 0::2]` and `x[..., 1::2]`
  (`export_mmdit_stage.py:258`) — **stride-2 gathers**, the mirror image of the
  interleave.
- the 108 `Reshape` are the rank bookkeeping around the stack.

So **~87 ms of the 113 ms is pure layout churn**, and only ~15 ms is arithmetic.

Trap #1 was right that the stock rank-6 `freqs_cis` had to go, and the rank-4/rank-3
rewrite did remove the 6-D tensors. But it **kept the interleaved pair layout**, and
that is where the cost actually lived. Reducing the rank did not reduce the scatter.

## The fix

Two changes, both exact — no approximation, no retraining, no calibration change.

### 1. Make the even/odd halves contiguous, offline

Permute the **output channels of the Q and K projections** from pair-interleaved order
to half-split order, once, at export time:

```
perm = [0,2,4,…,62,  1,3,5,…,63]      # per head, head_dim = 64
```

applied to the rows of `to_q.weight` / `to_k.weight` / `add_q_proj.weight` /
`add_k_proj.weight` (and their biases), **and to the per-channel weights of the
`norm_q` / `norm_k` / `norm_add_q` / `norm_add_k` RMSNorms**, which are applied over
the same axis before RoPE.

Then `x[..., 0::2]` becomes `x[..., :32]` and `x[..., 1::2]` becomes `x[..., 32:]` —
contiguous slices instead of stride-2 gathers.

`rope_cos` / `rope_sin` are **unchanged**: they are indexed by pair index `i ∈ [0,32)`,
and the reordering does not renumber pairs, it only moves each pair's two components
apart. This matters because it means the existing 300-sample calibration set stays
valid — all graph *inputs* are untouched.

### 2. Never reassemble — score with two half MatMuls

The interleave exists only so the attention MatMul can see a contiguous `[.,64]`
head. It does not have to. With `q'` the rotated query and `k'` the rotated key,

```
score = Σᵢ (q'₂ᵢ·k'₂ᵢ + q'₂ᵢ₊₁·k'₂ᵢ₊₁)  =  ⟨q'_even, k'_even⟩ + ⟨q'_odd, k'_odd⟩
```

so

```python
score = q_even @ k_even.transpose(-1,-2) + q_odd @ k_odd.transpose(-1,-2) + mask
```

Two half-size (`d=32`) MatMuls instead of one full-size (`d=64`) one — **identical FLOPs**
— and the `stack`, both `Unsqueeze`es and the flattening `Reshape` all disappear.
`V` never goes through RoPE, so `softmax(score) @ v` and `to_out` are untouched and the
output channel order is unchanged.

This requires replacing `scaled_dot_product_attention` with the explicit
matmul/add/softmax/matmul it already lowers to (`export_mmdit_stage.py:350`).

### Expected saving

| | cycles | ms |
|---|---:|---:|
| removed: 36 `Concat` interleave | −108.6M | −74.9 |
| removed: 108 `Reshape` around the stack | −16.4M | −11.3 |
| reduced: 72 stride-2 `StridedSlice` → contiguous | ~−15M | ~−10 |
| added: 18 extra `[24,408,408]` score adds | +7.8M | +5.4 |
| **net** | **≈ −132M (−35%)** | **≈ −91 ms** |

The added cost is priced from the measured mask-add on the same shape: 18 ops,
7,798,731 cycles.

**248.3 ms → ~157 ms projected.** Still above the paper's 104.7 ms, but the remaining
budget is then dominated by ops that are genuinely doing work: Softmax (7.1%),
RmsNorm + LayerNorm (12.3%) and the elementwise AdaLN modulation.

## What it cost, and what is on top now

Predicted −91 ms, measured **−76.1 ms**. The shortfall is the extra `[24,408,408]`
score add, which came in at 15.6M cycles rather than the 7.8M estimated from the
mask-add — the second add lands on a tensor that is not already resident, so it does
not price identically.

The remaining RoPE bill is **14.3%**, down from 43.3%:

| op | shape | n | cycles | share |
|---|---|---:|---:|---:|
| Eltwise_Binary (the rotation itself — real work) | `[408,24,32]` | 216 | 25,660,450 | 8.4% |
| StridedSlice (the stride-2 gathers — still removable) | `[408,24,32]` | 72 | 18,049,520 | 5.9% |

The new op-type ranking (307.0M cycles total, detailed profile):

| share | op | note |
|---:|---|---|
| 24.0% | Eltwise_Binary | AdaLN modulation, residuals, RoPE rotation |
| 13.6% | MatMul | 54 now, was 36 |
| 12.3% | FullyConnected | |
| 12.2% | StridedSlice | 5.9% is RoPE, 5.0% is AdaLN chunking |
| 8.7% | Softmax | |
| 8.5% / 7.1% | RmsNorm / LayerNorm | |
| **0.0%** | **Concat** | **was 28.7%** |

### The next lever is op COUNT, not any single op

`StridedSlice [1,1536] ×214` — the AdaLN `chunk(6)` splits — costs 15.3M cycles, i.e.
**71.6K cycles each to copy 6 KB**. That is not work, it is per-op launch overhead. With
2378 ops in the graph and 307M cycles total (129K cycles/op average), a large fraction
of what remains is fixed dispatch cost rather than arithmetic. That reframes the
remaining 172 → 104.7 ms gap: the win will come from *fewer, larger ops*, not from
optimising any hotspot.

Two concrete follow-ups, in order of value:

1. **Fold the AdaLN `chunk(6)` into the projection.** 214 tiny slices at 5.0%. The six
   modulation tensors come from slicing one Linear output; emitting them as six
   separate outputs, or keeping them fused and broadcasting, removes the slices.
2. **Permute the Q/K projection weights to half-split order** (section below), removing
   the 72 stride-2 gathers at 5.9%.

Together ~11% ≈ 19 ms.

## What this does *not* explain

Nothing here touches accuracy. The 19.47 dB gap is the separate 8-bit K/V issue
(`docs/phase5-mmdit-accuracy.md`), and the two fixes are independent: one changes
operand *precision*, the other changes operand *layout*.

## Trap to add

> **26. `torch.stack(..., dim=-1)` is an element interleave, and it is the most
> expensive thing you can do to a tensor.** On the MMDiT it was 28.7% of the whole
> graph in 36 ops, while 54 concats of *twice the volume* on axis 0 cost exactly zero
> cycles — QNN folds an outermost-axis concat into the producers' allocation. The cost
> is granularity, not bytes, so `analyze_net.py`'s byte accounting cannot see it and
> the ms/MB constant from Phase 4 does not apply. Interleaved RoPE is the usual source;
> permute the projection weights to a half-split layout offline and the pairs never
> need to be reassembled at all. Same family as traps #1, #11, #16 and #23 — **the win
> keeps coming from removing structure, not from tuning the backend.**
