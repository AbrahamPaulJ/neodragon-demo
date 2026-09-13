# Stage 2 is bandwidth-bound, not dispatch-bound — and the RoPE fix backfired at 1728 tokens

**Session 7, 2026-08-23. Entirely static + a re-read of an existing device profile.
No new device time was needed to establish any of this.**

The roadmap (Track A1) said stage 2's 1851 ms was **dispatch** cost:

> "Stage 0's profile said the remaining cost is dispatch, not arithmetic — 2378 ops at
> ~129K cycles each. Stage 2 has 1730 nodes and the cleanest graph of the three
> (transpose/compute 0.06), so the layout work is done; what is left is the sheer number
> of dispatches over a 1728-token sequence."

**That is wrong, and two independent measurements already in the repo say so.**

---

## 1. Op count cannot explain it — the three stages have the SAME op count

`work/device/stage_census.py` over the three shipping `*_net.json`:

| | nodes | GMAC | op output MB | measured latency |
|---|---:|---:|---:|---:|
| `mmdit_s0g` | 1719 | 214.97 | 1808.2 | 169.2 ms |
| `mmdit_s1f` | 1719 | 351.33 | 3550.6 | 375.8 ms |
| `mmdit_s2f` | 1730 | 1043.82 | 17559.7 | **1851 ms** |
| **stage 2 / stage 0** | **1.01×** | 4.86× | **9.71×** | **10.94×** |

The graphs are structurally identical — same 18 blocks, same ops — so node count is
*flat* across stages while latency moves **10.94×**. A quantity that does not vary
cannot explain a quantity that varies eleven-fold.

What does track: **op output bytes, 9.71× against 10.94× measured.** MACs (4.86×) badly
under-predict. Per-stage that is:

| stage | ms / MB of op output |
|---|---:|
| s0g | 0.0936 |
| s1f | 0.1058 |
| s2f | 0.1054 |

## 2. The already-measured control: removing 27.6% of the ops bought 1.7%

The AdaLN fix (session 6) took stage 0 from **2374 nodes to 1719** — it deleted 655 ops,
including 216 `StridedSlice` and 532 `Reshape`. Latency went **172.2 ms → 169.2 ms**.

> **A 27.6% cut in op count bought a 1.7% cut in latency.** If dispatch were the binding
> cost this is precisely the experiment that would have shown it, and it did the opposite.

## 3. Why bytes, validated inside a single graph

Three points from one graph family is not enough to design on, so the same claim was
tested against the existing stage-0 detailed profile — ~1700 ops, each with a measured
cycle count (`work/device/bytes_model.py`, joining `prof_s0_rope.csv` to
`mmdit_s0_net.json`; 2374 of 2378 rows matched):

| op type | n | cycles | share | out MB | cyc/byte | vs bulk |
|---|---:|---:|---:|---:|---:|---:|
| Eltwise_Binary | 634 | 73,789,048 | 24.0% | 626.5 | 0.12 | 0.82× |
| MatMul | 54 | 41,769,050 | 13.6% | 310.2 | 0.13 | **0.94×** |
| FullyConnected | 255 | 37,779,154 | 12.3% | 201.4 | 0.19 | 1.31× |
| StridedSlice | 324 | 37,317,994 | 12.2% | 68.0 | 0.55 | 3.83× |
| Softmax | 18 | 26,824,383 | 8.7% | 143.8 | 0.19 | 1.30× |
| RmsNorm | 72 | 26,176,628 | 8.5% | 45.1 | 0.58 | 4.05× |
| LayerNorm | 72 | 21,708,597 | 7.1% | 45.6 | 0.48 | 3.32× |
| Reshape | 670 | 10,491,710 | 3.4% | 427.0 | 0.02 | 0.17× |
| Transpose | 121 | 6,141,650 | 2.0% | 92.0 | 0.07 | 0.47× |
| Concat | 57 | 103,527 | 0.0% | 68.5 | 0.00 | 0.01× |

Overall **0.14 cycles/byte**. The three op types that dominate stage 2 — MatMul,
Eltwise_Binary, Softmax, together **82.3%** of its output bytes — all sit within 30% of
the bulk rate. The model is sound exactly where stage 2 lives.

The outliers are informative rather than contradictory: `StridedSlice` (3.83×) and the
norms (3.3–4.1×) pay a *granularity* penalty, and `Reshape`/`Transpose`/`Concat` are
sub-bulk because QNN folds them. That is trap #26 restated — granularity, not bytes —
and it applies to small ops, not to the 143 MB score matrix.

---

## 4. Where stage 2's bytes actually go: the score matrix, materialised five times

At 1728 tokens the attention score is `[24,1728,1728]` at 16 bits = **143.3 MB — one
tensor**. Per block the graph writes it five times:

| # | op | writes |
|---|---|---:|
| 1 | `MatMul` q_even · k_evenᵀ | 143.3 MB |
| 2 | `MatMul` q_odd · k_oddᵀ | 143.3 MB |
| 3 | `Eltwise_Binary` add the two halves | 143.3 MB |
| 4 | `Eltwise_Binary` add the mask | 143.3 MB |
| 5 | `Softmax` | 143.3 MB |
| | **per block** | **716.6 MB** |
| | **× 18 blocks** | **12.9 GB** |

That is **74% of stage 2's entire 17.56 GB of op output**, and the census confirms it
exactly rather than approximately: `MatMul` totals 5255.3 MB = 18 × (2 × 143.3 + 5.3 for
the A·V product); `Softmax` totals 2579.9 MB = 18 × 143.3 to the byte.

### The RoPE fix was right for stage 0 and wrong for stage 2

Session 2's rewrite replaced one full-width `MatMul` with **two half-width ones plus an
add**, to avoid `torch.stack(dim=-1)` — the element interleave that was 28.7% of stage 0
(trap #26). It was a clear win there: −30.6%, 248.3 → 172.2 ms.

But that trade **buys a fixed saving and pays a cost proportional to the score matrix**,
and the score matrix is quadratic in sequence length. The interleave it avoids is linear.

| | stage 0 | stage 2 |
|---|---:|---:|
| score matrix | 8.0 MB | **143.3 MB** |
| interleave avoided (per block) | 2.5 MB | 10.6 MB |
| extra score traffic paid (per block) | 16.0 MB | **286.6 MB** |

At 408 tokens that is a good bargain. At 1728 it costs **5.16 GB of extra writes per
inference — 29% of the whole graph.**

---

## 5. The fix: half-split weight permutation, then ONE full-width MatMul

Both goals at once — no interleave *and* no doubled score — by making the even and odd
halves **contiguous offline**, which is follow-up 2 in `phase5-mmdit-latency.md`, never
built.

**Offline**, permute the output channels of `to_q` / `to_k` / `add_q_proj` /
`add_k_proj` (weight rows and bias) and the per-channel weights of `norm_q` / `norm_k` /
`norm_add_q` / `norm_add_k`, per head, by

```
perm = [0,2,4,…,62, 1,3,5,…,63]
```

Then `x[..., 0::2]` becomes `x[..., :32]` and `x[..., 1::2]` becomes `x[..., 32:]` —
contiguous slices, not stride-2 gathers.

**In the graph**, rotate the two contiguous halves and concatenate them back:

```python
x1, x2 = x[..., :32], x[..., 32:]
q = torch.cat([cos*x1 - sin*x2, sin*x1 + cos*x2], dim=-1)   # [S, h, 64]
score = torch.matmul(q, k.transpose(-1, -2))                # ONE full-width MatMul
```

This is exact, not an approximation:

* after the permutation `new[..., :32]` **is** `old_even` and `new[..., 32:]` **is**
  `old_odd`, so the concatenated vector is `cat(q'_even, q'_odd)`;
* `⟨cat(a,b), cat(c,d)⟩ = ⟨a,c⟩ + ⟨b,d⟩`, which is the identity the two-MatMul form was
  already relying on — read in the other direction;
* `V` is **not** permuted, so `softmax·V` and `to_out` see the original channel order and
  are untouched;
* RMSNorm normalises over the last axis by mean-of-squares, which is permutation
  invariant, so permuting its per-channel weight alongside is sufficient;
* `rope_cos`/`rope_sin` are indexed by **pair index** `i ∈ [0,32)` and are unchanged —
  so **every graph input is untouched and the existing calibration sets stay valid.**

The reassembly concat is on the innermost axis, which is what made the old interleave
expensive — but it now copies **32-wide contiguous runs (64-byte blocks)** rather than
alternating single 2-byte elements. Same op, ~32× the granularity.

### Projected saving

Per block: −1 `MatMul` (143.3 MB) − 1 `Eltwise_Binary` (143.3 MB) + 2 concats (10.6 MB)
= **−276.0 MB**; × 18 = **−4.97 GB**, i.e. −28.3% of stage 2's output bytes.

| stage | now | projected | Δ |
|---|---:|---:|---:|
| 0 | 169.2 ms | ~146 ms | −13% |
| 1 | 375.8 ms | ~307 ms | −18% |
| 2 | **1851 ms** | **~1327 ms** | **−28%** |

On the video that is **MMDiT 14.4 s → ~10.5 s, whole video 23.5 s → ~19.6 s**, before
counting the 72 stride-2 `StridedSlice` gathers this also deletes (191 MB at the measured
3.83× granularity penalty, ~77 ms more on stage 2).

### It should help ACCURACY too, which is the part worth testing

Stage 2 is the worst module at 20.01 dB against a 24 dB target. The current form
quantises **two** half-scores to 16 bits and then quantises their **sum** again; the
single full-width MatMul accumulates all 64 products and quantises **once**. That removes
two rounding stages from the graph's largest tensor.

The roadmap says stage 2 "is one problem, not two". This change is one fix aimed at both,
which is exactly why it should be measured on both axes when the device is back.

## Built, and the graph came out as designed

**Host equivalence, against the stock model** (`export_mmdit_stage.py --unit 6 --stage 2`):

```
[check] unit 6 stage 2   1600 img tok  (1,16,1,40,64)  max|diff|=7.629e-06  SNR=126.8 dB
```

126.8 dB is the fp32 floor — arithmetically identical, which is the correct outcome for a
pure layout change.

**Graph structure, from the converted `net.json`** (`work/device/score_traffic.py`,
`stage_census.py`):

| | `mmdit_s2f` | `mmdit_s2fs` | |
|---|---:|---:|---|
| score materialisations | 90 (**5.00**/block) | 54 (**3.00**/block) | as designed |
| — MatMul | 36 | 18 | one contraction, not two |
| — Eltwise_Binary | 36 | 18 | the half-score add is gone |
| — Softmax | 18 | 18 | unchanged |
| score traffic, 16-bit-normalised | 12,899.5 MB | **7,739.7 MB** | **0.600×, −5,159.8 MB** |
| Concat | 54 | 90 | +36, the two half-joins per block |
| Transpose | 125 | 89 | −36, a bonus: q/k/v not qe/qo/ke/ko/v |
| **GMAC** | **1043.82** | **1043.82** | **identical — same FLOPs** |
| nodes | 1730 | 1640 | |

−5,159.8 MB is exactly the predicted 2 × 143.3 MB × 18 blocks.

Whole-graph projection, A16: 17,559.7 − 2,579.4 − 2,579.4 + 191.2 = **12,592 MB**, i.e.
**−28.3%**. At the measured 0.1054 ms/MB that is **1851 ms → ~1327 ms**, −524 ms per call
and **−3.1 s** on the 49-frame video.

> Compare like with like. The layout build is **fp32** and the shipping build is **A16**,
> so raw byte totals differ ~2× for reasons unrelated to the change — the first run of
> `score_traffic.py` reported "1.200× score traffic" for a graph that had in fact cut it
> by 40%. The tool now normalises to 16-bit and says so when the widths differ.

### The transpose/compute ratio gets WORSE, and that is correct

`analyze_net.py` is mandatory before device time, and on the new graph it reports
**0.11 against the old graph's 0.06**. That looks like a regression and is not one.
Normalising the fp32 layout build to 16-bit:

| | `mmdit_s2f` | `mmdit_s2fs` |
|---|---:|---:|
| Transpose output bytes | 392.53 MB | **392.53 MB** — identical |
| compute output bytes | 6118.71 MB | **3538.8 MB** — −42% |
| ratio | 0.06 | 0.11 |

Layout traffic did not move at all: two transposes of `[1728,24,64]` carry exactly the
volume the four `[1728,24,32]` ones did. The **denominator** shrank, by 2579.9 MB — which
is the removed MatMul (18 × 143.3 MB) to within 0.5 MB.

> **The ratio is a smell test, not a cost.** It rises whenever you delete compute, which
> is the thing you were trying to do. Trap #12 already says to convert bytes to
> milliseconds before panicking; this is the case that makes it bite. Read the absolute
> Transpose bytes, not the ratio, whenever the graph's arithmetic has changed.

Whole-graph output bytes, from the census normalised to 16-bit: **17,559.7 → ~12,496 MB,
−28.8%**, against the −28.3% predicted from first principles. At the measured
0.1054 ms/MB that is **1851 → ~1317 ms**, −534 ms per call and **−3.2 s** on the video.

### The one residual risk

The 36 added `Concat` ops are on the **innermost** axis — the same axis that made the old
`torch.stack(dim=-1)` interleave cost 28.7% of stage 0. They should be far cheaper (32-wide
contiguous runs, 64-byte blocks, versus alternating single 2-byte elements), and they move
only 191 MB against the 5,160 MB removed. But QNN folds an *outermost*-axis concat into its
producers and these will not fold, so they are not free. Even at the worst granularity
penalty measured anywhere in this graph (`StridedSlice`, 3.83× bulk) they cost ~76 ms
against ~524 ms saved. **Check `Concat` in the first detailed profile of the new binary.**

## The shipping W8A16 build — `mmdit_s2fs_v79.bin`

Converted 2026-08-23, **2:28:21** wall clock (s2f took 2:52:07), 1,573,344,888 B.
Both graphs are now A16, so these compare directly with no dtype normalisation:

| | `mmdit_s2f` | `mmdit_s2fs` | |
|---|---:|---:|---|
| score materialisations | 90 (5.00/block) | **54 (3.00/block)** | as designed |
| score traffic | 12,899.5 MB | **7,739.7 MB** | **0.600×, −5,159.8 MB** |
| **total op output bytes** | **17,559.7 MB** | **12,591.0 MB** | **0.72×, −28.3%** |
| GMAC | 1043.82 | **1043.82** | identical |
| float-fallback tensors | 0 | **0** | no fallback |
| Transpose bytes | 392.53 MB | 392.53 MB | identical |
| nodes | 1730 | 1676 | |

**12,591.0 MB against 12,592.1 MB predicted from first principles — within 1 MB.**

At the measured 0.1054 ms/MB that is **1851 ms → ~1327 ms**, −524 ms per call and
**−3.1 s** on the 49-frame video (23.5 → ~20.4 s).

Trap #27 is unchanged and still expected: 36 MatMuls carry an 8-bit dynamic operand (K,
V), proportionally the same as s2f's 54. That remains a dead end — `DYN16=1` buys
+0.01 dB for +18.6% latency.

## All three stages, built 2026-08-23

The fix is not stage-2-specific — it applies wherever the score is contracted, so all
three were rebuilt. Every graph input is unchanged, so each stage's calibration set was
simply rebuilt from the kept raw captures (~30–55 s) and the 300-sample Table 9 count
kept.

| stage | binary | output bytes | Δ | score traffic | latency now | projected |
|---|---|---:|---:|---:|---:|---:|
| 0 | `mmdit_s0fs_v79.bin` 1454.5 MB | 1808.2 → **1565.7 MB** | −13.4% | 719.1 → 431.5 MB | 169.2 ms | **~147 ms** |
| 1 | `mmdit_s1fs_v79.bin` 1462.5 MB | 3550.6 → **2896.7 MB** | −18.4% | 1814.0 → 1088.4 MB | 375.8 ms | **~306 ms** |
| 2 | `mmdit_s2fs_v79.bin` 1500.5 MB | 17559.7 → **12591.0 MB** | −28.3% | 12899.5 → 7739.7 MB | 1851 ms | **~1327 ms** |

Converter wall times 35:16 / 47:25 / 2:28:21 (against 38:32 / 48:43 / 2:52:07 before).
GMAC is **identical** in every stage — 214.97 / 351.33 / 1043.82 — confirming the FLOPs
never moved.

**Score traffic falls to exactly 0.600× in all three stages.** That is structural: 3 of 5
materialisations survive regardless of sequence length. What varies is how much of each
graph the score dominates — 40% of stage 0 but **74% of stage 2** — which is why the same
change is worth −13% at 408 tokens and −28% at 1728.

Each build landed within **1 MB** of the byte count predicted from first principles
(1565.7 vs 1565.7 · 2896.7 vs 2896.4 · 12591.0 vs 12592.1).

**Whole pipeline, projected: MMDiT 14.4 s → ~10.7 s, video 23.5 s → ~19.8 s.**

Padding was verified transparent for all 3 stages × 6 units through their envelopes
(`--pad-check`): 121.6–127.1 dB, `max|diff| ≤ 1.14e-05` — the fp32 floor.

## Status

**Built and statically verified; NOT measured on device.** Every number above is a byte
count from `net.json` plus the bytes→ms fit. The fit is validated on stage 0 and across
the three stages, but the ms figure for *this* graph is a projection. The plausible band,
driven by how QNN executes the 36 new innermost-axis Concats:

| Concat behaves like | stage 2 |
|---|---:|
| bulk bandwidth (expected) | ~1320 ms, −29% |
| worst granularity penalty in this graph (3.83×) | ~1390 ms, −25% |
| the pathological old 2-byte interleave | ~1575 ms, −15% |

Read `Concat` first in the new binary's detailed profile, and read the deploy SNR in the
same run — the fused contraction quantises the score once where the split form quantised
it three times, so 20.01 dB may move too.

## Trap to add

> **40. A latency fix that trades a LINEAR cost for a QUADRATIC one silently inverts at
> longer sequences.** The RoPE rewrite swapped one full-width attention MatMul for two
> half-width ones plus an add, to avoid an element interleave. The interleave it removes
> scales with sequence length; the extra score traffic it adds scales with sequence
> length *squared*. Measured as a 30.6% win on stage 0 (408 tokens), the same code costs
> 5.16 GB of extra writes on stage 2 (1728 tokens) — 29% of that graph. **Re-price every
> shape-dependent optimisation at the largest shape you actually ship**, not at the one
> you profiled.
