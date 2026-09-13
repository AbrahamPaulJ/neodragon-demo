# Phase 5 - the MMDiT accuracy gap: FIXED for stage 0 (2026-08-23)

> ## WHERE THIS STANDS
>
> | stage | before | after | target | latency | paper |
> |---|---:|---:|---:|---:|---:|
> | 0 | 19.48 dB | **27.89 dB** | 29.0 | 169.2 ms | 104.7 |
> | 1 | -0.13 dB (broken) | **23.99 dB — MET** | 22.0 | 375.8 ms | 218.3 |
> | 2 | not built | 20.01 dB | 24.0 | 1851 ms | 938.3 |
>
> All three stages are on device and computing correctly. Accuracy falls with sequence
> length (408 / 648 / 1728 tokens), which is residual accumulation, not a new fault.
>
> Stage 0's gap was the AdaLN modulation path, and **hoisting `time_text_embed` to the
> host is the whole fix** - the chunk split (trap #31) contributes ~0.3%, see the section
> below. Latency is unchanged: 171.5 -> **169.2 ms**, min-of-N interleaved.
>
> Stage 1 turned out to have a **second, unrelated bug**: QNN executes a rank-2
> multi-input `Concat` above ~320 rows as eight round-robin chunks, which scrambles the
> token order under the host-supplied RoPE and mask. Replacing that one `torch.cat` with
> a pad-and-sum (`_join`) moved stage 1 from -0.13 dB to **23.99 dB**. Stage 0 sits at
> 280 rows and was safe **by luck, not by design**. Full account, with the two-minute
> reproducer that pins the threshold: `notes/2026-08-23-adaln-fix.md`.

---


> ## THE ANSWER
>
> **It is the AdaLN modulation path.** Measured from the device's own intermediates,
> inside block 0:
>
> | path | tensor | SNR |
> |---|---|---|
> | image tokens | `_Concat_output_0` (patchified) | **54.05 dB** |
> | image tokens | `_norm1_norm_LayerNormalization_output_0` | **54.96 dB** |
> | modulation | `_norm1_Slice_{0..5}` (the `chunk(6)`) | **13.90 - 34.85 dB** |
> | modulation | `_norm1_context_Slice_{0..5}` | **11.97 - 27.50 dB** |
> | they meet | `_norm1_Add_2_output_0` | **16.51 dB** |
>
> A 55 dB image stream is destroyed by a 14 dB modulation vector, in block 0, before
> anything else happens. Two causes, both fixable in the export:
>
> 1. **The six AdaLN chunks share one encoding**, sized by the largest (18.41) while the
>    smallest is 1.68 - trap #31. Fix: six separate Linear ops.
> 2. **`time_proj`'s `Sin` is quantised at its argument**, which is encoded over
>    `[0, 1000]` at 16 bits (0.0153 rad step, ~47 dB ceiling; measured 46.41 dB). Fix:
>    hoist `time_text_embed` to the host and pass `temb` in - trap #20's argument.
>
> Full evidence and the next step: **`notes/2026-08-23-HANDOFF.md`**, raw tables in
> `work/device/dnam_s0_snr.txt` and `work/device/dblk_s0_snr.txt`.
>
> **Seven hypotheses died getting here** - padding, outlier ranges, 8-bit K/V, the -100
> mask, residual accumulation, W8 weights, clipping. The history below is kept so nobody
> re-runs them.

---

## Which of the two causes actually matters (2026-08-23)

The two causes above are not equal partners. **Cause 2 is the entire effect; cause 1
contributes about 0.3% of it** - and this is settled by arithmetic on data already on
disk, with no conversion and no device time.

Take the session-5 dump and turn each tensor's SNR back into an **absolute** error rms
(`err = rms(device) x 10^(-SNR/20)`):

| tensor | SNR dB | rms | **err rms** |
|---|---:|---:|---:|
| `_norm1_linear_Gemm_output_0` | 29.35 | 1.157 | **0.0394** |
| `_norm1_Slice_0` (shift_msa) | 16.93 | 0.255 | **0.0363** |
| `_norm1_Slice_1` (scale_msa) | 28.24 | 1.082 | **0.0419** |
| `_norm1_Slice_2` (gate_msa) | 20.44 | 0.358 | **0.0341** |
| `_norm1_Slice_3` (shift_mlp) | 13.90 | 0.161 | **0.0325** |
| `_norm1_Slice_4` (scale_mlp) | 26.94 | 1.004 | **0.0452** |
| `_norm1_Slice_5` (gate_mlp) | 34.85 | 2.373 | **0.0429** |
| `_norm1_context_linear_Gemm_output_0` | 22.24 | 0.571 | **0.0441** |
| `_norm1_context_Slice_{0..5}` | 11.97 - 27.50 | 0.150 - 1.070 | **0.0378 - 0.0465** |

The signal rms spans **16x** (0.150 to 2.373) and the SNR spans **23 dB**, while the
absolute error is flat at **0.032 - 0.046** across all fourteen tensors. That is the
signature of a **constant additive error inherited from upstream**, not of an encoding
problem in the chunks.

And the chunk encoding can be sized directly. `net.json` gives the parent Gemm
`scale = 4.69191e-4` (shared by all six slices, exactly as trap #31 says), so the
quantisation noise the sharing imposes is:

    4.69191e-4 / sqrt(12) = 1.354e-4

**291x to 343x smaller than the error that is actually there.** Trap #31 is a real
mechanism and it is really present in this graph - it just is not doing any damage here,
because something 300x bigger arrives first.

That something is the conditioning path, and the dump traces it end to end:

| tensor | SNR dB |
|---|---:|
| `time_proj_Sin` / `Cos` (quantised at an argument encoded over `[0, 1000]`) | 46.41 / 47.32 |
| `timestep_embedder_linear_1_Gemm` | 39.92 |
| `timestep_embedder_act_Mul` (the SiLU) | **28.50** |
| `timestep_embedder_linear_2_Gemm` | 45.94 |
| `time_text_embed_Add` (= `temb`) | 43.14 |
| `norm1_silu_Sigmoid` | 24.71 |
| `norm1_linear_Gemm` | 29.35 |

**So the fix that matters is hoisting `time_text_embed` (and the SiLU with it) to the
host.** The six-way Linear split is kept because it is free - identical FLOPs, it deletes
the 214 `StridedSlice [1,1536]` nodes that `docs/phase5-mmdit-latency.md` measures at
5.0% of the graph, and it removes the shared encoding as a *future* constraint once the
300x-larger error is gone. But it is insurance, not the cure.

> **The general lesson, and it is trap #28 pointing the other way.** A confirmed
> mechanism is not a confirmed cause - and *magnitude* is what separates them. Trap #31
> was found by reading `net.json` and confirmed by the dump; both were right that the six
> chunks share one encoding. Neither said how much that costs. One line of arithmetic on
> the numbers already printed - turn SNR into an absolute error and compare it against
> the encoding's own step - answered it before a single minute of conversion was spent.

---


**Target (paper Table 9): 29 dB for stage 0.** First build measured **19.47 dB**
(16 held-out cases, mean; min 12.05 dB), no NaN/inf, outputs distinct per case — so the
graph is *correct*, just imprecise.

> ## Result, measured 2026-08-21
>
> | | baseline | 16-bit K/V | verdict |
> |---|---|---|---|
> | deploy SNR | 19.47 dB | **19.48 dB** | **+0.01 — nothing** |
> | latency (min-of-3) | 248.3 ms | **294.5 ms** | **+18.6% — worse** |
> | conversion wall | 35:04 | 42:26 | +21% |
>
> The K/V precision problem below is **real** and the fix does exactly what it claims —
> `matmul_bits.py` confirms all 36 MatMuls moved to `u16 × u16`. It simply **does not
> explain the accuracy gap.** It is now **off by default** (`DYN16=1` to re-enable) and
> should be treated as ruled out, alongside padding and outlier ranges.

This is the third hypothesis to die by measurement on this module. Record it so nobody
spends another 42-minute conversion on it.

## The diagnosis (correct, but not the cause)

QNN treats a MatMul's **second operand as a "weight"**, and 16-bit *dynamic* weights are
off by default. In an attention block the second operands are `Kᵀ` and `V` — pure
activations. So they were quantised to **8 bits in all 18 blocks** while everything
around them ran at 16.

The converter says so during conversion, and it is easy to scroll past:

> `mixedPrecisionForWeights: The override provided for the following weight:`
> `/Transpose_71_output_0 would not be honoured. Since 16 bit dynamic weights are not`
> `supported by default. Kindly use the flag --use_dynamic_16_bit weights`

`work/device/matmul_bits.py` makes it explicit — on the first build:

```
   255x  FullyConnected   in u16 x u8* x s32*   -> out u16
    36x  MatMul           in u16 x u8           -> out u16      <-- no star: an ACTIVATION
     3x  Conv2d           in u16 x s8* x s32*   -> out u16
  MatMuls with an 8-bit DYNAMIC operand: 36
```

36 = 18 blocks × 2 (`q @ kᵀ` and `scores @ v`). The `*` marks a static tensor; the second
MatMul operand has none — it is not a weight, the converter just classifies it as one.

This is invisible to every other tool in the repo: `analyze_net.py` sees clean
transposes, `analyze_encodings.py` ranks ranges but not bitwidths, and there is zero
float fallback. Only the **operand dtypes** show it.

## It is a converter default, not a hardware limit

HTP V79 lists **u16 × u16 → u16** among its 21 MatMul kernels (`htp.json`), same for
FullyConnected. So the 8-bit operand was never forced by the backend. (Same static check
that killed the MaskedSoftmax idea in trap #25 — reading `htp.json` is free and settles
these questions in minutes.)

## The flags, and the quoting trap

```
--use_dynamic_16_bit_weights
--restrict_quantization_steps "-0x8000 0x7F7F"
```

- `--use_dynamic_16_bit_weights` is real but **hidden** — `help=argparse.SUPPRESS` at
  `qnn_quantizer.py:156`, `default=False`, honoured for every backend except `lpai`
  (`qnn_quantizer.py:392`). It will not appear in `--help`.
- `--restrict_quantization_steps` is the documented companion: its help says *"This
  argument is required for 16-bit Matmul operations."* Only honoured when the param
  quantizer is symmetric **or** per-channel/per-row (`qnn_quantizer.py:399`) — we pass
  `--use_per_channel_quantization`, so it applies.

**The quoting trap is the reusable lesson.** `validation_utils.two_hex:112` does
`hex_pair.split()` and requires exactly two tokens, so the pair must arrive as **one
argv entry** `-0x8000 0x7F7F`. The script built its quantiser flags as a single
word-split string, which would have passed the literal quote characters through as part
of the argument. `QFLAGS` is now a **bash array**, expanded `"${QFLAGS[@]}"`.

(argparse accepts a value beginning with `-` here only because it contains a space —
`_parse_optional` returns `None` for any arg string containing one.)

## The structural change did land — it just didn't help

Verified statically on the real 300-sample build before any device time:

| | baseline | with the flags |
|---|---|---|
| MatMul | `u16 × u8` ×36 | **`u16 × u16` ×36** |
| MatMuls with an 8-bit dynamic operand | 36 | **0** |
| FullyConnected | `u16 × u8* × s32*` ×255 | **unchanged** |
| Conv2d | `u16 × s8* × s32*` ×3 | **unchanged** |
| `mixedPrecisionForWeights` warning | present | **gone** |

The FullyConnected row was the one worth checking: the risk with a global
`--restrict_quantization_steps` was that it would drag the **static** weights off 8-bit
and quietly leave the paper's W8A16 recipe. It did not — only the dynamic operands moved.

```
py -3.10 work/device/matmul_bits.py work/device/mmdit_s0_net.BASELINE.json \
                                    work/device/mmdit_s0_net.json
```

## Why it changed nothing — and the caveat

Per-case SNR is essentially unchanged, case by case, not just in the mean (baseline
min 12.05 → 12.17). Whatever dominates the error is **not** K/V operand precision, and
it swamps it by enough that a full 8→16 bit move on 36 tensors is invisible.

**The caveat that keeps this from being deleted outright:** a non-effect at the current
error floor is not proof of irrelevance. If the dominant source is found and fixed, 8-bit
K/V could become the next binding constraint. That is why the flags are kept behind
`DYN16=1` rather than removed — but re-measure **both** axes, because the latency cost is
real and large.

## Ruled out so far — do not re-spend a conversion on these

1. **Padding.** Per-case SNR correlates with padding fraction (r = −0.66) and the worst
   cases are unit 1 (200 of 280 image tokens padded), but
   `work/audit/mmdit_residual_scan.py` shows padded tokens peak **lower** than real ones
   — padded/real 0.59×. Second-order.
2. **Outlier-driven activation ranges.** `analyze_encodings.py` finds late residual adds
   with range 55,130 against a graph median of 24.47 — looks exactly like trap #3. But
   max/p99.99 on the image stream is only **1.1–1.5×**: the distribution is genuinely
   wide, not outlier-driven. **Percentile/MSE activation calibration would reclaim
   almost nothing.**
3. **8-bit K/V** — this document. Fixed at the graph level, +0.01 dB, −18.6% speed.
4. **The finite `-100` attention mask inflating the score encodings.** Plausible, and
   free to check from `net.json` alone — it is not the cause:

   | `[24,408,408]` tensor | encoding range | width |
   |---|---|---|
   | raw MatMul score (pre-mask) | −45.9 … +99.3 | 145 |
   | after the mask add | −144.1 … +99.3 | 243 |

   1.68× wider, i.e. **0.75 effective bits** — nowhere near the ~10 dB deficit. And the
   masked entries become `exp(−100) ≈ 0` in the softmax, so their precision never
   reaches the output. Softmax outputs are all encoded [0, 1] exactly as expected.

The residual really does grow 197 → 27,922 (140×) across the 18 blocks on the text
stream. But per-tensor SQNR from (range, std) is 60–80 dB everywhere, which cannot
produce a 19 dB output. **That mismatch is still unexplained and is now the main lead:**
if every tensor is individually good but the output is not, the error is either
accumulating across the residual chain or concentrated in an op the static analysis does
not model.

## THE DECISIVE RESULT: faithful quantisation cannot reproduce the device

Measured on the host, in fp32 torch, replaying the converter's own encodings. No device,
no NDK, no conversion — each group quantised alone, everything else exact.

| what is quantised | deploy SNR |
|---|---|
| residual block boundaries only (real `net.json` encodings) | **53.33 dB** |
| AdaLN / modulation elementwise chain, A16 (246 points) | **52.07 dB** |
| W8 per-channel weights only (249 Linear, 1498 M params) | **43.14 dB** |
| weights + all Linear outputs + softmax, A16 | **43.11 dB** |
| **the actual device** | **19.48 dB** |

Sanity: the fp32 rerun driven by the device's own input raws reproduces the stored
`nref` reference at **130.03 dB**, so inputs and reference are consistent and the
comparison is sound.

**Nothing reproduces 19.48 dB.** Combining the groups is dominated by the worst of them
(~42.6 dB). The gap to the device is ~23 dB, and it is *not* explained by W8 weights,
A16 activations, the residual chain, the modulation chain, softmax encoding, the mask,
padding, or calibration width.

### Calibration is tight, so it is not that either

`work/device/enc_vs_actual.py` compares each block output's calibrated encoding width
against the width actually observed:

```
image-stream enc/act ratio: min 1.13x  median 1.21x  max 2.47x
```

A median of 1.21× is a well-fitted range. This independently confirms the earlier
"percentile/MSE calibration would buy nothing" conclusion, from a different direction.

### Where the requantisation points actually are

All 2374 nodes emit a quantised tensor, but **1127 of them preserve their input's
encoding exactly** and therefore contribute **zero** error:

| op | count | encoding |
|---|---:|---|
| Reshape | 670 | preserved |
| StridedSlice | 324 | preserved |
| Transpose | 121 | preserved |
| **Eltwise_Binary** | **629** | requantises |
| FullyConnected | 255 | requantises |
| LayerNorm + RmsNorm | 144 | requantises |
| Convert / Concat / Softmax / MatMul | 158 | requantises |

**1247 genuine requantisation points.** The ablations above exercise the two largest
groups and still land 23 dB short.

### What this means

The device is doing something worse than a correct W8A16 execution of this graph. The
remaining candidates are no longer about *calibration* or *bit width choices* — they are
about what the hardware actually computes:

1. **Internal precision of specific ops.** The ablations quantise op *outputs*; they
   cannot simulate an op whose *internal* accumulation is narrower than assumed. The
   norms are the prime suspect — 144 requantisation points, and trap #3 already
   established that HTP runs float graphs in fp16 with no fp32 upcast in the layer norm.
   The text-stream residual reaches **52,000** by block 14 (`enc_vs_actual.py`), and a
   norm computing `mean(x²)` on that would need 2.7e9 of headroom — fp16 tops out at
   65504.
2. **A deployed-graph mismatch** that the fp32 export does not have.

Either way the next step is unchanged and now much better motivated: **look at the
actual intermediate values on the device.** We no longer need it to choose between
plausible quantisation stories — those are all eliminated — we need it because the
device disagrees with a faithful simulation of its own graph.

## Getting the layerwise dump — the prerequisite

`work/device/find_block_io.py` and `work/device/block_snr.py` are written and tested.
The dump is the blocker, and three routes are already closed:

| route | result |
|---|---|
| `qnn-net-run --set_output_tensors` on device | ❌ *"can only be used with graph prepared online using --model or --dlc_path"* |
| `qnn-net-run --debug` on device | ❌ same restriction; explicitly excludes `--retrieve_context` |
| host CPU backend + the x86 model lib we already have | ❌ `[QNN_CPU] OpConfig validation failed for Transpose` — rank-5 latent transpose unsupported |
| **aarch64-android model lib + `--debug`** | ✅ the way through — needs the **Android NDK, not installed** |

## The tools, all reusable

| file | what |
|---|---|
| `work/device/resid_ablate.py` | replays real `net.json` encodings on the residual stream |
| `work/device/quant_ablate.py` | groups: `weights` / `linear` / `softmax` / `all` |
| `work/device/adaln_ablate.py` | the modulation chain, 246 points |
| `work/device/enc_vs_actual.py` | calibrated width vs observed width, per block |
| `work/device/find_block_io.py` | block boundaries by topology (period shifts every rebuild) |
| `work/device/block_snr.py` | per-block SNR once a dump exists |

The encoding convention, verified against printed ranges:
`real = scale * (q + offset)`, `q` integer in `[0, 2**bits - 1]`.

## The old plan — superseded

## The next thing to do — localise, stop guessing

Three hypotheses have now died. Stop proposing mechanisms and **measure where the error
enters**:

1. **Per-layer intermediate dump.** `qnn-net-run --debug` writes every intermediate
   tensor; compare block-by-block against onnxruntime fp32 on the same input. This
   converts "19 dB somewhere" into "block N, tensor T". It is the only approach that
   cannot be wrong. Cost: one device run plus a host comparison, no conversion.
2. The SDK ships `qti/aisw/accuracy_debugger`, which automates exactly this
   golden-vs-device layerwise comparison. Prefer it over hand-rolling.
3. Only after localisation: `--use_per_row_quantization`, `--algorithms cle`, or
   residual scaling (trap #3's Phase 1 fix folded into weights).

**The mask constant was the obvious next suspect and is already dead** (ruled out #4
above) — checked straight out of `net.json` for zero cost. Worth remembering as a
method: an encoding-range hypothesis can almost always be settled by reading the
converter's own encodings before building anything.

## Trap to record

> **27. QNN quantises a MatMul's *second operand* as a weight, so attention K and V land
> at 8 bits in a W8A16 graph.** Pure activations at half the intended precision in every
> block, invisible to every other tool — use `work/device/matmul_bits.py`, which prints
> operand widths and stars the static tensors. `htp.json` lists `u16 × u16 → u16`, so it
> is a converter default, not a hardware limit; the hidden
> `--use_dynamic_16_bit_weights` plus `--restrict_quantization_steps "-0x8000 0x7F7F"`
> removes it. **On the MMDiT that bought +0.01 dB and cost 18.6% latency** — know the
> mechanism, but do not assume it is your accuracy bug. And build converter flags as a
> **bash array**: `--restrict_quantization_steps` must arrive as one argv entry.
