# Phase 4 — TinyAEHV video VAE decoder

Started 2026-08-14. Target from paper Table 7: **248.9 ms** on 8 Elite Gen4, **0.21 GB** peak
memory. Table 9: W8A16 PTQ, **50** calibration samples, **35 dB** deploy SNR.

## The decoder is much friendlier than the module name suggests

`asymmetric_causal_video_vae/modeling_causal_ops.py` contains `CausalConv3d` with a stateful
`deque` feature cache — unexportable as written. **None of it is on the decoder path.** That
machinery belongs to the *encoder*; "asymmetric" is precisely the point (Table 7: encoder
1206.5 ms vs decoder 248.9 ms).

The decoder (`decoder.py`, TAEHV, adapted from `madebyollin/taehv`, MIT) is:

- `nn.Conv2d` throughout — `conv()` is a 3×3 Conv2d, `TPool`/`TGrow` are 1×1 Conv2d
- `MemBlock` = 3 convs + ReLU with the previous frame concatenated on channels
- `nn.Upsample(scale_factor=2)` ×3 for space, `TGrow` ×3 for time (channels→frames reshape)
- `Clamp` = `tanh(x/3)*3`

The only 3-D op is `post_quant_conv`, a 1×1×1 `CausalConv3d` (so `time_pad = 0`, no padding,
no cache). 9.91 M parameters.

`apply_model_with_memblocks(parallel=True)` is fully traceable: `mem` is just a
shift-by-one of `x` built with pad+slice, all static.

Latent scale/shift (`VAE_SCALE_FACTOR` etc.) is applied **outside** `decode()` in
`generation_utils.py`, so it stays out of the graph.

## Exported graph

`[1,16,T,40,64] → [1,3,8T−7,320,512]` (`frames_to_trim = 2³−1 = 7`).

| T | out frames | matches |
|---|---|---|
| 1 | 1 | Fig 13 first-frame path `16×1×40×64 → 3×1×320×512` |
| 7 | 49 | Fig 13 video path `16×7×40×64 → 3×49×320×512` |

**134 nodes** after folding, ONNX 37.8 MB. Op histogram — nothing exotic:

```
Conv×36, Relu×29, Reshape×23, Pad×10, Slice×10, Concat×9,
Add×9, Resize×3, Transpose×2, Div×1, Tanh×1, Mul×1
```

ONNX vs torch at T=1: **112.8 dB**, zero unsafe constants.

## The 49-frame decode cannot be one graph

T=7 exports fine but its intermediates hit **2,348,810,240 bytes** — host verification OOMs.
Peak activation is `8T × 64 × 320 × 512 × 4 B`, i.e. **335 MB per latent frame** in fp32.

Table 7 reports the deployed decoder at **0.21 GB**. T=1 in fp16 is ~168 MB of activation
plus ~20 MB of fp16 weights ≈ **0.19 GB** — a close match. So the reference deployment is
almost certainly the **single-latent-frame graph invoked repeatedly**, not a 49-frame
monolith. Consistent with 248.9 ms being a per-invocation figure.

Note T=1 wastes 7/8 of its work to the trim. A production video decoder would pass the 9
`MemBlock` previous-frame states in and out as explicit graph tensors instead — the same
"fixed-size explicit state" pattern §4.1 describes for the MMDiT's autoregressive history.
That is the real Phase 4 design decision and it is not yet made.

## `--preserve_io layout` is mandatory for any conv model

Without it the converter rewrote the 5-D input to **channel-last** while leaving the output
NCDHW:

```
Input   latent  dims=[1, 1, 40, 64, 16]   <-- NDHWC, silently
Output  video   dims=[1, 3, 1, 320, 512]  <-- NCDHW, unchanged
```

Raw files written in torch order then decode to garbage: measured **−0.75 dB**, with outputs
still varying per input so nothing looks obviously broken. Adding `--preserve_io layout`
(the flag the proven `LocalDream/npuconvert/npuconvertv2/convert_inpaint_unet.sh` already
uses, and which was omitted here) restores `latent [1,16,1,40,64]`.

Phase 1 did not need this because DistilT5's inputs are rank-2. **Every remaining module —
SSD1B UNet, VAE, QuickSRNet, MMDiT — has spatial inputs and will need it.**

## VTCM behaviour differs sharply from DistilT5

| | DistilT5 | VAE decoder T=1 |
|---|---|---|
| spill_bytes | 0 | **285,736,960** |
| fill_bytes | 0 | **338,165,760** |
| Graph Optimizations | — | 2.49 s |
| Graph Sequencing | 0.39 s | 5.26 s |
| Parallelization Opt | 0.16 s | 5.22 s |
| context binary | 260 MB | 23.2 MB |

The 335 MB activation against 8 MB of VTCM means heavy spilling, and the compile is ~14 s
versus DistilT5's ~5 s. Whether that spilling dominates the 248.9 ms budget is the open
question — `vtcm_mb` is still at the value inherited from the 8gen2 SD config and has never
been tuned.

## Device results — and the headline finding

With `--preserve_io layout` corrected:

| metric | value |
|---|---|
| SNR vs fp32 reference | **55.52 dB** (55.50 / 55.65 / 55.41) |
| pixel error | mean 0.043 of 255 levels, p99 0.162, max 0.509 |
| nan+inf | 0 |
| **latency (accelerator)** | **10,876,947 µs = 10.88 s** |

Accuracy is excellent — visually lossless. **Latency is 43.7× the paper's 248.9 ms.**

### Float convolution on HTP does not use HMX

The decoder does **205.03 GMAC (410 GFLOP)** to produce a single 320×512 frame at T=1.
Dominated by the full-resolution stages — `decoder.blocks.20` alone is 48.32 GMAC.

| | latency | implied throughput |
|---|---|---|
| ours, float | 10,877 ms | **18.85 GMAC/s** |
| paper, W8A16 | 248.9 ms | **823.74 GMAC/s** |

SM8750's NPU is quoted in the tens of TOPS for int8. Achieving only ~19 GMAC/s means the
matrix engine (HMX) is not being used at all — consistent with float convolution running on
HVX vector units alone. HMX accelerates *quantised* convolution; a float graph cannot reach
it.

> **This changes the plan for every remaining module.** W8A16 is not an optimisation to apply
> after a working float baseline — for conv-heavy modules it is **mandatory before any
> latency number means anything**. Phase 1 was misleading in this respect: DistilT5 is
> matmul-heavy *and* is deployed FP16 in the reference (Table 9), so its float build was
> legitimately comparable. Every other module in Table 9 — VAE enc/dec, SSD1B UNet/Dec, all
> three MMDiT stages, QuickSRNet — is W8A16, so float measurements of those are close to
> meaningless as latency predictions.

### The MAC count also confirms the chunking inference

248.9 ms at 823.74 GMAC/s corresponds to **205 GMAC** — exactly the T=1 figure. Independent
of the 0.21 GB memory argument, this confirms the paper's VAE-Dec number is a
**single-latent-frame invocation**, not a 49-frame decode.

It also shows the padded `parallel=True` T=1 graph wastes 7/8 of its compute: 205 GMAC
produces 8 internal frames of which the trim keeps 1. The explicit-state version would keep
all 8 for the same work — an 8× saving that is a correctness-preserving restructure, not a
tuning knob.

## W8A16 quantisation (Phase 4d)

Calibration data comes from the **real reference pipeline**, run locally on the RTX 3050
6 GB for the first time in this project (`work/pipeline/run_reference.py`). 8 prompts × 7
latent frames = 56 samples in 7.0 min; the converter uses the first 50, matching Table 9.

### Real latents are ~3x wider than N(0,1)

| source | range | std |
|---|---|---|
| `N(0,1)` used for the float functional check | ±~4 | 1.00 |
| **real decoder inputs, 8 prompts** | **−16.34 … +16.24** | **2.71 – 3.10** |

Calibrating on the synthetic samples would have set activation ranges roughly 3× too tight
and clipped hard on real content. Because the float path was already accurate (55.52 dB),
that would have presented as a quantisation-quality problem rather than a
calibration-data problem. **The paper's per-module sample counts (50 / 300 / 500) assume
real pipeline data**; synthetic noise is not a substitute for any module.

### Conversion

`work/qnn/convert_vae_w8a16.sh` — `--act_bitwidth 16 --weights_bitwidth 8
--bias_bitwidth 32 --use_per_channel_quantization --preserve_io layout`.

| | float | **W8A16** |
|---|---|---|
| context binary | 23.2 MB | **12.9 MB** |
| spill_bytes | 285,736,960 | **214,958,080** |
| fill_bytes | 338,165,760 | **251,658,240** |
| converter wall clock | ~30 s | **11 min** (50 calibration passes) |

Spilling drops ~25% with 16-bit activations, though it is far from eliminated.

### Device results — W8A16 vs float

Measured on **held-out** latents 0050–0055 (never seen by the quantiser) against fp32 ONNX
references, matching the paper's definition of deploy SNR.

| | float | **W8A16** | paper |
|---|---|---|---|
| SNR (mean) | 64.18 dB | **31.55 dB** | 35 dB |
| SNR (min) | 63.03 dB | **30.95 dB** | — |
| latency | 10.62 s | **6.10 s** | **248.9 ms** |
| pixel error | — | mean 2.18, p99 10.19, max 31.95 of 255 | — |

Two misses:

1. **SNR is 3.5 dB below target.** The reference hits 35 dB with the same 50-sample budget,
   so quality is being left on the table. Candidates: per-channel activation handling,
   AIMET AdaRound (`--algorithms`), or `--quantization_overrides` with AIMET encodings.
2. **Latency is still 24× off.**

### The HMX hypothesis was wrong

After Phase 4c this document argued the 43.7× float gap was because HMX only accelerates
quantised convolution, and predicted W8A16 would recover roughly 40×.

**It recovered 1.74×** (10.62 s → 6.10 s). Quantisation helped, but it is not the
explanation. Corrected accounting:

| | throughput on 205 GMAC |
|---|---|
| float | 18.85 GMAC/s |
| W8A16 | **33.6 GMAC/s** |
| paper | 823.7 GMAC/s |

Still ~0.15 % of what a ~45 TOPS-class NPU should reach. And it is **not** DDR-bound:
spill+fill is 467 MB per inference, which at 6.10 s is ~76 MB/s — around 0.1 % of LPDDR5X
bandwidth. So neither arithmetic precision nor memory bandwidth accounts for the gap;
something in the graph is on a slow path.


## The 24× gap, diagnosed (2026-08-21)

**It is layout-transpose traffic.** Found statically from the converter's own
`vaedec1q_net.json`, with no device time, using `work/device/analyze_net.py`.

### Two hypotheses eliminated first

- **`soc_model: 69` is correct.** `QnnTypes.h:1872` gives `QNN_SOC_MODEL_SM8750 = 69`. The
  "unverified guess for SM8750" flagged in `notes/2026-08-14-audit-phase1.md` was right.
  (The SDK's own `htp_v2.json` also maps `SM8750 -> v79`, confirming `dsp_arch`.)
- **There is no float fallback.** All 223 tensors are `UFIXED_16` (151), `SFIXED_8` (36),
  `SFIXED_32` (36). Zero `FLOAT_*`. The whole graph really is running quantised, so "some op
  fell off the fast path into float" is dead.

### The measurement

The ONNX has **2** Transposes. The converted QNN graph has **47** — the converter inserted 45.

| | DistilT5 (3.7× off) | VAE dec W8A16 (24× off) |
|---|---|---|
| nodes | 517 | 149 |
| Transpose nodes | 48 | 47 |
| **Transpose output bytes** | **18.87 MB** | **779.80 MB** |
| compute output bytes | 67.11 MB | 671.17 MB |
| **transpose ÷ compute** | **0.28** | **1.16** |
| largest single Transpose | 0.39 MB | **167.77 MB** |

DistilT5 is the control and it is a good one: **near-identical Transpose count, 41× less
data**. Its transposes are attention head permutes on 0.39 MB tensors; the VAE's are
full-resolution feature maps. Node count is not the discriminator — **bytes moved is** — and
41× is the right order of magnitude for a 24× latency gap.

The decoder moves **more bytes through layout conversion than through its convolutions.**
This also explains the spilling (215 MB / 252 MB) that neither precision nor the DDR
bandwidth argument could account for: transposes materialise full tensors that cannot live
in 8 MB of VTCM, and a 5-D NCDHW↔NDHWC permute has a tiny innermost stride, so it defeats
both vectorisation and tiling. It is slow *per byte*, not merely voluminous — which is why
the earlier "467 MB at 6.10 s is only 76 MB/s, so we are not DDR-bound" reasoning was
correct on bandwidth and still missed the cause.

### The mechanism, in five lines of graph

```
Conv2d                 [4, 320, 512, 128]   NHWC -- HTP native, fine
Transpose              [4, 128, 320, 512]   167.77 MB  ->  forced to NCHW
Reshape                [8,  64, 320, 512]   TGrow: 128ch -> 2 frames x 64ch
Transpose              [8, 320, 512,  64]   167.77 MB  ->  back to NHWC
Conv2d                 [8, 320, 512,  64]   NHWC again
```

335 MB of pure data movement to express one channels→frames split, and it happens at every
`TGrow` stage plus every frame-trim `Slice`.

**Why it is not a tuning knob.** HTP runs convolution channel-last; batch is outermost in
both layouts. `TGrow` moves frames *into the batch axis*, which is a free view in NCHW
(C is adjacent to N) and a genuine scatter in NHWC (C is innermost, maximally far from N).
So no converter flag, weight permutation or `vtcm_mb` value makes this cheap. `--preserve_io
layout` is not the culprit either — it is required for correctness (trap #7) and pins only
the I/O; these transposes are interior.

This is the conv-model analogue of **trap #1**: the paper's §4.1 step 3 reduces RoPE's 6-D
tensors because "high dimensional inputs means more complicated *tiling* which usually ends
up penalizing performance." Same phenomenon, same fix shape — restructure so the layout
never has to round-trip.

### The fix: the explicit-state decoder, promoted to critical path

Stop moving frames into the batch axis at all. Decode **one frame per invocation** and carry
the 9 `MemBlock` previous-frame states as explicit graph inputs and outputs — the same
"fixed-size explicit state" pattern §4.1 describes for the MMDiT's autoregressive history,
and exactly the restructure this document already identified as "the real Phase 4 design
decision". It was listed as a cleanup. It is now the fix.

One rewrite removes **two independent multipliers**:

1. the ~780 MB of transposes (the channels↔frames reshapes disappear), and
2. the 8× compute waste (205 GMAC currently produces 8 internal frames and the trim keeps 1).

Do not promise a landing point from this — the two effects are not cleanly separable and the
paper's own 248.9 ms appears to correspond to the full 205 GMAC. Build it, then measure.

### What this means for Phase 5

Probably **less bad than feared.** The MMDiT is a transformer, so DistilT5's profile
(ratio 0.28, 18.87 MB) is the relevant analogue, not the VAE's. Extrapolating 24× onto the
MMDiT was pessimistic. But the same failure mode is exactly what trap #1's rank-6→4 rewrite
prevents, so **apply the RoPE fix before the first MMDiT conversion**, not after — and run
`analyze_net.py` on the result before ever pushing it to the device. The conv-heavy SSD1B
UNet is the module most likely to inherit this problem.

## Phase 4f -- the streaming decoder (built 2026-08-21)

`work/export/export_vae_decoder_stream.py`. Follows upstream's `parallel=False`
streaming order instead of `parallel=True`, so time never enters the batch axis.

| | T=1 parallel (old) | streaming (new) |
|---|---|---|
| input rank | 5-D `[1,16,1,40,64]` | **4-D `[1,16,40,64]`** |
| graph inputs | 1 | 10 (latent + 9 MemBlock states) |
| video frames produced | 8, **7 discarded** | **8, all kept** |
| invocations for 49 frames | 49 | **7** |
| ONNX nodes | 134 | 233 |
| ONNX `Reshape` | 23 | **0** |
| ONNX `Pad` | 10 | **0** |
| ONNX `Transpose` | 2 | **0** |

Three changes do the work:

1. **`TGrow` becomes conv + channel chunk.** Upstream does `conv -> reshape(-1,C,H,W)`
   then the consumer does `.view(N, stride*C, H, W).chunk(stride, 1)`. The two reshapes
   cancel, so it is really a 1x1 conv followed by a slice on the channel axis --
   contiguous on the innermost axis in NHWC, and free. Skipping the round trip is what
   keeps time out of the batch axis.
2. **The 9 MemBlock states become graph inputs and outputs.** Within one invocation the
   state is threaded internally (blocks after the first `TGrow` run 2x, 4x, 8x); only the
   last value crosses the graph boundary.
3. **`post_quant_conv` folds to a `Conv2d`.** It is a 1x1x1 `CausalConv3d`, so
   `time_kernel_size == 1` and every entry of `time_causal_padding` is 0 -- pointwise. That
   removes the graph's only 5-D op, and with it the NCDHW<->NDHWC transposes on the input.

### Correctness

Verified against the stock `parallel=True` decoder at T=3, replaying 3 invocations and
applying `frames_to_trim` caller-side:

```
reference (1, 3, 17, 320, 512)   streaming (1, 3, 17, 320, 512)
max|diff| = 4.113e-06   SNR = 115.6 dB
```

That is fp32 rounding, not a behavioural difference. The exported ONNX then matches torch
at 114.2 dB with zero unsafe constants.

The frame trim moved out of the graph, which is what makes 8 frames per invocation
useful: `frames_to_trim = 7` applies to the whole video, so the caller drops the first 7
frames of invocation 1 only. 7 invocations x 8 frames - 7 = **49 frames**, matching Fig 13.

### Calibration: states cannot be zeros

`work/device/make_stream_calib.py` replays each prompt's 7 latent frames in temporal order
and dumps the real `(latent, state_0..state_8)` tuple at every step. Frame 0 of each prompt
legitimately has zero state, so the zero case appears at its true 1-in-7 frequency.

| | n | `|latent|max` | `|state|max` |
|---|---|---|---|
| frame 0 (zero state) | 8 | 7.361 | **0.000** |
| frames 1-6 | 48 | 16.344 | **6.119** |

Calibrating the 9 state inputs on zeros would have been trap #4 exactly over again -- an
activation range of 0 for nine of the ten graph inputs. 56 samples, ~3.09 GB (the states
are 13.76 M floats = 55 MB per sample in fp32); **delete `work/calib/vae_dec_stream` once
the context binary is built.**

### Result: the transposes are gone

`analyze_net.py` on `vaedecs_net.json`:

| | T=1 parallel | **streaming** |
|---|---|---|
| nodes | 149 | 180 |
| **Transpose nodes** | 47 | **20** |
| **Transpose output bytes** | **779.80 MB** | **63.00 MB** |
| transpose / compute | 1.16 | **0.09** |
| `Reshape` / `Pad` | 23 / 9 | **0 / 0** |
| float fallback tensors | 0 | 0 |
| MACs per inference | 205.07 G | 205.07 G |
| context binary | 12.88 MB | 12.13 MB |

Same arithmetic, 12.4x less layout traffic, and it now yields 8 frames instead of 1. For
reference DistilT5 -- only 3.7x off its paper figure -- sits at ratio 0.28; the streaming
decoder is now cleaner than that.

**Every one of the 20 survivors is an I/O boundary transpose**, not interior:
`state_k_nhwc` x9 in, `nstate_k_nchw` x9 out, plus `video_nchw` and `latent_nhwc`. The
interior is completely clean.

### Device results

Held-out real latents 0050-0055 (prompt 7 frames 1-6, real non-zero state), never seen by
the quantiser. `--perf_profile burst`, min-of-12 per round, 3 rounds in one thermal session
(trap #10).

| | T=1 parallel | **streaming** | paper |
|---|---|---|---|
| deploy SNR, mean | 31.55 dB | **34.27 dB** | 35 dB |
| deploy SNR, min | 30.95 dB | **33.75 dB** | — |
| pixel error, mean | 2.18 | **1.25** | — |
| pixel error, p99 | 10.19 | **4.63** | — |
| nan+inf | 0 | **0** | — |
| **latency per invocation** | **6.10 s** | **114.04 ms** | 248.9 ms |
| usable frames per invocation | 1 | **8** | — |

Derived:

| | T=1 parallel | streaming | factor |
|---|---|---|---|
| per invocation | 6100 ms | 114.04 ms | **53.5x** |
| per usable frame | 6100 ms | 14.26 ms | **428x** |
| throughput | 33.6 GMAC/s | **1798 GMAC/s** | 53.5x |
| 49-frame video decode | 298.9 s | **0.80 s** | 374x |

**Accuracy improved as well as latency** -- +2.72 dB, now 0.73 dB short of the paper's 35 dB
rather than 3.45 dB short. Two plausible reasons: the removed reshape/transpose chains were
themselves quantised tensors, each an extra requantisation step; and the frames that used to
be computed and discarded were still consuming activation range in the calibration
statistics.

### Reading the latency number honestly

Min per round was **114.0 / 146.7 / 139.7 ms**, averages 174.5 / 188.6 / 184.8 ms. That
spread is the DSP throttling exactly as trap #10 describes, so the reported figure is the
min across the session. The average is the more honest number for a sustained real workload,
and even at 188 ms the streaming graph beats the paper's per-invocation 248.9 ms.

Note also `NetRun` (135-210 ms) sits close to accelerator time here, unlike DistilT5 where
RPC dominated. This graph moves ~28 MB in and ~70 MB out per invocation in fp32. In a real
pipeline the 9 states never leave the device, so accelerator time is the right metric --
but a naive host-round-trip implementation would pay that I/O.

### Comparison with the paper, with the caveat stated

At 114.04 ms against the paper's 248.9 ms the streaming decoder is **2.18x faster**, and at
the throttled average it is still ahead. That comparison assumes the paper's VAE-Dec
invocation is the same 205 GMAC unit of work. The evidence for that is real but indirect --
Table 7's 0.21 GB peak memory matches a single-latent-frame graph, and 205 GMAC is what a
single latent frame costs. It is *not* independently confirmed, and the earlier "823.7
GMAC/s implied" figure in this document was derived by assuming it, so it cannot be used to
re-confirm it. Treat "faster than the paper" as likely but unproven; treat "53.5x faster
than our own previous build, at higher accuracy" as measured fact.

### Spilling went up, and it did not matter

| | T=1 parallel | streaming |
|---|---|---|
| spill_bytes | 214,958,080 | **482,344,960** |
| fill_bytes | 251,658,240 | **550,502,400** |

The graph now holds 8 frames of activations plus 9 states, so it overflows 8 MB of VTCM
harder than before -- and is 53x faster anyway. This retires the VTCM/spilling theory
entirely: spill volume was never the problem, and `vtcm_mb` never needed tuning. What
mattered was that the *transposes* were what overflowed, and how badly a 5-D permute with a
tiny innermost stride runs once it does.

### Phase 4g -- NHWC states, the shipping build (`vaedecsn`)

The 63.00 MB left in `vaedecs` was **all I/O boundary**: `state_k_nhwc` x9 in,
`nstate_k_nchw` x9 out, plus `video_nchw` and `latent_nhwc`. `--preserve_io layout` takes an
explicit tensor list, so naming **only `latent` and `video`** lets the 18 state tensors go
channel-last. That is safe precisely because states are opaque round-trip data -- `nstate_k`
feeds straight back into `state_k` and nothing on device interprets them. `latent` and
`video` the host *does* interpret, so trap #7 still binds for those two.

Host-side state raws must then be NHWC: `work/device/permute_states_nhwc.py` (in place, with
a `.layout_nhwc` sentinel, `--revert` to undo).

| | vaedec1q | vaedecs | **vaedecsn** |
|---|---|---|---|
| preserve_io | layout (all) | layout (all) | **layout latent video** |
| **Transpose nodes** | 47 | 20 | **2** |
| **Transpose bytes** | **779.80 MB** | **63.00 MB** | **7.95 MB** |
| transpose / compute | 1.16 | 0.09 | **0.01** |
| nodes | 149 | 180 | 162 |
| deploy SNR | 31.55 dB | 34.27 dB | **34.27 dB** |

The only survivors are `video_nchw` (7.86 MB) and `latent_nhwc` (0.08 MB) -- exactly the two
that must stay pinned. **Accuracy is unchanged to 2 decimal places**, confirming the layout
change is numerically neutral.

#### Interleaved A/B, and why the first comparison was thrown out

Measured separately the two builds looked -5% on min and -19% on average -- two numbers that
disagree, which under trap #10 means thermal drift, not signal. Re-run interleaved in one
session (`work/device/ab_stream.sh`), min-of-12 per round:

| round | S (NCHW states) | N (NHWC states) | delta |
|---|---|---|---|
| 1 | 121,663 us | 108,111 us | **-11.1%** |
| 2 | 111,370 us | 100,184 us | **-10.0%** |
| 3 | 181,919 us | 104,673 us | -42.5% (S throttled) |
| **min-of-all** | **111.37 ms** | **100.18 ms** | **-10.0%** |

Rounds 1 and 2 agree closely; round 3's S is a thermal outlier and is reported but not
relied on. **NHWC states buy ~10%.**

### Final Phase 4 result

| | T=1 parallel | **vaedecsn (shipping)** | paper |
|---|---|---|---|
| latency / invocation | 6.10 s | **100.18 ms** | 248.9 ms |
| usable frames | 1 | **8** | — |
| deploy SNR | 31.55 dB | **34.27 dB** | 35 dB |
| throughput | 33.6 GMAC/s | **2047 GMAC/s** | 823.7 implied |
| **49-frame decode** | **298.9 s** | **0.70 s** | — |

**60.9x per invocation. 487x per usable frame. +2.72 dB.** Identical arithmetic throughout --
205.07 GMAC in every build. The whole gap was layout and discarded work.

### A gotcha the multi-input graph introduces

`qnn-onnx-converter` reads the calibration list on the **host**; `qnn-net-run` reads its
input list on the **device**. With one input that distinction was invisible, because
`convert_vae_w8a16.sh` built its own host list with `find`. With ten inputs the list has to
carry `name:=path` pairs, and it is tempting to reuse the one the generator already wrote --
which carries `/data/local/tmp` paths. So the generator emits **both**
`input_list.txt` (device) and `calib_list_host.txt` (host), and the conversion script
asserts every referenced file exists before spending 50 calibration passes on it.

### Reproduce

```powershell
wsl -d Ubuntu -e cp ~/neodragon-build/vaedec1q/vaedec1q_net.json /mnt/c/<scratch>/
py -3.10 work/device/analyze_net.py <scratch>/vaedec1q_net.json
```

**Was unresolved; diagnosed 2026-08-21 - see the section above.** Of the four suspects
listed here at the time, #2 was right: the channel-to-frame reshapes do defeat HTP's tiling,
though the cost lands on the *transposes the converter inserts around them* rather than on
the reshapes themselves. #3 was half right - the transposes are real but interior, not
caused by `--preserve_io layout`, which pins only the I/O and is required for correctness.
#1 (`Resize`, 3 nodes) and #4 (`vtcm_mb`) are not the cause; `vtcm_mb` is a symptom, since
what overflows VTCM is the transpose output.

The per-op device profile (`--profiling_level detailed` + `analyze_prof.py`) is still worth
running to confirm the cycle split, but it is no longer the *next* step - it would confirm a
cause already established statically, and the rewrite has to happen either way. If it is
ever run, `--profiling_option optrace` additionally emits a QHAS HTML report summarising
cycles per HTP op type.

### Another gotcha: quantised graphs take float input files

The W8A16 graph declares `UFIXED_POINT_16` for **both** input and output, where the float
graph declares `FLOAT_32`:

```
vaedec1   Input latent  FLOAT_32          Output video FLOAT_32
vaedec1q  Input latent  UFIXED_POINT_16   Output video UFIXED_POINT_16
```

So `--use_native_input_files` — which the float build *requires* — makes the quantised
build fail, demanding 81,920-byte (16-bit) files instead of 163,840-byte fp32 ones:

```
Given input file ... with file size in bytes 163840. If the model expects a batch size
of one, the file size should match the tensor extent: 81920 bytes.
```

Feed fp32 raws **without** the flag and the runtime quantises on the way in and dequantises
the output. The rule is per-build, not per-model: **float graph → use the flag; quantised
graph → omit it.**

## Status

- 4a source audit — **done**
- 4b ONNX export — **done** (T=1 verified 112.8 dB; T=7 exports, host cannot verify)
- 4c float device run — **done**: 55.52 dB on synthetic / 64.18 dB on real latents, 10.6 s
- 4d W8A16 — **done**: 31.55 dB, 6.10 s. Below the 35 dB target and 24× off 248.9 ms.

- 4e diagnosis  - **done 2026-08-21**: the gap is layout-transpose traffic (779.8 MB per
  inference, exceeding the 671.2 MB of convolution output). Static finding, no device time.
- 4f streaming decoder - **done 2026-08-21, on device**: explicit MemBlock state, no batch
  fold. 114.04 ms and 34.27 dB, from 6.10 s and 31.55 dB. Transposes 779.80 -> 63.00 MB.
- 4g NHWC states - **done 2026-08-21, on device**: `--preserve_io layout latent video`.
  Transposes 63.00 -> **7.95 MB**, a further **-10%** interleaved. **Shipping build is
  `vaedecsn`: 100.18 ms, 34.27 dB, 8 frames -- 60.9x and +2.72 dB vs the T=1 build.**

**Phase 4 is done.** The decoder converts, quantises, runs on device, is numerically sane,
and is no longer the pipeline's bottleneck. A 49-frame decode went from **299 s to 0.70 s**.

| open question | status |
|---|---|
| **The 24x latency gap** | **closed.** Cause was converter-inserted layout transposes around `TGrow`'s channels-to-frames batch fold. Fix was the explicit-state streaming rewrite. Measured on device. |
| **0.73 dB SNR shortfall** | 34.27 vs 35 dB, down from a 3.45 dB gap. Route if it is ever worth closing: `--algorithms` / AdaRound, or `--quantization_overrides` with AIMET encodings. Low priority -- mean pixel error is 1.25/255. |
| **Boundary transposes** | **closed.** `--preserve_io layout latent video` + NHWC state raws took 63.00 -> 7.95 MB and bought a further 10% (interleaved A/B). Shipping build is `vaedecsn`. |

### What carries to the rest of the pipeline

1. **Run `analyze_net.py` on every `*_net.json` before spending device time.** Transpose
   bytes vs compute bytes is the cheap pass/fail. Ratio 1.16 was catastrophic; 0.28
   (DistilT5) is livable; 0.09 is clean.
2. **Never let a temporal or batch axis be synthesised by a reshape of the channel axis.**
   It is free in NCHW and a full scatter in NHWC, and HTP is NHWC. Carry state explicitly
   instead. This is the same lesson as trap #1's RoPE rank reduction, and it is the shape of
   the fix paper 4.1 keeps describing.
3. **Spill volume is not a proxy for anything.** Spilling nearly tripled while latency fell
   53x.
4. **Quantisation quality can improve when you delete graph structure** -- fewer requantised
   intermediates, and no discarded frames polluting the calibration statistics.
