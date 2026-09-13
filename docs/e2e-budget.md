# End-to-end budget — measured, not projected

*2026-08-21; reconciled against the shipping app 2026-08-23.*

> **The whole pipeline now runs in an app: 23.5–26.9 s per 49-frame video, 7.2 s for a
> single image.** That is WALL CLOCK and it is not the same quantity as the compute
> budget below — see "Wall clock vs compute budget" at the end. Every row marked
> "not built" in the original table has since been built and run; the annotations are
> kept as written so the projection can be scored against the outcome.

One 49-frame 320×512 video, the AR t2v path, reference config
(`num_stages=3`, `frames_per_unit=1`, one denoising step per stage).

| component | invocations / video | per invocation | total | source |
|---|---|---|---|---|
| CLIP L + CLIP G | 1 | 90.5 ms | 90.5 ms | paper Table 7 (FP16, not built) |
| **DistilT5** | 1 | **12.84 ms** | **12.8 ms** | measured, Phase 1 |
| Context adapter | 1 | — | — | folded into the MMDiT rows below |
| SSD1B UNet + Dec | 1 | 814.6 ms | 814.6 ms | paper Table 7 (not built) |
| **VAE encoder** | 1 | **166.83 ms** | **166.8 ms** | measured, Phase 4b |
| **MMDiT stage 0** | 6 | 104.7 ms | 628.2 ms | paper (rewritten, not converted) |
| **MMDiT stage 1** | 6 | 218.3 ms | 1309.8 ms | paper (rewritten, not converted) |
| **MMDiT stage 2** | 6 | 938.3 ms | 5629.8 ms | paper (rewritten, not converted) |
| **VAE decoder** | 7 | **100.18 ms** | **701.3 ms** | measured, Phase 4 |
| QuickSRNet | 49 | 6.5 ms | 318.5 ms | paper Table 7 (not built) |
| | | | **~9.7 s** | |

**The MMDiT is 7.57 s of that — 78% of the total, and 90% of everything still unbuilt.**
Nothing else is worth optimising until it lands.

## Where the measured numbers beat the paper — one row CORRECTED 2026-08-23

| component | paper (8 Elite Gen4) | measured here | |
|---|---|---|---|
| VAE encoder | 1206.5 ms | **166.83 ms** | **7.2× faster** |
| VAE decoder | 248.9 ms **for the whole 49-frame video** | 701.3 ms (7 × 100.18) | **2.8× SLOWER** |
| DistilT5 | 3.5 ms | 12.84 ms | 3.7× slower — see below |

> ### The decoder row used to read "2.5× faster". That was wrong.
>
> It compared the paper's number against **one of our seven invocations**. The paper is
> explicit (§3.2): *"the modified version achieves a latency of **143ms when decoding a
> [49×320×512] video** from a latent tensor of shape [7×40×64]"* — that 143 ms (X Elite,
> 248.9 ms on 8 Elite) is the **entire video**, one number, not per chunk.
>
> Our streaming decoder emits 8 frames per invocation and needs **7 invocations** for 49
> frames (trap #38), so the comparable figure is **701.3 ms against 248.9 ms**.
>
> The check that settles it: summing Table 7's X Elite column with the decoder read as
> whole-video gives **6,602 ms**, and the paper states **6.6 s**. Read per-invocation it
> gives ~7.5 s and nothing lines up. The whole-video reading is the one that reconstructs
> their own headline.
>
> **What is still true:** the 60.9× speedup over our own T=1 parallel build is real and
> unaffected — that was ours-vs-ours. What changed is only the comparison to the paper,
> which now shows a gap rather than a win. The decoder is a **latency target**, not a
> solved component.

The encoder win came from deleting structure the reference left in: at T=1 every causal
conv front-pads two planes of zeros, so the whole encoder collapses exactly to a 2-D
convnet. The decoder rewrite (removing `TGrow`'s 779.80 MB of layout transposes) was also
a real 60.9× win over where it started — it simply started much further behind than this
table used to imply.

DistilT5 is the one component slower than the paper. It is FP16 by design (Table 9), and
12.84 ms against 3.5 ms is a tuning gap, not a structural one — and at 0.1% of the budget it
is not worth a session. See `docs/device-results.md`.

## What this implies for the MMDiT

If the paper's MMDiT figures are, like its VAE figures, the cost of an unoptimised graph,
there is room. The two structural things already applied to the rewrite before its first
conversion (`work/export/export_mmdit_stage.py`):

- **RoPE at rank 4** instead of rank 6 (trap #1) — verified bit-exact inside the real model
- **the mask/RoPE preamble hoisted to the host**, which removed 835 nodes of shape
  arithmetic and, separately, is what makes the padded 3-graph design exact

Both were applied *before* the first conversion rather than after, which is the lesson
Phase 4 paid 53× to learn.

Unknown until measured: how much of the paper's per-stage figure is padding waste. The
3-graph design pads to the per-stage maximum, costing 71% / 38% / 20% wasted tokens at the
shortest unit — but stage 2, which is 74% of the MMDiT budget, wastes only 20%, and the
paper's own `[7×…]` naming says its numbers are already padded-to-max.

## Caveats

- SSD1B and QuickSRNet rows are paper figures for components not built here. Phase 2's
  QuickSRNet has pre-compiled QNN binaries on AI Hub, so that row should be cheap to
  replace with a measurement.
- The MMDiT rows are paper figures for a graph that has been rewritten but not converted.
  Treat the 9.7 s as an upper bound with three unmeasured components in it, not a result.
- Everything measured here is a single module in isolation. Compounding quantisation loss
  across the chain is the paper's own stated headline risk (§4.2) and is not captured by any
  per-module SNR.


---

# Phase 6 groundwork, measured 2026-08-23

## Residency: the three MMDiT stages ARE co-resident

This was the open risk that decided the E2E driver's architecture -- the AR loop visits
all three stages every unit, six units per 49-frame video, so if the ~1.5 GB context
binaries could not be held at once the driver would have to load and unload 18 times per
video and that cost would dominate everything else measured.

`work/device/residency_probe.sh` holds each graph open in its own `qnn-net-run` and
samples `/proc/meminfo` after each addition:

| step | MemAvailable |
|---|---:|
| baseline | 5119 MB |
| + MMDiT stage 0 | 3157 MB |
| + MMDiT stage 1 | 2461 MB |
| + MMDiT stage 2 | **1541 MB** |

**All three hold, for ~3.58 GB, with ~1.5 GB still free.** Adding DistilT5 (260 MB) and
the two VAEs (54 MB) leaves ~1.2 GB of headroom, so **the whole video path fits and the
driver can load once**. The SSD1B first-frame modules run once and can be loaded and
released before the loop starts.

> **Trap: process RSS is the wrong number for a QNN graph.** Each `qnn-net-run` holding a
> 1.5 GB context binary reports **5-12 MB of RSS** -- the weights live in DSP/ION memory,
> not the CPU process's address space. An earlier reading of 4.1 GB RSS during a stage-2
> run was an execution working set, not residency, and reading it as residency predicted
> ~12 GB for three stages and would have forced a needless load/unload redesign.
> **Measure MemAvailable across load, not RSS.**

## Measured MMDiT cost of one 49-frame video

Per-invocation, min-of-N on device, against paper Table 7:

| stage | tokens | ours | paper | ratio |
|---|---:|---:|---:|---:|
| 0 | 408 | 169.2 ms | 104.7 | 1.62x |
| 1 | 648 | 375.8 ms | 218.3 | 1.72x |
| 2 | 1728 | 1851 ms | 938.3 | 1.97x |

The AR loop runs units 1..6, all three stages each:

    6 x (169.2 + 375.8 + 1851) = 14.4 s of MMDiT per video   (paper: 7.6 s)

VAE decode adds 0.70 s for 49 frames and DistilT5 12.84 ms once, so the MMDiT remains
~95% of the video path. The gap to the paper is a consistent 1.6-2.0x, not a blow-up.


---

## Wall clock vs compute budget (2026-08-23, corrected)

The app measures **23.5 s** best / 26.9 s typical for a 49-frame video. An earlier draft of
this section blamed the gap against the ~9.7 s projection on context loading and host-side
Kotlin. **That was wrong**, and the MMDiT's own measured numbers say so:

| component | invocations | measured each | total |
|---|---:|---:|---:|
| MMDiT stage 0 | 6 | 169.2 ms | 1.02 s |
| MMDiT stage 1 | 6 | 375.8 ms | 2.25 s |
| MMDiT stage 2 | 6 | 1851 ms | **11.11 s** |
| first frame (SSD1B path, in-app) | 1 | **2.2 s warm / 7.5 s cold** | see the correction below |
| VAE decoder | 7 | 100.18 ms | 0.70 s |
| VAE encoder | 1 | 166.83 ms | 0.17 s |
| DistilT5 + context adapter + CLIP | 1 | — | ~0.2 s |
| | | | **~22.6 s** |

That accounts for essentially all of the 23.5 s. Context loading and host-side Kotlin are
the ~1 s remainder, not the story.

**The projection was optimistic because it used the paper's MMDiT latencies, and ours run
a consistent 1.6-2.0x slower** (1.62x / 1.72x / 1.97x). The MMDiT is **14.4 s of the
23.5 s**, and *stage 2 alone is 11.1 s* -- 47% of the entire video.

So the honest ranking of what to optimise:

1. **Stage 2.** 1851 ms x 6. It is the longest sequence (1728 tokens), the slowest ratio to
   paper (1.97x), *and* the worst accuracy (20.01 dB vs a 24 dB target). Everything about
   stage 2 is the bottleneck.
2. **The first frame**, 7.2 s -- which disappears entirely if the user supplies an image.
3. Everything else put together is under 1.1 s and is not worth touching.

Per-module numbers were taken min-of-N in one thermal session (trap #10), so they remain a
floor; the ~1 s of unexplained wall clock is where loading and single-threaded host work
live. But the shape of the problem is arithmetic, not overhead.


---

## The first frame was never 4.7x the paper — measured 2026-08-23

`7.2 s` was recorded here as the first-frame cost and compared against paper Table 7's
1.53 s, giving a 4.7x gap that was called "the worst ratio in the pipeline". **Both halves
of that comparison were wrong.**

Instrumenting `FirstFrame.kt` to time LOAD separately from EXECUTE (`adb logcat -s
neodragon`):

| | before | after keeping CLIP G resident |
|---|---:|---:|
| load clipl | 376 ms | 66 ms |
| exec clipl | 6.1 ms | 5.9 ms (paper 14.0) |
| **load clipg** | **1897 ms** | **0.0 ms** |
| exec clipg | 40.4 ms | 44.6 ms (paper 76.5) |
| unet x4 | 1237 ms | 1178 ms (paper 938) |
| vae decode | 652 ms | 669 ms (paper 580) |
| **TOTAL** | **4436 ms** | **2204 ms** |

**Two separate errors.**

1. **The 7.2 s was a COLD run.** Measured cold-vs-warm on the same build: 7468 ms then
   2479 / 2204 / 2438 ms. The binaries are `mmap`'d, so the first run page-faults ~2.9 GB
   in from storage and later runs hit the page cache. A cold number was recorded as steady
   state.
2. **Wall clock was compared against inference-only.** Table 7 measures per-module
   latency; our figure included loading 1.6 GB of CLIP context. Different quantities.

**Compute alone is ~1.90 s against the paper's 1.61 s — 1.18x.** The first-frame path was
essentially at parity all along.

### The fix, and why the old justification had lapsed

`FirstFrame.kt` released both CLIP encoders after use, citing trap #35's lmkd incident and
"they reload in ~1 s". The real cost was **1897 ms for CLIP G alone** -- paid on every
generation, to reclaim memory for a graph that runs for 40 ms.

Trap #35 was about **heap copies** of context binaries. These are `mmap`'d, so the pages
are file-backed and evictable; keeping CLIP G mapped cost **nothing measurable in RSS**
(238-268 MB with it resident, against the 0.40 GB peak documented when it was released).
`KEEP_CLIPG = true`.

### Still on the table

The remaining 1.9 s cold load is storage-bound (1.4 GB). **Preloading CLIP G on a
background thread at app start would hide it entirely**, since the user spends seconds
typing a prompt first. Same argument applies to the three ~1.5 GB MMDiT stages.

> **Trap to add: instrument LOAD separately from EXECUTE before comparing against any
> published per-module latency.** Half of this pipeline's apparent worst-in-class gap was
> model loading plus a cold page cache, and it survived as "the worst ratio in the
> pipeline" for a whole session because the number was never broken down.


---

# Phase 2 — QuickSRNet, built and measured 2026-08-23

`work/device/quicksrm2x_v79.bin`, **377,880 bytes**. QuickSRNet **Medium, 2x**,
320x512 -> 640x1024. This is the step that takes the pipeline to the [640x1024] the
paper's headline quotes; without it our video is 320x512 and the two are not the same
product.

**Variant chosen by arithmetic.** At the ~2050 GMAC/s this chip sustains on the conv-heavy
W8A16 VAE decoder (205.07 GMAC in 100.18 ms):

| variant | GMAC @320x512 | est/frame | 49 frames |
|---|---:|---:|---:|
| small | 3.73 | ~1.8 ms | ~90 ms |
| **medium** | **8.26** | **~4 ms** | **~200 ms** |
| large | 67.9 | ~33 ms | ~1.6 s |

Large cannot be the paper's 6.5 ms and is ruled out. Medium sits next to the paper's
318 ms for a whole video.

## Deploy SNR: 28.06 dB against a 48 dB target

Measured on **49 held-out frames** (video 8, excluded from the 343-frame calibration set --
scoring on calibration data would be optimistic and could not fail).

Table 9 gives QuickSRNet W8A16 + **AdaRound** and credits AdaRound with "7+ dB SQNR"; we
apply `--algorithms cle` only. That accounts for ~7 dB of the 20 dB gap. **The remaining
~13 dB is unexplained.** Untested hypotheses: the paper never says which variant it used;
`use_ito_connection`'s residual may be quantisation-sensitive; 8-bit weights may hurt
disproportionately on a 50K-param, 7-conv model.

## But the number that decides it is a different one

There is no ground truth at 640x1024 -- the frames are natively 320x512 -- so "28 dB" is
distance from the fp32 model, not from truth. The question that matters is whether the
quantised model still beats the free alternative:

| | vs fp32 QuickSRNet |
|---|---:|
| **device (W8A16 + CLE)** | **27.90 dB** |
| bilinear 2x | 23.22 dB |

**The quantisation error is smaller than the thing QuickSRNet exists to fix.** Rendered
side by side (`work/device/render_quicksr.py`), the device output is essentially
indistinguishable from fp32 and both recover edge definition bilinear smears. Worth
shipping at 377 KB and ~200 ms per video.

> **Do not read a deploy-SNR shortfall as "not worth shipping" without asking what the
> alternative costs.** 28 dB sounds like a failure next to 48; it is 4.7 dB BETTER than the
> zero-cost baseline the module replaces.

## Integration

`Video.kt` upscales **inside the decode loop**, not over a finished list, so only one
640x1024 frame is alive at a time -- holding 49 alongside the 320x512 originals would be
~120 MB of bitmaps in a process lmkd has already killed once (trap #35). It is conditional
on the model being present, so the app still works at 320x512 without it.

**Ranges are the trap.** The decoder emits [-1,1]; the QuickSRNet checkpoint trained on
[0,1] and returns [0,1]. Getting either end wrong does not crash -- it produces a
washed-out but plausible video. The calibration set is built in [0,1] for the same reason.
