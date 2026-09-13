# Roadmap — where this goes next

*Rewritten 2026-08-23 (end of session 7). The pre-session-7 version scored tracks that are now done or refuted; it is preserved in git history.*

---

## Where we are

Everything in the plan is converted and shipping. All three MMDiT stages exceed their
paper accuracy targets. Video 23.5-26.9 s, image 2.1-2.5 s warm.

## The gap that remains

Against the paper's Snapdragon **X Elite** 6.7 s (the phone demo's own figure -- Table 7
gives no 8 Elite E2E number, and its 8 Elite column sums to ~10.6 s):

| | ours | paper 8E | note |
|---|---:|---:|---|
| MMDiT | 14.4 s | 7.57 | **the whole gap** |
| first frame | 2.1 s warm | 1.53 | ~parity once loading is excluded |
| VAE decode | 0.70 s | 0.25 | 2.8x |
| VAE encode | 0.17 s | 1.21 | we win 7x |

**The MMDiT is the only thing worth optimising.** Everything else combined is under 1 s of
the difference.

## 1. Softmax and the mask add — 43% of stage 2, untouched

| | share of stage 2 | |
|---|---:|---|
| **Softmax** | **27.2%** | 0.221 cycles/byte for what should be 4-5 passes. Looks unfused or badly quantised |
| **mask add** | **15.8%** | `[24,1728,1728] + [1,1728,1728]` broadcast costs **14x** a plain add of the same output |

Both operate on the score matrix. The `crepro_*` micro-benchmark graphs already in the
repo are the cheap way to probe either before touching the real graph.

**Effort:** unknown, investigation-first. **Value:** highest on the board.

## 2. Image-to-video is built — measure what it actually saves

Skipping SSD1B removes the first frame AND 1.68 GB of context that never has to map.
Measured end to end it should be well under the T2V number, but that has not been timed.

**Also unmeasured:** `vaeenc` was calibrated on SSD1B renders, not photographs. Expect some
accuracy loss on real photos and check whether it is material.

## 3. QuickSRNet's missing 13 dB

28.06 dB against the paper's 48. AdaRound accounts for ~7 by their own account; the rest is
unexplained. Untested hypotheses: wrong variant (they never say which), `use_ito_connection`
being quantisation-sensitive, or 8-bit weights hurting a 50K-param model disproportionately.
**Low priority** -- it already beats bilinear by 4.7 dB, which is the comparison that
decides whether to ship it.

## 4. Longer videos are a rebuild, not a setting

The token count grows linearly with unit index (the coarse history term is `(u-2)` frames),
so attention cost grows quadratically. 5 s = 15 AR units = 2088 tokens for stage 2 vs 1728
today, and 45 MMDiT calls vs 18. **Stage 2 alone goes 11.1 s -> ~37 s.** The graphs are
static, so it needs new envelopes, three re-conversions and fresh calibration.

Worth exploring first: `frames_per_unit` is 1; raising it cuts the unit count for a given
length and attacks the 2.5x call count directly.

## 5. Polish

* Per-model locks in `ndqnn.cpp` — the global `g_mu` covers load AND execute, so a
  background preload blocks generation. Worth ~1 s on the first tap only, and it is a real
  concurrency change to NPU-driving native code.
* The bit-level parity harness for the Kotlin port (`bilinear`, `blockNoise`, VAE sampling)
  is still unwritten. Tokenizers are verified id-for-id; nothing else is.

## Dead — do not revisit

* **Byte reductions in the MMDiT** (trap #42): -28% bytes bought -4.4% cycles.
* **Op-count reduction** (trap #41): -27.6% ops bought -1.7% time.
* **"Accuracy falls with sequence length = residual accumulation"**: wrong. It was the
  split score contraction; the fix is uniform across all three stages.
* **8-bit K/V** (trap #27): +0.01 dB, +18.6% latency.
