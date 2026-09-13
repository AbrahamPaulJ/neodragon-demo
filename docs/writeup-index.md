# Writeup index — the arc, and where every piece of evidence lives

*For turning this project into a paper or article. This file deliberately contains almost
no detail: the detail is already written down, and duplicating it is how two versions of a
number start disagreeing. What is here is the **through-line** and a **map**.*

---

## 1. The thesis the project kept re-proving

> **Almost every win came from deleting structure from the graph, not from tuning the
> backend.** Not one came from a knob.

Six independent instances, each measured:

| # | structure removed | measured effect |
|---|---|---|
| 1 | RoPE 6-D tiling → rank 4 | identical multiply count; the win is the *rank* |
| 11 | `TGrow` folding time into batch (`reshape(-1,C,H,W)`) | 779.80 MB of layout traffic/inference, **53.5× latency** |
| 16 | causal pad against a size-1 time axis | 1071.2 → **363.33 GMAC**, all 5-D tensors gone |
| 19 | `t.view(t.shape[0], …)` asking a tensor for a known dim | `Shape` ×186 → ×55; killed `Where` ×196, `ConstantOfShape` ×201 |
| 23 | rank-3 layout mismatch (NCF vs NFC) around the score matrix | transpose/compute 1.10 → **0.25**, 191 → 85 Transposes |
| 26 | `torch.stack(…, dim=-1)` in RoPE — an element interleave | 28.7% of the whole graph; RoPE total was **43.3%** |

The counterexamples matter as much: `vtcm_mb` never needed tuning (spill volume nearly
tripled while latency fell 53×, trap #13), hand-converting ONNX to fp16 was **3.7× slower
and 10 dB worse** (trap #9), and the `--use_dynamic_16_bit_weights` fix for 8-bit K/V —
a real, confirmed mechanism — bought **+0.01 dB for +18.6% latency** (trap #27).

**Corollary, stated as trap #28: a plausible mechanism is not a cause.** The AdaLN
investigation ended with exactly this: the shared-chunk encoding (trap #31) was genuinely
present, genuinely a bug, and responsible for **~0.3%** of the error. The conditioning-path
hoist was the whole effect. Both were built; only one mattered.

## 2. How the project avoided fooling itself

The methodological spine, and the best material for a "lessons" section:

* **Read the converter's own output before building anything.** Two accuracy hypotheses
  were killed for free by reading encodings out of `net.json`; a third cost a 42-minute
  conversion to move SNR by 0.01 dB.
* **Size the candidate against the error that is actually there.** The AdaLN fix was
  chosen by converting each tensor's SNR into an *absolute* error and comparing it to the
  quantisation step the shared encoding imposes — 1.354e-4 against an observed 0.032–0.046,
  i.e. 291–343× too small to be the cause. One line of arithmetic on data already on disk.
* **Static analysis before device time.** `analyze_net.py` (transpose bytes ÷ compute
  bytes) is free and caught the catastrophe; `matmul_bits.py` is the only tool that can see
  trap #27; `prof_by_optype.py` found trap #26.
* **Min-of-N in one thermal session, never averages.** The DSP clock swung 776 → 596 MHz
  and reversed an A/B verdict (trap #10); one build read 291.8 / 265.4 / 192.9 ms across
  three consecutive rounds.
* **Assert two different inputs give two different outputs.** Trap #8's failure mode is
  *plausible* output, identical for every prompt.
* **Verify every rewrite bit-exact against the stock model before converting.** The RoPE
  rewrite, the T=1 collapse, the padding envelopes (123–128 dB, the fp32 floor) and the
  AdaLN split were each proved equivalent first.

## 3. Chronology

| session | date | what changed |
|---|---|---|
| 1 | 08-14 | Phase 1 DistilT5 on device, 49.04 dB. Trap #3: HTP runs float graphs in fp16 with no fp32 upcast — the unscaled build returned **74% NaN** |
| 2 | 08-21 | Phase 4 VAE decoder. The 24× gap diagnosed as layout traffic and closed: **60.9×** per invocation, 487× per usable frame, *and* +2.72 dB |
| 3 | 08-21 | Phase 4b VAE encoder — a phase absent from the plan. The feared `deque` was **dead code on the deployed path**; T=1 collapse to 2-D. 7.2× under the paper, first try |
| 4 | 08-21/22 | Phase 5 scoped and stage 0 converted. Latency solved: 248.3 → 172.2 ms. Accuracy stuck at 19.48 dB with seven hypotheses dead |
| 5 | 08-22 | Layerwise device dump (aarch64 model lib + `--set_output_tensors`). **Gap located**: image stream enters `norm1` at 54.96 dB and leaves at 16.51 dB |
| 6 | 08-23 | AdaLN fix built (**27.89 / 23.99 / 20.01 dB**); stages 1–2 converted; Phase 3 + first-frame path; **whole pipeline shipped as an APK** |

## 4. Beliefs that turned out to be wrong

The most useful table in any honest writeup.

| believed | actually | where |
|---|---|---|
| weights at `karnewar/Neodragon` | 401s — real repo is `Qualcomm-AI-Research/Neodragon` | CLAUDE.md corrections |
| needs a 12 GB GPU / cloud rental | RTX 3050 6 GB runs the reference fine (~35 s/video) | CLAUDE.md |
| the project was blocked on GPU compute | it was never blocked; a session simply ended | CLAUDE.md's own warning |
| the VAE encoder "is not exportable as written" | the `deque` branch is never executed on the AR path | `docs/phase4b-vae-encoder.md`, trap #15 |
| Phase 5 is "3 static graphs" | **18** distinct token counts collapsing into 3 padded envelopes | `docs/phase5-mmdit-scope.md` |
| `num_stages` varies | it is **always 1** at the call site | same |
| `soc_model: 69` was suspect | correct — `QNN_SOC_MODEL_SM8750 = 69` | `docs/phase4-vae-decoder.md` |
| spilling indicates a problem | spill nearly tripled while latency fell 53× | trap #13 |
| the shared AdaLN chunk encoding is the accuracy bug | real, but ~0.3% of the effect | `notes/2026-08-23-adaln-fix.md` §1 |
| an app can exec `qnn-net-run` like the desktop harness | denied the DSP by SELinux; must load in-process | `notes/2026-08-23-android-e2e.md`, trap #33 |
| the app's wall clock is dominated by context loading | it is the MMDiT: 14.4 s of 23.5 s | `docs/e2e-budget.md` (corrected) |

## 5. Where the evidence lives

| topic | file |
|---|---|
| **all 39 traps**, with their measurements | `CLAUDE.md` |
| the three statically-inspectable traps, in depth | `docs/trap-audit.md` |
| the paper's own claims (Tables 7/9) read at source | `docs/paper-notes.md` |
| VAE decoder: the 24× diagnosis and streaming rewrite | `docs/phase4-vae-decoder.md` (601 lines, the fullest single story) |
| VAE encoder | `docs/phase4b-vae-encoder.md` |
| MMDiT scope / latency / accuracy | `docs/phase5-mmdit-{scope,latency,accuracy}.md` |
| the AdaLN fix, weighed then built | `notes/2026-08-23-adaln-fix.md` (653 lines) |
| the layerwise dump that located the gap | `notes/2026-08-22-layerwise-dump.md` |
| the Android app and traps #33–#39 | `notes/2026-08-23-android-e2e.md` |
| end-to-end latency, measured and reconciled | `docs/e2e-budget.md` |
| what to do next | `docs/roadmap.md`, `HANDOFF.md` |
| per-session narrative | `notes/2026-08-*` |

## 6. Headline numbers

* **49-frame 320×512 video, entirely on a phone, offline: 23.5 s.** Single image: 7.2 s.
* VAE decoder **60.9×** faster per invocation than the naive build, **487×** per usable
  frame, and 2.72 dB *more* accurate.
* VAE encoder **7.2×** under the paper's latency, above its accuracy target, first try.
* RoPE was **43.3%** of the MMDiT; one `torch.stack` was **28.7%**.
* MMDiT accuracy after the AdaLN fix: 27.89 / 23.99 / **20.01** dB at 408 / 648 / 1728
  tokens (targets 29 / 22 / 24) — accuracy falls with sequence length.
* App peak RSS **0.40 GB** while driving 7.96 GB of models.

## 7. Honest caveats to state in any writeup

* Stage 2 misses its accuracy target by 3.99 dB and its latency by 1.97×, and it is 47% of
  the video. It is the weakest part of the port.
* The end-to-end latent SNR against fp32 is **7.81 dB** (4.41 dB by the last frame). The
  output looks good; SNR is not the metric a reader should be handed alone. No perceptual
  metric (LPIPS/FVD) has been run — roadmap B2.
* The Kotlin host-side port is verified only at the tokenizers. Roadmap B1.
* QuickSRNet is unconverted, so the shipped resolution is 320×512 upscaled by the display,
  not by the paper's upscaler.
* Nothing here is a controlled comparison against the paper's own hardware; the paper's
  numbers are quoted from Tables 7/9, not reproduced.
