# Neodragon on S25 Ultra — project index

Porting Qualcomm's Neodragon text-to-video model to the S25 Ultra NPU via QAIRT.
Extension of the same deployment pattern as `controlnet-mobile` / LocalDream.

**Status: shipping app (2026-08-23, session 7).** Prompt -> 49-frame video, photo ->
video, prompt -> 640x1024 image, all offline. The app downloads its own models from a
**public** HF repo, refuses to download on a device that cannot run them, upscales 2x with
QuickSRNet, and writes an MP4 to the gallery.

**Every module in the plan is converted.** All three MMDiT stages now exceed their paper
accuracy targets for the first time -- 39.67 / 37.80 / 33.89 dB against 29 / 22 / 24 --
after the fused-score rebuild (trap #40).

Video 23.5-26.9 s; image 2.1-2.5 s warm; peak RSS 238-268 MB.

> ### ➤ START HERE: `HANDOFF.md` (repo root)
> State, the numbers, the one thing to do next, and what NOT to redo. Then
> `docs/roadmap.md`. For a writeup, `docs/writeup-index.md` maps the whole arc.

The reference PyTorch pipeline runs locally on the RTX 3050. Nothing is blocked.

> Keep this file free of self-contradiction. The 2026-08-14 session left a header saying the
> project was blocked on GPU compute while a later section said that was resolved, and a week
> later it read as "we were stuck" when the truth was "the session ended". **When you close a
> blocker, rewrite the section that names it.**

## Read on demand

| Doc | Read it when |
|---|---|
| `notes/2026-08-23-adaln-fix.md` | session 6: the AdaLN fix weighed, built and measured — 27.89 / 23.99 / 20.01 dB |
| `notes/2026-08-23-android-e2e.md` | session 6: the whole pipeline on the phone from an APK, traps #33-#39 |
| `notes/2026-08-23-HANDOFF.md` | Phase 5 accuracy: Session 5: the accuracy gap LOCATED to the AdaLN modulation path, the two fixes, the new dump toolchain |
| `notes/2026-08-22-layerwise-dump.md` | session 5 detail: NDK setup, the online-prep control, per-block table, the clipping scan |
| `notes/2026-08-22-HANDOFF.md` | session 4 history. Its "one thing to do next" (the layerwise dump) is DONE — see the 2026-08-23 note |
| `notes/2026-08-21-HANDOFF.md` | session 3 history. Its "one thing to do next" (8-bit K/V) was measured and is a dead end — see the 2026-08-22 note |
| `HANDOFF.md` | **read first.** State, next step, what not to redo, rebuild commands |
| `docs/traps.md` | **all 47 traps.** Read before converting, measuring, or debugging something that runs but is wrong |
| `docs/phase5-stage2-bandwidth.md` | why stage 2 is bandwidth-bound, the fused-score fix, and the per-op cost table |
| `notes/2026-08-23-session7-app.md` | session 7: I2V, QuickSRNet, MP4, device gating, and the Compose/MediaCodec traps |
| `docs/writeup-index.md` | the through-line, the chronology, the beliefs that were wrong, and where every number lives |
| `docs/porting-other-socs.md` | **porting to 8 Gen 3 / 8 Gen 2 and older.** What changes, what to re-measure, the app build flags |
| `docs/roadmap.md` | **what to do next and in what order.** Nine tracks scored for effort and risk, incl. the QAIRT LoRA toolchain find |
| `docs/trap-audit.md` | **read first.** The three statically-inspectable traps, measured, with verified fixes |
| `docs/paper-notes.md` | paper §4.1/§4.2 read at source, incl. Table 7/9 — **supersedes the research brief**, which has a wrong quantization table |
| `docs/device-results.md` | Phase 1 DistilT5 on-device: 49 dB, 12.8 ms, plus latency-tuning findings |
| `docs/phase4-vae-decoder.md` | **the most important doc.** Phase 4 end to end: the 24× gap diagnosed, the streaming rewrite, and the 53.5× device result |
| `docs/deployment-size.md` | APK and model payload sizes, measured where measured; includes the MMDiT weight-sharing question |
| `docs/e2e-budget.md` | the whole-pipeline latency budget, measured where measured and flagged where not |
| `docs/phase5-mmdit-scope.md` | **read before touching Phase 5.** Three corrections to what this file used to say about the MMDiT, all measured |
| `docs/phase5-mmdit-latency.md` | **the latency answer.** RoPE is 43.3% of the MMDiT; `torch.stack(dim=-1)` alone is 28.7%. Per-op profile + the exact fix |
| `docs/phase5-mmdit-accuracy.md` | **the accuracy answer, now located.** The AdaLN modulation path, plus the seven dead hypotheses in the order they died |
| `docs/phase4b-vae-encoder.md` | Phase 4b end to end: why the `deque` is dead code on the E2E path, the exact T=1 collapse to 2-D, and the 166.83 ms device result |
| `notes/2026-08-14-audit-phase1.md` | session 1 history and environment corrections |
| `notes/2026-08-21-vae-streaming.md` | session 2 history: diagnosis, rewrite, device results, exact reproduce steps |
| `notes/2026-08-21-vae-encoder.md` | session 3 history: the encoder audit, the T=1 collapse, and the device numbers |
| `notes/2026-08-21-mmdit-scope.md` | session 3 part 2: the three Phase 5 corrections, the bit-exact rewrite, and the padding insight |
| `2511.06055v1 arxiv paper.pdf` | the paper itself; text extracted to `work/paper/paper.txt` |
| `neodragon-port-research-brief.md` | the original brief. **Largely superseded** — see `docs/paper-notes.md` for what it gets wrong |

## Corrections to the research brief (verified 2026-08-10)

The brief is sound on strategy. Three facts in it are wrong or stale:

| Brief says | Actually |
|---|---|
| Weights at `huggingface.co/karnewar/Neodragon` | **401s.** Real repo is `Qualcomm-AI-Research/Neodragon` — public, BSD-3-Clause-Clear, 17.5 GB total |
| "fits on RTX 3060 12GB" | This machine has an **RTX 3050 6 GB laptop GPU**. The bf16 reference pipeline does not fit naively |
| — | Upstream repo `qualcomm-ai-research/neodragon` last pushed 2026-07-02, ~72 MB, BSD-3-Clause-Clear. Paper HTML at `arxiv.org/html/2511.06055v1` is live |

Minimal weight subset for the AR t2v path is **~8.6 GB**, not the full 17.5 GB:
`text_encoder_3` + `tokenizer_3` (DistilT5, 259 MB) · `diffusion_transformer_320p` (3.1 GB) ·
`context_adapter` (520 MB) · `causal_video_vae` (239 MB) · `ssd_1b_unet` (2.6 GB) ·
`ssd_1b_vae` fp16 (167 MB) · `ssd_1b_text_encoder` + `_2` fp16 (1.6 GB) · tokenizers.
The `_multistep_t2v` variants are a separate 3.7 GB and are not needed for the AR path.

## What is built (2026-08-23)

| Phase | State | Numbers |
|---|---|---|
| 1 — DistilT5 | **done, on device** | 49.04 dB, 12.84 ms (paper: FP16, 3.5 ms) |
| 2 — QuickSRNet | **DONE, on device** | Medium 2x, 320x512 -> 640x1024. **28.06 dB** (paper 48, they use AdaRound). 377 KB. Beats bilinear by 4.7 dB |
| 3 — SSD1B UNet | **DONE, on device** | 32.51 dB W8A16. Plus `clipl` 60.75, `clipg` 14.71, `ssd1bvaedec` 32.43, `cliplp` (video-path pooled) |
| 4 — VAE decoder | **DONE, on device — but SLOWER than the paper** | W8A16 **34.27 dB, 100.18 ms / 8 frames = 701 ms per video**. Paper's 248.9 ms is the **whole 49-frame video**, so this is **2.8× slower**, not faster (corrected 2026-08-23, `docs/e2e-budget.md`). Build `vaedecsn` |
| 4b — VAE **encoder** | **DONE, on device** | W8A16 **41.60 dB, 166.83 ms** (paper: 40 dB, 1206.5 ms). Build `vaeenc` |
| 5 — Pyramidal MMDiT | **DONE, all 3 stages, ALL TARGETS MET** | **39.67 / 37.80 / 33.89 dB** (targets 29 / 22 / 24) after the fused-score rebuild (trap #40): +10.9/+12.2/+12.0 dB, -4.4% cycles. Binaries `mmdit_s0fs/s1fs/s2fs` |
| 6 — full E2E | **DONE, shipping app** | video **23.5-26.9 s**; image **2.1-2.5 s warm**; peak RSS **238-268 MB**. Adds image-to-video, 2x upscale, MP4 export, self-download from a public HF repo, and a device-support canary |

Key files for Phase 4 (the working template for every remaining module):

| File | What it is |
|---|---|
| `work/export/export_vae_decoder_stream.py` | the streaming rewrite — the pattern to copy |
| `work/device/analyze_net.py` | **run this on every `*_net.json` before any device time** |
| `work/device/make_stream_calib.py` | real-state calibration capture |
| `work/qnn/convert_vae_stream_w8a16.sh` | multi-input W8A16 conversion |
| `work/device/push_stream.ps1` / `run_stream.sh` / `compare_stream.py` | device loop |
| `work/export/export_vae_encoder_2d.py` | Phase 4b: the T=1 causal-pad collapse, with the equivalence asserted per-conv |
| `work/pipeline/capture_first_frames.py` | real SSD1B first frames for encoder calib/test, no DiT needed |
| `work/device/make_enc_io.py` / `compare_enc.py` / `push_enc.ps1` / `run_enc.sh` | Phase 4b device loop |
| `work/audit/mmdit_shapes.py` | Phase 5: the 18 shapes, from pure arithmetic. No weights, no GPU |
| `work/audit/mmdit_trace_probe.py` | Phase 5: does the stock DiT run and trace? (yes) |
| `work/export/export_mmdit_stage.py` | **Phase 5's core.** Single-stage rewrite, host-hoisted mask/RoPE, rank-4 RoPE, padding to the per-stage envelope |
| `work/pipeline/capture_mmdit_calib.py` | Phase 5 calibration: 18 real DiT calls per reference video |
| `work/device/make_mmdit_io.py` / `compare_mmdit.py` / `push_mmdit.ps1` / `run_mmdit.sh` | Phase 5 device loop |
| `work/device/trace_transposes.py` | **why** a graph transposes: groups every Transpose by producer/consumer op type and ranks by bytes. The companion to `analyze_net.py` |
| `work/device/matmul_bits.py` | **run on every quantised attention graph.** Operand bitwidth of every MatMul/FullyConnected, static tensors starred. The only tool that sees trap #27 |
| `work/device/prof_by_optype.py` | per-op-type cycle census from a `--profiling_level detailed` run, joined against `net.json`. The tool that found trap #26 |
| `work/qnn/build_android_lib.sh` | Phase 5: aarch64-android model lib (5:52). **The only route to intermediate values** — `--debug`/`--set_output_tensors` refuse a context binary |
| `work/device/run_mmdit_lib.sh` | device modes `acc` / `accdef` / `blocks` / `names <file>` against the model lib. Online prep of the 1.5 GB graph is **65 s**, not minutes |
| `work/device/layer_snr.py` | **layerwise device-vs-fp32 SNR**, joined to the ONNX by name (trap #29), topological order with per-step deltas. The tool that located the accuracy gap |
| `work/device/block_tensors.py` | every tensor inside block N, for a targeted `--set_output_tensors` (`--debug` cannot finalize — trap #28b) |
| `work/device/clip_scan.py` | true fp32 range vs the **calibrated** encoding, per tensor per case. The one check the host ablations structurally cannot do |

The Android app (`work/android/`) — the deliverable:

| File | What it is |
|---|---|
| `app/src/main/cpp/ndqnn.cpp` | **the reason the app works.** In-process QNN runner: dlopen the backend, parse the binary's own tensor metadata, quantise/dequantise I/O, execute. Replaces exec'ing `qnn-net-run`, which an APK is not allowed to do (trap #33) |
| `app/src/main/java/.../QnnRunner.kt` | graph cache over the JNI layer; loads by name, releases on demand |
| `app/src/main/java/.../FirstFrame.kt` | prompt → image: CLIP L/G → SSD1B UNet ×4 (LCM) → VAE decode |
| `app/src/main/java/.../Video.kt` | prompt → 49 frames: T5 → context adapter → first frame → VAE encode → 18 MMDiT calls → streaming decode |
| `app/src/main/java/.../VideoStructure.kt` | the precomputed per-(unit,stage) structure, plus the on-device mask rebuild (trap #36) |
| `app/src/main/java/.../LatentOps.kt` | bilinear/nearest resize, pyramid, block noise — matched to `torch.nn.functional.interpolate(align_corners=False)` |
| `app/src/main/java/.../ClipTokenizer.kt` · `T5Tokenizer.kt` | CLIP BPE and T5 Unigram (Viterbi), both verified id-for-id against `transformers` |
| `work/export/export_app_assets.py` | host-side weights + constants → APK assets |
| `work/export/export_video_structure.py` | the 18 RoPE/layout tables + scheduler constants → APK assets |

Source of truth: **this repository**.
Models: **https://huggingface.co/AbrahamPJ/neodragon-npu-s25u** (public, 13 binaries).
App: **v0.2.0**, self-downloads its models, gated by an on-device support canary. Everything below
that is a weight, a converted binary, a generated asset or vendor SDK material is
gitignored — see `SETUP.md` to restore it.

Workspace layout:

```
src/neodragon/     upstream clone (BSD-3-Clause-Clear)
work/models/       10.27 GB hybrid pipeline weights
work/audit/        trap-audit scripts
work/export/       ONNX export + graph_fixes.py (reusable QNN graph passes)
work/qnn/          convert*.sh, htp_*_v79.json
work/device/       device I/O, compare*.py, analyze_prof.py, analyze_net.py, ab_*.sh
work/pipeline/     run_reference.py — memory-managed reference pipeline for 6 GB
work/android/      the shipping app (Compose UI + JNI QNN runner + host-side assets)
work/onnx/         EMPTY as of 2026-08-23 -- 16 GB of pure intermediates, deleted.
                   See work/onnx/README.md for the regeneration commands
work/calib/vae_enc/         64 real SSD1B first frames + nhwc/ copies (252 MB).
                            DELETABLE — regenerate with capture_first_frames.py (~3 min)
work/calib/vae_dec/         56 real calibration latents (keep — cheap, 8.8 MB)
work/calib/vae_dec_stream/  56 latents + 9 MemBlock states each. ~3.09 GB, states
                            currently NHWC (see .layout_nhwc sentinel).
                            DELETABLE — regenerate with make_stream_calib.py (~5 min)
work/calib/mmdit_s0r,_s1f/  kept: needed to re-quantise the MMDiT for the AdaLN fix.
                            The superseded mmdit, mmdit_s0 and mmdit_s0f sets were
                            deleted 2026-08-23
docs/ notes/       findings and session history
```

## Environment

Full install steps are in `SETUP.md`. What the scripts assume:

- **Windows host** with Python 3.10 (torch 2.8+cu129, onnx, onnxruntime, transformers 4.55.3,
  diffusers, timm, sentencepiece, onnxconverter_common). Invoke as `py -3.10`.
- **WSL Ubuntu** with a Python 3.10 venv for the QAIRT tools (numpy 1.26.4 = the SDK's tested
  pin; the SDK's `libPyIrGraph310` needs 3.10, not the system 3.12). The conversion scripts
  source `work/qnn/qnn_env.sh` (copy it from `qnn_env.example.sh`) or `$QNN_ENV`.
- **`QNN_SDK_ROOT`** points at QAIRT SDK **2.49.0.260730**. Conversion targets
  `x86_64-linux-clang`, so it runs under **WSL**.
- **Target SoC** is `HTP_ARCH` / `SOC_MODEL`, default `v79` / `69` (SM8750). See
  `docs/porting-other-socs.md`.
- Run WSL as `wsl -d Ubuntu -e bash <script>`; `bash -lc "..."` and inline quoting both break.
- **Android NDK r26d (Linux) inside WSL** for `qnn-model-lib-generator -t aarch64-android`,
  the only route to device intermediates. A Windows NDK is not usable: lib-gen runs under
  WSL and shells out to `ndk-build`.
- **adb** on `PATH` (or `$env:ADB`). **Run it from PowerShell** — Git Bash mangles
  `/data/...` paths. If the phone is attached over both USB and wireless adb, always pass
  `-s <DEVICE_SERIAL>`.
- GPU: the reference pipeline runs on a **6 GB RTX 3050 laptop GPU** (~35 s/video,
  oversubscribing via WDDM rather than OOMing). Conversion is RAM-bound, not GPU-bound:
  stage 2 of the MMDiT needed WSL at 11 GB + 48 GB swap.
- Reference device: Samsung S25 Ultra (SM-S938B), SM8750, HTP **V79**. The desktop harness
  lives at `/data/local/tmp/nd` (stage it with `work/device/push_harness.ps1`); the APK's own
  models live in `/sdcard/Android/data/com.neodragon.demo/files/nd/ctx/`.
- Disk: budget **~50 GB** for weights, ONNX intermediates, calibration sets and binaries.

## The 24x VAE gap: found and closed (2026-08-21)

**Compute was never the blocker.** An older version of this section said the 6 GB VRAM gap
was "the reason this is paused"; that was already false when written. The RTX 3050 runs the
reference pipeline fine (~35 s/video). The project was not stuck -- it ran out of session on
2026-08-14 and sat for a week.

**The real blocker was layout-transpose traffic, and it is now fixed and measured on device.**

`TGrow` synthesises the time axis by reshaping the channel axis into the batch axis. That is
a free view in NCHW and a full scatter in NHWC -- and HTP runs convolution channel-last. The
converter therefore wrapped every `TGrow` in a transpose pair on the largest tensors in the
graph. Rewriting the decoder to carry the 9 MemBlock states explicitly, never folding time
into batch (`work/export/export_vae_decoder_stream.py`):

| | T=1 parallel | **vaedecsn (shipping)** | paper |
|---|---|---|---|
| Transpose bytes / inference | 779.80 MB | **7.95 MB** | — |
| transpose / compute ratio | 1.16 | **0.01** | — |
| deploy SNR (held-out) | 31.55 dB | **34.27 dB** | 35 dB |
| **latency per invocation** | **6.10 s** | **100.18 ms** | 248.9 ms |
| usable frames per invocation | 1 | **8** | — |
| **49-frame video decode** | **298.9 s** | **0.70 s** | — |

**60.9x per invocation, 487x per usable frame, and +2.72 dB accuracy.** Same 205.07 GMAC in
every build -- the entire win was layout and discarded work, not arithmetic.

**The shipping artefact is `work/device/vaedecsn_v79.bin`** (12.1 MB). It expects its 9
MemBlock state tensors **channel-last (NHWC)**; `latent` and `video` stay NCHW. Host-side
raws are permuted by `work/device/permute_states_nhwc.py` (sentinel `.layout_nhwc`).
`vaedecs_v79.bin` is the same graph with NCHW states, 10% slower, kept only for A/B.

Eliminated along the way: `soc_model: 69` is **correct** (`QNN_SOC_MODEL_SM8750 = 69`,
`QnnTypes.h:1872`); there is **zero float fallback**; and **spilling is not a proxy for
anything** -- spill_bytes nearly tripled (215 -> 482 MB) while latency fell 53x, so
`vtcm_mb` never needed tuning.

Full detail in `docs/phase4-vae-decoder.md`.

## Phased plan (from brief §7, unchanged)

| Phase | Task | Est. |
|---|---|---|
| 1 | DistilT5 → ONNX → QNN | 85% |
| 2 | QuickSRNet integration (pre-compiled QNN binaries exist on AI Hub) | 90% |
| 3 | SSD1B UNet conversion — closest to existing ControlNet/inpaint work | 70% |
| 4 | TinyAEHV decoder conversion | 65% |
| 4b | *(absent from the brief)* CausalVaeEncoder conversion | — |
| 5 | Pyramidal MMDiT: 3 static graphs + quantization | 55-60% |
| 6 | Full E2E on device within latency/RAM budget | 30-40% |

Phases 1, 2 and 4 are the cheap early wins and need no GPU at all. **4b was missing from the
brief entirely** and is on the critical path; it is now done, and turned out to be cheaper
than 4.

The row above is the brief's original wording, kept verbatim. On Phase 5 the brief's
*approach* — "3 static graphs, padding+masking, RoPE reduction, per-stage calibration"
(brief §5) — is confirmed correct; what it does not say is that there are **18** distinct
token counts collapsing into those 3 padded graphs, and where the padding has to go. See
`docs/phase5-mmdit-scope.md`.

## Traps — 41 of them, in `docs/traps.md`

**Read `docs/traps.md` before converting a model, touching layout or quantisation, or
measuring latency on device.** The ones that cost the most time, as a triage list:

| # | one line | bites when |
|---|---|---|
| 4 | calibration must use REAL pipeline tensors, states included | any conversion |
| 7 | `--preserve_io layout` is mandatory for conv models | any conv conversion |
| 8 | `--use_native_input_files` for float graphs, omit for quantised | every device run |
| 10 | the DSP throttles — interleave A/B in ONE session, min-of-N, never averages | every latency claim |
| 11 | never synthesise a time/batch axis by reshaping channels | any 5-D rewrite |
| 23 | rank is not an escape from layout rules — `NONTRIVIAL` is | attention graphs |
| 26 | `torch.stack(dim=-1)` is an element interleave | any RoPE/pair layout |
| 31 | a tensor feeding `chunk`/`split` gets ONE encoding, sized by its largest part | AdaLN-style heads |
| 33 | an app cannot exec a helper binary onto the DSP — dlopen it in-process | any Android port |
| 35 | `mmap` context binaries, never heap-copy | any Android port |
| 40 | a fix trading a LINEAR cost for a QUADRATIC one inverts at long sequences | shape-dependent optimisation |
| 41 | the MMDiT is bandwidth-bound; op COUNT is a dead lever | MMDiT latency work |
| 42 | per-op cycles/byte vary 20x — an aggregate ms/MB cannot price an op-MIX change | any byte-based projection |
| 43 | instrument LOAD separately from EXECUTE before comparing to published latency | any wall-clock comparison |


## Open decisions

**Resolved:** compute path — the local RTX 3050 6 GB runs the reference pipeline fine
(~35 s/video, 7 min for 8 prompts). No rental or Kaggle needed. Workspace layout — everything
lives under `Neodragon/`; keep clear of `local-dream/`, which is CC BY-NC 4.0 and must never
be published.

**Resolved 2026-08-21 (same day it was raised):** the VAE encoder. It really was a missing
phase — `generation_utils.py:531` encodes the SSD1B first frame back to latent space, and the
plan only ever listed the decoder — but it was **not** the hard half. The `deque` feature
cache lives solely on `CausalConv3d`'s `temporal_chunk=True` branch, which the AR t2v path
never takes; and at T=1 every causal conv front-pads two planes of zeros, so the whole
encoder collapses **exactly** to a 2-D convnet on `w[:, :, -1]`. Done at 41.60 dB /
166.83 ms. See `docs/phase4b-vae-encoder.md`.

Still open:

- **Phase 6 is DONE** (2026-08-23) — the app runs the whole pipeline on device. What is
  left there is polish, not risk: nothing writes an MP4 (playback is a Compose frame
  stepper), and the app has no model-download path (binaries are pushed over adb).

- **Phase 5 is BUILT — all three stages** (2026-08-23). The scoping corrections that made
  it possible are in `docs/phase5-mmdit-scope.md`, the latency work in
  `docs/phase5-mmdit-latency.md`, the accuracy work in `notes/2026-08-23-adaln-fix.md`.
  Final standing:

  | stage | tokens | accuracy | target | latency | paper |
  |---|---:|---:|---:|---:|---:|
  | 0 | 408 | 27.89 dB | 29.0 | 169.2 ms | 104.7 |
  | 1 | 648 | **23.99 dB** | 22.0 ✔ | 375.8 ms | 218.3 |
  | 2 | 1728 | 20.01 dB | 24.0 | **1851 ms** | 938.3 |

  **What remains open is stage 2, and it is one problem, not two:** it has the longest
  sequence (1728 tokens), the worst accuracy gap (−3.99 dB), the worst latency ratio
  (1.97×), and it is **11.1 s of the 23.5 s video — 47%**. Accuracy falls with sequence
  length (27.89 / 23.99 / 20.01), which is the residual-accumulation signature. Latency is
  dispatch-bound, not arithmetic-bound (2378 ops at ~129K cycles on stage 0; stage 2's
  graph is already the cleanest of the three at transpose/compute **0.06**).
  Converter wall times were 38:32 / 48:43 / **2:52:07** — budget ~3 h for stage 2.

- **0.73 dB on the VAE decoder** — 34.27 vs the paper's 35 dB. Low priority; mean pixel error
  is 1.25/255. Route is `--algorithms` / AdaRound or AIMET encodings
- **Phase 3 (SSD1B UNet)** — conv-heavy, so it is the module most likely to inherit trap #11.
  Run `analyze_net.py` on it before pushing anything to the device
- **Overnight autonomy depth** — unchanged; longest compile so far is ~12 min (W8A16
  calibration of the streaming decoder)

## Working rules

- Reuse the `work/qnn/convert_*.sh` scripts as the conversion template; do not write a new pipeline from scratch
- Conversion runs in WSL (`x86_64-linux-clang`), adb runs in PowerShell
- Measure before concluding. Generation is seed-reproducible, so single-run A/B is valid
- Keep this file an index — findings go in `docs/`, session history in `notes/`
