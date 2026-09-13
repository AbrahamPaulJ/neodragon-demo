# Neodragon On-Device Port — Research Brief

**Target:** Samsung Galaxy S25 Ultra (Snapdragon 8 Elite Gen4, Hexagon NPU)
**Context:** Extension of `controlnet-mobile` — same QAIRT/local-dream deployment pattern already used for SD ControlNet (canny) and inpainting.
**Prior art established by user:** working ControlNet-canny and inpaint pipelines on-device via QAIRT, 512x512 @ ~6.6s/20 steps.

---

## 1. Decision: Neodragon over MobileWan

Two Qualcomm AI Research mobile video models were evaluated. Neodragon was selected.

| | Neodragon | MobileWan |
|---|---|---|
| Release | Nov 2025 (ICLR 2026) | Jul 2026 |
| Benchmark chip | **Snapdragon 8 Elite Gen4** (same generation as S25 Ultra) | Snapdragon 8 Elite **Gen5** (newer, ~37% faster NPU — not representative of our device) |
| Architecture | 4-stage pipeline: DistilT5 text encoder → SSD1B first-frame gen → pyramidal MMDiT denoiser → TinyAEHV decoder → QuickSRNet 2x SR | Monolithic 5B DiT, custom causal/chunked linear attention |
| VAE decoder | Custom lightweight TinyAEHV, ported and benchmarked on-device | **Not released** — falls back to stock heavy Wan2.2 decoder |
| Overlap with existing skillset | SSD1B is a distilled SDXL UNet — directly analogous to prior ControlNet/inpaint work; QuickSRNet has pre-compiled QNN `.so` binaries already on Qualcomm AI Hub | None |
| Deployment code released | No (GitHub repo is PyTorch/GPU sampling only) — but the paper documents the exact on-device compilation recipe used | No, and no equivalent recipe documented |
| Resolution / length | 320x512 native, 640x1024 with SR, 49 frames @ 24fps (2s) | 480x832, 81 frames @ 16fps (5s) |
| License | BSD-3-Clause-Clear | (check on release) |

**Bottom line:** Neodragon is better documented, targets our exact chip generation, and reuses components we already have working muscle memory for.

---

## 2. Resources

- Paper: https://arxiv.org/abs/2511.06055 (full HTML: https://arxiv.org/html/2511.06055v1 — **Section 4 "End-to-End Integration" and 4.1 "Model Compilation" are the critical reference for QNN porting**)
- Code (PyTorch/GPU sampling only, no QNN code): https://github.com/qualcomm-ai-research/neodragon
- Weights: https://huggingface.co/karnewar/Neodragon
- Project page: https://qualcomm-ai-research.github.io/neodragon
- QuickSRNet (pre-compiled QNN assets, Snapdragon 8 Elite ready): https://huggingface.co/qualcomm/QuickSRNetMedium — via Qualcomm AI Hub, free for developers
- Qualcomm AI Hub (free tier, for compile/profile jobs and hosted device testing): https://aihub.qualcomm.com

---

## 3. The core technical challenge: pyramidal flow matching vs. fixed-graph NPU

Neodragon's denoiser is built on Pyramidal Flow Matching: the same MMDiT weights process video at multiple spatial resolutions within one denoising trajectory (S=3 stages: low → mid → high res, only the final stage runs at full resolution), plus a temporal pyramid where autoregressive history frames are spatially downsampled with distance into the past.

**Problem:** QNN/Hexagon HTP requires fully static tensor shapes at ahead-of-time compile time. It has no support for a single graph handling variable resolutions or variable-length sequences.

**How Qualcomm actually solved it (documented in paper Section 4.1) — this is the reproducible recipe:**

1. **Compile 3 separate static DiT graphs** (low/mid/high resolution — shapes `[7×10×16]`, `[7×20×32]`, `[7×40×64]`), not one dynamic graph. Same "chain of separately-compiled fixed submodels" pattern already used in controlnet-mobile.
2. **Autoregressive history handled via fixed-length padding + masking**, not per-length graphs: expand to the last-frame graph shape, zero-pad earlier frames, modify attention mask and positional embeddings so padding doesn't contribute.
3. **Precompute DiT input constants** per stage (input shapes, token lengths, timestep embeddings) since the graph is static.
4. **RoPE 6D-tensor reduction — the single biggest compile-time trap.** Naive RoPE computation across temporal+spatial dims produced high-dimensional tiling that pushed DiT-high graph compile time to **near 24 hours**. Reducing along the sin/cos and broadcast dimensions cut this to **under 2 hours** and took inference latency from seconds to sub-second. Budget time to hit this wall before finding the fix, or replicate the reduction proactively.
5. **Two numerical-precision fixes needed for on-device stability:**
   - Default causal mask value (T5 and DiT) is too large for the fixed-point runtime — rescale to a value large enough to mask correctly but small enough to avoid overflow.
   - T5's residual-add and feed-forward-add operations exceed FP16 range — apply a scaling factor to residual connections without changing model behavior.

---

## 4. Quantization scheme (from paper Table 9 — use as starting reference)

W8A16 PTQ via AIMET (AI Model Efficiency Toolkit) for all modules except QuickSRNet (AdaRound). **Critically: separate calibration sets per pyramid stage** — reusing one calibration set across stages risks bad quantization since noise statistics differ drastically between the noisy low-res early stage and the near-clean full-res final stage.

| Module | Calibration samples | Deploy SNR |
|---|---|---|
| VAE Enc / Dec | 50 / 50 | 40dB / 35dB |
| SSD1B UNet / Dec | 500 / 500 | 33dB / 31dB |
| MMDiT stage [7×10×16] | 300 | 29dB |
| MMDiT stage [7×20×32] | 300 | 22dB (worst — not monotonic across stages) |
| MMDiT stage [7×40×64] | 300 | 24dB |
| QuickSRNet | 500 | 48dB (AdaRound) |

Quantization error compounds across the pipeline stages — the paper notes reducing to 1 timestep per stage (via step distillation, already baked into the released checkpoint) was what kept compounding error in check.

---

## 5. On-device benchmarks — Snapdragon 8 Elite Gen4 (= our exact chip)

From paper Table 7. **Note: the paper's headline "6.7s E2E" figure is reported only for the laptop SoC (Snapdragon X Elite) — no equivalent full E2E figure is given for the phone chip.**

| Component | Latency (8 Elite Gen4) |
|---|---|
| CLIP-L / CLIP-G text encode | 14.0ms / 76.5ms |
| DistilT5 | 3.5ms |
| VAE Encoder (first frame only) | 1206.5ms |
| VAE Decoder | 248.9ms |
| SSD1B UNet | 234.6ms |
| SSD1B Decoder | 580.0ms |
| MMDiT+CA @ [7×10×16] | 104.7ms |
| MMDiT+CA @ [7×20×32] | 218.3ms |
| MMDiT+CA @ [7×40×64] | 938.3ms |

Sum of one denoising pass across all components ≈ 3.6s. Phone components ran consistently ~1.2-2x slower than the laptop chip on the same table, so realistic full E2E on S25 Ultra is plausibly **8-10s+**, not the 6.7s laptop figure — plan around this, don't market against it.

---

## 6. Compute requirements — no rented GPU needed

All expensive training (H100-cluster-scale: text-encoder distillation, decoder distillation, block-pruning fine-tune, step-distillation) was already done by Qualcomm and is baked into the released checkpoint. Remaining work is inference-scale only:

- PyTorch reference pipeline validation (~5-6GB bf16 weights): fits on RTX 3060 12GB or Kaggle 2x T4.
- ONNX export/debug: CPU/GPU light.
- AIMET PTQ calibration (300-500 samples/module): a few hours on a single consumer GPU.
- QNN graph compilation (`qnn-context-binary-generator`): CPU-bound on x86 host, not GPU-bound.
- On-device testing: physical S25 Ultra via Termux/ADB.

**Money, worst case: ~$0-100.** QAIRT SDK and Qualcomm AI Hub are free for individual developers. Only cost is electricity or optional burst-rental of a consumer GPU for parallelization (not required).

**Time, worst case: 4-8 weeks realistic, 3 months ceiling** if genuine blockers hit (unsupported op in current QAIRT SDK version, undocumented numerical issues beyond the two already known). Coding-assistant help (Opus, etc.) speeds up code/debug turnaround but does not reduce hard wall-clock costs: compile waits, calibration runs, on-device flash-test cycles.

---

## 7. Phased plan with estimated success rates

| Phase | Task | Est. success rate | Notes |
|---|---|---|---|
| 1 | DistilT5 text encoder → ONNX → QNN | 85% | Small standard transformer |
| 2 | QuickSRNet integration | 90% | Pre-compiled QNN binaries already exist on AI Hub |
| 3 | SSD1B UNet conversion | 70% | Same class of problem as existing ControlNet/inpaint SDXL work |
| 4 | TinyAEHV decoder conversion | 65% | Custom architecture, less precedent |
| 5 | Pyramidal MMDiT conversion + quantization (3 stages) | ~55-60% (revised up from initial 35% once Qualcomm's documented recipe was found) | Follow Section 4.1 recipe: 3 static graphs, padding+masking, RoPE reduction, mask/residual numerical fixes, per-stage calibration |
| 6 | Full pipeline stitched, running E2E on S25 Ultra within reasonable latency/RAM | 30-40% | Compounding risk across stages; budget for the AIMET-vs-existing-toolchain gap |

---

## 8. Key risks / blockers to watch for

1. **RoPE 6D-tensor compile blowup** — apply the sin/cos/broadcast dimension reduction proactively rather than discovering it via a day-long compile.
2. **Causal mask value overflow** on fixed-point runtime — check mask constant magnitude before first compile attempt.
3. **T5 residual FP16 overflow** — check residual-add magnitudes in DistilT5/MMDiT before quantizing.
4. **Per-stage calibration data** — do not reuse one calibration set across the 3 MMDiT resolution stages.
5. **AIMET toolchain gap** — paper's PTQ pipeline used AIMET; confirm whether existing QAIRT/local-dream workflow already covers this or an equivalent quantization path needs setting up first.
6. **No official on-device deployment code exists** — the GitHub repo is PyTorch/GPU inference only. Everything in Section 4 must be independently reproduced from the paper's description, not copied from a released script.
7. **VAE encoder cost (1.2s, first frame only)** — one-time cost per generation, not per-frame, but still the single largest individual latency line item on-device.
