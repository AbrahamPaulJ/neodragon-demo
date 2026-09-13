# Paper §4 read at source — verified against the brief

Read 2026-08-14 from `2511.06055v1 arxiv paper.pdf` (36 pages), text extracted with PyMuPDF
to `work/paper/paper.txt`. Until now everything about "the paper says" in this project was
relayed second-hand through `neodragon-port-research-brief.md`. This is the primary source.

## Verdict

**§4.1 is accurate in the brief. §4.2 / Table 9 is not — the brief omits a fact that changes
the plan.**

---

## §4.1 Model Compilation — five steps, all confirmed

Quoted, not paraphrased:

1. **Porting multi-resolution DiTs.** "we need to port 3 DiT graphs-low, mid, and high…
   the PyTorch model takes past history which dynamically grows… Specifically, we expand
   the last frame graph and pad zeros when running the inference for earlier frames. Doing
   such requires changes to the attention mask and positional embedding implementation such
   that the zero paddings are not contributing to the next frame generation."

   Note the precise shape: **one graph per resolution, sized for the last frame** (maximum
   history), zero-padded for earlier frames. Three graphs total — not three per frame count.

2. **Precomputation of DiT inputs.** "the input merge function of DiT's forward pass computes
   constant information for each stage. These include input shapes and trainable token
   lengths… as we run a fixed amount of timesteps, we also have timestep embeddings
   precomputed as **inputs** to the DiT graph."

   Timestep embeddings are graph *inputs*, not baked-in constants.

3. **Reduction of 6D tensors.** "the RoPE layer computation 6D tensor mul/add. Unlike torch
   dynamic graphs, high dimensional inputs means more complicated **tiling** which usually
   ends up **penalizing performance**… we reduce along the sin/cos and broadcast dimensions.
   This… reduces DiT-high compilation time from near a day to <2h. Gaining latency
   performance from seconds to sub-second."

   This confirms our audit exactly, including the framing. The mechanism is **tiling**, not
   arithmetic — `trap-audit.md` §1 measured that the multiply count is unchanged and the win
   is rank 6→4. "Reduce along the sin/cos and broadcast dimensions" is precisely the
   cos/sin split plus dropping the size-1 head broadcast dim that we implemented and verified
   bit-exact.

4. **Optimizing causal mask value.** "the causal mask value in T5 and DiT is set to an
   extremely large number… we adjust the mask value to a more suitable number—large enough
   to prevent the model from attending to masked tokens, but small enough to avoid overflow."

   Our `-100` satisfies both halves quantitatively: leaked attention weight 2.8e-42 ("large
   enough") and finite ("small enough").

5. **Rescaling activations in T5.** "T5 includes res_add and ff_add operations with values
   exceeding the **FP16** numerical range. To ensure numerical stability and maintain
   functional equivalence, we apply a scaling factor to these residual connections,
   effectively transforming them without altering the model behavior."

   Exactly the fix we deployed (S=16 on both `res_add` and `ff_add`, rel_err 1.2e-06 =
   "without altering the model behavior"). Our device A/B independently rediscovered this:
   NaN → 49 dB.

---

## §4.2 Pipeline Quantization — Table 9 in full

| Module | Quant scheme | Calib size | deploy SNR |
|---|---|---|---|
| CLIP L | **FP16** | NA | NA |
| CLIP G | **FP16** | NA | NA |
| **DistilT5** | **FP16** | **NA** | **NA** |
| VAE Enc | W8A16, PTQ | 50 | 40 dB |
| VAE Dec | W8A16, PTQ | 50 | 35 dB |
| SSD1B UNet | W8A16, PTQ | 500 | 33 dB |
| SSD1B Dec | W8A16, PTQ | 500 | 31 dB |
| MMDiT+CA [7×10×16] | W8A16, PTQ | 300 | 29 dB |
| MMDiT+CA [7×20×32] | W8A16, PTQ | 300 | 22 dB |
| MMDiT+CA [7×40×64] | W8A16, PTQ | 300 | 24 dB |
| QuickSRNet | W8A16, **AdaRound** | 500 | 48 dB |

> ### The correction that matters
>
> **All three text encoders — CLIP L, CLIP G, and DistilT5 — are deployed in FP16, not
> W8A16.** The research brief's §4 table omits these three rows entirely and states "W8A16
> PTQ via AIMET for all modules except QuickSRNet", which is wrong.
>
> Two consequences:
>
> 1. **Do not quantise DistilT5.** Our FP16 + residual-scaling build *is* the reference
>    deployment configuration. Phase 1 is architecturally complete, not a stepping stone.
> 2. **The A16 resolution problem in `trap-audit.md` §3 does not apply to DistilT5.** The
>    0.6–2.3 effective-bits finding is real arithmetic, but it describes a scenario the
>    reference design never enters. It stays relevant only as a warning for the MMDiT, which
>    *is* W8A16 and *does* carry a residual stream.

Other §4.2 facts:

- AIMET is the PTQ toolkit; AdaRound is applied to QuickSRNet only, "which ends up adding
  7+dB SQNR".
- SQNR is measured "between original FP models vs. the deployed models" — the same quantity
  our `compare2.py` computes as `20·log10(‖ref‖/‖ref−got‖)`, so our numbers are directly
  comparable in *metric*, though ours is FP32-vs-FP16 rather than FP-vs-W8A16.
- Compounding is the stated headline risk: "the quantization loss, despite minimal for each
  model, compounds quickly… Reduction to only 1 timestep on each stage doesn't just help
  reduce latency but also greatly mitigates the compounding quantization loss."

---

## Table 7 — on-device latency, confirmed verbatim

| Component | X Elite (laptop) | **8 Elite Gen4 (our chip)** |
|---|---|---|
| CLIP L / CLIP G | 5.9 / 43.6 ms | 14.0 / 76.5 ms |
| **DistilT5** | 3.0 ms | **3.5 ms** |
| VAE Enc / Dec | 941.7 / 143.0 ms | 1206.5 / 248.9 ms |
| SSD1B UNet / Dec | 151.5 / 378.6 ms | 234.6 / 580.0 ms |
| MMDiT+CA [7×10×16] | 54.9 ms | 104.7 ms |
| MMDiT+CA [7×20×32] | 101.4 ms | 218.3 ms |
| MMDiT+CA [7×40×64] | 590.2 ms | 938.3 ms |
| QuickSRNet | 4.9 ms | 6.5 ms |

The brief was right that the **6.7 s E2E headline is Snapdragon X Elite only** — no E2E
figure is given for the phone.

Peak RAM (X Elite column, worth noting for the phone budget): MMDiT [7×40×64] **3.25 GB**,
SSD1B UNet 2.57 GB, CLIP G 2.64 GB.

---

## What this does to our 18.7 ms

Since the paper's DistilT5 is **also FP16**, the comparison is like-for-like and the gap is
**real, not a precision artefact**:

| | latency |
|---|---|
| Paper, DistilT5, FP16, 8 Elite Gen4 | **3.5 ms** |
| Ours, DistilT5, FP16, 8 Elite Gen4 (accelerator time) | **18.7 ms** |

**5.3× slower, unexplained.** Candidate causes, cheapest first:

1. **Unfused normalisation.** Our graph carries 37 `Pow`, 25 `ReduceMean`, 25 `Sqrt`, 25
   `Div` — T5's RMSNorm left decomposed instead of matched to an HTP fused op. This is the
   prime suspect.
2. Context binary is 260 MB for a ~130 M-param model, i.e. fp16 weights — so weight
   precision is already right and is *not* the cause.
3. `vtcm_mb: 8` and `O: 3.0` were inherited from the 8gen2 SD config and never tuned for V79.
4. Our `soc_model: 69` is still an unverified guess.

None of this blocks anything — 18.7 ms against a 6.7 s pipeline is noise. It matters as a
signal about how much performance the default conversion path leaves on the table, which
*will* matter for the 938 ms MMDiT-high stage.
