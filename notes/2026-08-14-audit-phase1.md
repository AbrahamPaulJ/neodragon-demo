# 2026-08-14 — trap audit, Phase 1, paper read, Phase 4

> **Session 1 of 2. Superseded on several points — see `notes/2026-08-21-vae-streaming.md`.**
>
> - `soc_model: 69` is listed below as an unverified guess. It is **correct**
>   (`QNN_SOC_MODEL_SM8750 = 69`, `QnnTypes.h:1872`). Resolved.
> - The Phase 4 VAE decoder numbers here and in the doc's early sections (6.10 s, 31.55 dB)
>   are the **superseded T=1 parallel build**. The shipping build is the streaming decoder:
>   **114.04 ms, 34.27 dB, 8 frames per invocation.**
> - `~/neodragon-qnn` has since been **deleted**, so `work/qnn/setup_env.sh` and
>   `work/qnn/install_onnx.sh` are dead bootstrap scripts.
> - Disk is now ~33 GB free, not 88 GB.


> **Session ended on hold.** Went considerably further than the title suggests: Phase 1
> complete on device, the paper read at source (which corrected the plan), the reference
> pipeline running locally, and Phase 4 built through W8A16. See `docs/` for the numbers and
> `CLAUDE.md` for current state. The one blocking question is the **24× VAE latency gap**.
>
> Chronology: trap audit → Phase 1 export/convert/device → latency tuning → paper §4 read
> → Phase 4 export/convert/device → reference pipeline on the 3050 → Phase 4 W8A16.



First working session. Project moved from "planning only, nothing downloaded" to
"three traps defused, DistilT5 exported and verified".

## What now exists on disk

```
src/neodragon/          depth-1 clone of qualcomm-ai-research/neodragon (BSD-3-Clause-Clear)
work/models/neodragon/  text_encoder_3 (259 MB) + tokenizer_3 — the only weights pulled
work/audit/             audit_rope.py, audit_mask.py, audit_t5*.py
work/export/            export_distilt5.py, finalize_distilt5.py, graph_fixes.py
work/onnx/              distilt5_qnn.onnx  (522 nodes, 495 MB fp32)
work/qnn/               convert_distilt5.sh, htp_{backend,config}_v79.json
docs/trap-audit.md      findings, with the numbers
```

Workspace layout decision made by default, not deliberation: everything lives under
`Neodragon/` rather than `LocalDream/neodragon/`. Still worth revisiting — it is listed as
an open decision in CLAUDE.md.

## Environment corrections (CLAUDE.md is stale on these)

- **Windows Python 3.10.11 already has the full stack**: torch 2.8.0, onnx 1.14.1,
  onnxruntime 1.22.0, transformers 4.55.3, diffusers 0.35.1, einops 0.8.2, hf_hub 0.34.4.
  CLAUDE.md says "No onnx/diffusers/transformers/AIMET yet" — that was only true of 3.11/3.13.
  Added `sentencepiece` this session (needed by `T5Tokenizer`).
- **Disk is now 88 GB free on C:, not 108 GB.**
- **WSL Ubuntu already has a working QNN venv** at `~/npuconvert/.venv` — Python 3.10.20,
  numpy 1.26.4 (exactly the SDK's tested pin), onnx 1.18.0, torch 2.5.1+cpu, sourced via
  `~/npuconvert/qnn_env.sh`. This is the env to use.
- Ubuntu's *system* python3.12 has **neither pip nor ensurepip**, and the SDK's
  `check-python-dependency` refuses to run outside a venv. I bootstrapped a separate
  3.12 venv at `~/neodragon-qnn` before finding the 3.10 one; **it is redundant and can be
  deleted** (`rm -rf ~/neodragon-qnn`). `libPyIrGraph310` in the SDK wants 3.10 anyway.

## Findings that change the plan

See `docs/trap-audit.md` for the numbers. The two that matter:

1. **Trap #3 is misdescribed in the brief.** fp16 is not the failure mode — DistilT5 in fp16
   has rel_err 0.0026. The failure is A16 *resolution*: 0.6–2.3 effective bits under
   per-tensor quantisation. And residual scaling — the brief's prescribed fix — provably
   cannot help, because it divides peak and median equally. The real lever is the ~5 outlier
   channels (190, 589, 399, 127, 16, stable across all blocks), worth +8 bits per-channel.

2. **Trap #1's win is rank, not arithmetic.** The rank-4 rewrite does the same number of
   multiplies. Anyone framing it as a FLOP optimisation will not understand why it fixes a
   compile-time problem. Better still: the tensor is constant-foldable and need not be in
   the graph at all.

## Phase 1 result — PASSED

Full chain ran clean: ONNX → `qnn-onnx-converter` → `qnn-model-lib-generator` →
`qnn-context-binary-generator`, all exit 0.

| artefact | size |
|---|---|
| `distilt5.cpp` / `distilt5.bin` | 1.8 MB / 519 MB |
| `libdistilt5.so` | 520 MB |
| **`ctx/distilt5_v79.bin`** | **260 MB** |

Context binary metadata confirms the graph:

```
graph: distilt5
  Input   input_ids       [1, 128]        QNN_DATATYPE_INT_32
  Input   attention_mask  [1, 128]        QNN_DATATYPE_INT_32
  Output  prompt_embeds   [1, 128, 4096]  QNN_DATATYPE_FLOAT_32
```

The converter auto-cast the int64 inputs to int32 — no manual intervention needed.

Compile telemetry, worth keeping as the baseline for Phase 5 comparisons:

- Graph Sequencing for Target 391 ms · VTCM Allocation 45 ms ·
  Parallelization Optimization 240 ms · **total wall clock 4.9 s**
- `spill_bytes=0`, `fill_bytes=0` — no VTCM thrashing, the graph fits
- `read_total_bytes=210 MB`

4.9 seconds is the number to hold against the paper's near-24-hour DiT-high compile. It
says nothing about the DiT, but it does establish the toolchain is healthy end to end.

**This is the float build.** W8A16 quantisation is not done and needs calibration data —
and per trap #3, the T5 activation encoding is an open question, so quantising now would
bake in a bad choice. Nothing has been run on device (phone not connected).

## Open, in priority order
- **Per-channel activations on HTP** — if unavailable, a SmoothQuant-style migration of the
  5 outlier channels into consuming weights is the fallback. Settle before freezing the
  T5 encoding.
- **`soc_model: 69`** in `htp_config_v79.json` is an unverified guess for SM8750; `dsp_arch:
  v79` is confirmed from CLAUDE.md and the SDK's `lib/hexagon-v79/`.
- **MMDiT export** is the real Phase 5 cost and is untouched. `PyramidMMDiT.forward` takes
  `List[List[Tensor]]`, loops stages in Python, and boolean-scatters. A single-stage forward
  has to be written from scratch.
- **226 MB `pos_embed` buffer** must be pre-cropped per stage.
- Device not connected this session — nothing was run on hardware.
