# Neodragon on the Snapdragon NPU

Porting Qualcomm's [Neodragon](https://arxiv.org/abs/2511.06055) autoregressive
text-to-video model to the **Samsung S25 Ultra's Hexagon NPU** (SM8750, HTP v79) via
QAIRT, and shipping it as a standalone Android app.

**Prompt → 49-frame video, entirely on the phone, offline: 23.5–26.9 s.**
Photo → video, and prompt → 640×1024 image in 2.1–2.5 s warm. No server: the app downloads
its models once, then runs offline.


---

## What this is

Every module of the AR text-to-video path is exported, quantised and running on the NPU —
thirteen context binaries, ~8 GB, driven by an in-process QNN runner inside a normal
Android app. The repo carries the port and the reasoning: export rewrites, conversion
recipes, device tooling, the app, and a detailed record of what went wrong and how it was
diagnosed.

| module | accuracy vs fp32 | target | latency | paper |
|---|---:|---:|---:|---:|
| DistilT5 | 49.04 dB | — | 12.84 ms | 3.5 |
| VAE encoder | 41.60 dB | 40 | 166.83 ms | 1206.5 |
| VAE decoder | 34.27 dB | 35 | 100.18 ms / 8 frames | 248.9 |
| MMDiT stage 0 | **39.67 dB** | 29 ✔ | 169.2 ms | 104.7 |
| MMDiT stage 1 | **37.80 dB** | 22 ✔ | 375.8 ms | 218.3 |
| MMDiT stage 2 | **33.89 dB** | 24 ✔ | 1851 ms | 938.3 |
| SSD1B UNet | 32.51 dB | — | — | — |
| ContextAdapter | 39.36 dB | — | — | — |
| QuickSRNet 2× | 28.06 dB | 48 | ~4 ms/frame | — |

Peak app RSS is **238–268 MB** while driving ~8 GB of models (they are `mmap`ed).

## The short version of what was learned

> **Almost every win came from deleting structure from the graph, not from tuning the
> backend.** Not one came from a knob.

* `TGrow` synthesising a time axis by reshaping the channel axis cost **779.80 MB of layout
  traffic per inference and a 53.5× latency penalty** — free in NCHW, a full scatter in the
  NHWC that HTP actually runs.
* RoPE was **43.3%** of the MMDiT, and a single `torch.stack(..., dim=-1)` — an element
  interleave — was **28.7%** of the whole graph.
* A causal conv padding against a size-1 time axis is multiplication by zero: deleting it
  took the VAE encoder from 1071.2 to **363.33 GMAC**.
* Meanwhile `vtcm_mb` never needed tuning (spill volume nearly *tripled* while latency fell
  53×), hand-converting to fp16 was 3.7× slower and 10 dB worse, and a confirmed 8-bit K/V
  bug bought **+0.01 dB for +18.6% latency**.

And the one that made the app possible: **an Android app cannot exec a helper binary onto
the DSP.** The byte-identical `qnn-net-run` works from `/data/local/tmp` and fails from
`/data/app` purely because of its SELinux context, so the backend has to be `dlopen`'d into
the app process.

47 such traps are catalogued with their measurements in [`docs/traps.md`](docs/traps.md).

## Where to start

| you want | read |
|---|---|
| current state and what to do next | [`HANDOFF.md`](HANDOFF.md) |
| to build and run it | [`SETUP.md`](SETUP.md) |
| **to port it to an 8 Gen 3 / 8 Gen 2 or older** | [`docs/porting-other-socs.md`](docs/porting-other-socs.md) |
| to rebuild every model from the weights | [`docs/rebuild-models.md`](docs/rebuild-models.md) |
| the plan, scored for effort and risk | [`docs/roadmap.md`](docs/roadmap.md) |
| the whole arc, for a writeup | [`docs/writeup-index.md`](docs/writeup-index.md) |
| the trap catalogue | [`docs/traps.md`](docs/traps.md) |
| the project index (file-by-file) | [`CLAUDE.md`](CLAUDE.md) |
| how the app works | [`notes/2026-08-23-android-e2e.md`](notes/2026-08-23-android-e2e.md) |

## Layout

```
work/export/     PyTorch -> ONNX rewrites (the interesting engineering)
work/qnn/        QAIRT conversion recipes, HTP configs
work/device/     device tooling: analyze_net, matmul_bits, layer_snr, compare_*
work/pipeline/   reference + NPU-backed drivers, calibration capture
work/audit/      shape/trap analysis
work/android/    the app: Compose UI, JNI QNN runner, Kotlin host-side port
docs/ notes/     findings, per-session history
```

Weights, converted binaries, the QAIRT SDK and the upstream clone are **not** in the repo —
see [`SETUP.md`](SETUP.md) to restore them.

## Status and honesty

**Only one device has ever run this: a Samsung S25 Ultra (SM8750, HTP v79).** Every
context binary is compiled for that Hexagon revision and will not load on an 8 Gen 3 or
older; the app detects this with a canary graph before downloading anything.
[`docs/porting-other-socs.md`](docs/porting-other-socs.md) is the route to other chips,
and nothing in it has been measured on one yet.

Stage 2 of the MMDiT is the latency weak point: 1.97× the paper, and nearly half the
video's wall clock. No perceptual metric has been run. QuickSRNet reaches 28.06 dB against
the paper's 48 (which uses AdaRound).

None of the paper's numbers were reproduced on the paper's own hardware; they are quoted
from its Tables 7 and 9.

## Licensing

The port — everything under `work/`, `docs/`, `notes/` — is released under the
[Clear BSD License](LICENSE) (BSD-3-Clause-Clear), the same license as upstream Neodragon.
Neodragon itself and its weights are **BSD-3-Clause-Clear** (Qualcomm AI Research). The
QAIRT SDK is Qualcomm's and is not redistributed here.
