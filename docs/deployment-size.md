# Deployment size — APK and models

*2026-08-21. Measured numbers are bold; the rest are projections from source weights.*

## On-device model payload

| component | params | precision | size | source |
|---|---|---|---|---|
| DistilT5 | 130 M | FP16 | **248.5 MB** | measured, `distilt5h_v79.bin` |
| CLIP L | 123 M | FP16 | ~235 MB | source `ssd_1b_text_encoder` on disk |
| CLIP G | 694 M | FP16 | ~1.33 GB | source `ssd_1b_text_encoder_2` on disk |
| Context adapter | ~120 M | W8A16 | ~125 MB | 497 MB fp32 on disk |
| **MMDiT, per stage** | 1512 M | W8A16 | **~1.5 GB** | measured, `mmdit_s0.bin` = 1.516 GB |
| MMDiT, 3 stages | | | **4.5 GB** | unless weights are shared — see below |
| SSD1B UNet | ~1.3 B | W8A16 | ~1.35 GB | 2.54 GB fp16 on disk |
| SSD1B VAE decoder | ~50 M | W8A16 | ~55 MB | 160 MB fp16 on disk |
| VAE encoder | 37 M | W8A16 | **39.7 MB** | measured, `vaeenc_v79.bin` |
| VAE decoder | ~6 M | W8A16 | **11.5 MB** | measured, `vaedecsn_v79.bin` |
| QuickSRNet | <1 M | W8A16 | ~2 MB | AI Hub pre-built |
| **total** | | | **~7.9 GB** | |

The three text encoders are FP16 by design, not by omission — paper Table 9, and
`docs/paper-notes.md` records why that correction matters.

## The weight-sharing question — worth ~2.9 GB

The three MMDiT stage graphs are **the same 1512 M weights** at three different input
shapes. They differ only in sequence length. Yet a QNN context binary embeds its own copy of
the weights, so naively that is 4.5 GB for one model.

`qnn-context-binary-generator --help` says:

> To create a context binary with multiple graphs, use comma-separated list of model.so files.

If composing the three stage graphs into a single context deduplicates identical weight
tensors, **4.5 GB → ~1.6 GB and the pipeline total drops to ~5.0 GB.** Unverified: it needs
stages 1 and 2 to exist first. There is also a separate
`--reference_weight_sharing_enabled_override`, but that targets weight sharing *across SoCs*
and is DLC-workflow only, so it is not the relevant mechanism here.

This is not only a storage question. The AR loop visits stage 0, 1 and 2 in sequence **every
unit**, six times per video, so either all three are resident or every unit pays three
context loads. Resolve it before quoting an E2E latency figure.

## APK

| part | size | source |
|---|---|---|
| QNN runtime `lib/` | 85.9 MB | **measured on device** |
| QNN runtime `dsp/` (HTP V79 skel) | 11.5 MB | **measured on device** |
| app code, UI, tokenizers | ~20–40 MB | estimate |
| **APK total** | **~120–140 MB** | |

The models cannot ship inside the APK: Google Play caps the base APK at 200 MB, and
install-time asset packs at 4 GB total. At ~7.9 GB the payload has to be a first-run
download, which is what LocalDream does.

## Reducing it, if it ever matters

Roughly in order of value per unit of effort:

1. **MMDiT weight sharing across the three stage contexts** — up to 2.9 GB, and it is a
   packaging question rather than a model change.
2. **CLIP G at 1.33 GB is the largest single FP16 blob.** Table 9 says FP16 for all three
   text encoders, so quantising it departs from the reference configuration — but it runs
   once per video, so any accuracy cost is not compounded through the AR loop.
3. **W4 on the MMDiT.** `--pack_4_bit_weights` exists. It would halve the largest component,
   at obvious risk to the stage SNRs, which Table 9 already puts at 22–29 dB, the worst in
   the pipeline. Last resort.
