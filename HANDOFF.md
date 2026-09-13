# HANDOFF — start here

*Rewritten at the end of session 7, 2026-08-23. This file is replaced each session rather
than appended to, so it never disagrees with itself.*

---

## State in one paragraph

Neodragon runs end to end on the S25 Ultra from a standalone APK: **prompt → 49-frame
video**, **photo → video** (image-to-video), **prompt → 640×1024 image**, all offline. The
app downloads its own models from a **public** Hugging Face repo, refuses to download on a
device that cannot run them, upscales the video 2× with QuickSRNet, and writes a real MP4
to the gallery. Every module in the plan is converted. **All three MMDiT stages now exceed
their paper accuracy targets for the first time.**

## Numbers that matter

| module | accuracy | target | latency |
|---|---:|---:|---:|
| DistilT5 | 49.04 dB | — | 12.84 ms |
| VAE encoder | 41.60 dB | 40 | 166.83 ms |
| VAE decoder | 34.27 dB | 35 | 100.18 ms / 8 frames |
| **MMDiT stage 0** | **39.67 dB** | 29 ✔ | ~169 ms |
| **MMDiT stage 1** | **37.80 dB** | 22 ✔ | ~376 ms |
| **MMDiT stage 2** | **33.89 dB** | 24 ✔ | ~1851 ms |
| SSD1B UNet | 32.51 dB | — | ~300 ms × 4 |
| QuickSRNet 2× | 28.06 dB | 48 | ~4 ms/frame |

**Wall clock:** video 23.5–26.9 s · image **2.1–2.5 s warm** (≈5 s on the first tap of a
cold app) · peak RSS **238–268 MB**.

## What session 7 changed

1. **The fused-score MMDiT rebuild (trap #40).** Contracting attention once at full width
   instead of as two half-width MatMuls plus an add. Exact in fp32 (124–127 dB against the
   stock model), padding transparent across all 18 unit/stage combinations. Worth
   **+10.9 / +12.2 / +12.0 dB** and **−4.4% cycles**. Binaries `mmdit_s0fs`, `s1fs`, `s2fs`.
2. **The first frame was never 4.7× the paper** — that was model loading plus a cold-cache
   measurement. Compute is **1.18×**, essentially at parity. Keeping CLIP G resident and
   preloading in the background took it 7.5 s → **2.1–2.5 s** at *lower* RSS.
3. **Image-to-video**, with a pan/zoom crop dialog.
4. **QuickSRNet 2×** (Phase 2 — the last unconverted module). 320×512 → 640×1024.
5. **MP4 export** to Movies/Neodragon, and a **device-support canary** that gates the
   download.

## The one thing to do next

**Softmax is 27.2% of stage 2 and nothing has touched it.** At **0.221 cycles/byte** for
what should be 4–5 passes over the data, it looks like an unfused or badly-quantised
kernel rather than an intrinsic cost. The broadcast mask add is another **15.8%** (a
`[24,1728,1728] + [1,1728,1728]` broadcast costing **14×** a plain add of the same output
shape). Together **43% of the graph**, both operating on the score matrix.

`docs/phase5-stage2-bandwidth.md` has the full per-op breakdown and the tooling
(`prof_by_shape.py`, `bytes_model.py`, `score_traffic.py`).

## What NOT to do

* **Do not chase byte reductions in the MMDiT.** That lever is spent — trap #42. Removing
  28% of the graph's bytes bought 4.4% of its cycles.
* **Do not treat op COUNT as a latency lever** — trap #41. Measured: −27.6% ops, −1.7% time.
* **Do not re-run the seven dead accuracy hypotheses** (`notes/2026-08-23-HANDOFF.md`), and
  note that the "accuracy falls with sequence length → residual accumulation" diagnosis was
  **wrong**: it was the split score contraction, and the fix is uniform across all three
  stages.
* **Do not trust a latency delta under 10%** without pinning the clock. Session 7's A/B
  showed a 63% within-build spread across six rounds with 45 s cooldowns.

## Reading order

| when | read |
|---|---|
| resuming | this file, then `docs/roadmap.md` |
| about to convert / measure / debug something odd | **`docs/traps.md`** (47 traps) |
| MMDiT latency | `docs/phase5-stage2-bandwidth.md` |
| whole-pipeline budget | `docs/e2e-budget.md` |
| the app | `notes/2026-08-23-android-e2e.md`, then `notes/2026-08-23-session7-app.md` |
| writing it up | `docs/writeup-index.md` |

`CLAUDE.md` is the index and carries a 14-row triage table of the load-bearing traps.

## Rebuild and ship

```powershell
# APK
cd work\android; .\gradlew.bat assembleDebug
adb -s <DEVICE_SERIAL> install -r app\build\outputs\apk\debug\app-debug.apk
# a prebuilt copy lives at work\android\dist\neodragon-debug.apk

# models: the app downloads them itself from the public repo
#   https://huggingface.co/AbrahamPJ/neodragon-npu-s25u
```

**Models are no longer pushed by hand.** The app fetches them, verifies each by byte
length, and resumes with HTTP Range. The phone was wiped clean at the end of session 7 —
`/data/local/tmp/nd` (the desktop harness) and the app's `ctx/` are both gone, 21 GB
reclaimed. Re-stage the harness with `work/device/push_harness.ps1` if you need
`qnn-net-run` again.

## Environment reminders

* Conversion runs in **WSL** (`source work/qnn/qnn_env.sh`); adb runs in **PowerShell**
  — **the Bash tool mangles `/data/...` paths and reports success**.
* Any list consumed by the QNN toolchain must be **LF-only, BOM-free, and use `/mnt/c/...`
  paths** — Windows paths make the converter report every file as missing.
* `adb tcpip 5555` **does not survive a reboot**; re-arm over USB.
* The WSL vhdx is **103 GB on disk for 48 GB of data**. Deleting inside WSL never returns
  space to C:. `wsl --shutdown && wsl --manage Ubuntu --set-sparse true` recovers ~55 GB.
* Disk: **36 GB free** after session 7's cleanup.
