# Session 7 — the app: I2V, QuickSRNet, MP4, device gating

*2026-08-23. Companion to `notes/2026-08-23-android-e2e.md`, which covers the original
port. This one covers what session 7 added and the traps it cost.*

## The first frame was never 4.7x the paper — it was model loading

Instrumenting `FirstFrame.kt` to time LOAD separately from EXECUTE found two errors in our
own accounting, both now corrected in `docs/e2e-budget.md`:

1. **The documented 7.2 s was a COLD run.** Binaries are `mmap`'d, so run one page-faults
   ~2.9 GB in from storage. Measured cold-vs-warm on one build: 7468 then 2479/2204/2438 ms.
2. **Wall clock was compared against inference-only.** Table 7 measures per-module
   latency; our number included mapping 1.6 GB of CLIP context.

**Compute alone is ~1.90 s against the paper's 1.61 s — 1.18x.** Essentially at parity.

| first frame | before | after |
|---|---:|---:|
| first run (tap immediately) | 7468 ms | **~5100 ms** |
| warm | 4436 ms | **2132–2520 ms** |
| RSS | 0.40 GB documented | **238–268 MB** |

Fixes: `KEEP_CLIPG = true` (it cost **1897 ms to load, 40 ms to run** — the worst
load-to-compute ratio in the pipeline, released after every generation to reclaim memory
that is file-backed anyway); background preload of the whole first-frame path at app
start; and the video path no longer maps CLIP G **twice** (it loaded, released, then
`FirstFrame` reloaded it 30 lines later).

**Why tapping immediately still costs ~5 s, and why the Kotlin `@Synchronized` fix would
NOT help:** `ndqnn.cpp` has a global `std::mutex g_mu` held by `load`, `execute`, `init`
AND `free`. While the preload maps a 1.3 GB model the native layer blocks every execute
regardless of what Kotlin does. Fixing it means narrowing `g_mu` — per-model execute locks
and concurrent `contextCreateFromBinary` — a real concurrency change to NPU-driving native
code for ~1 s on the first tap only. Not done deliberately.

## Image-to-video — built

`Video.generate(prompt, seed, image, onProgress)`. With a photo supplied SSD1B is skipped
entirely, so `clipl` / `ssd1bunet` / `ssd1bvaedec` (1.68 GB) never map. Crop dialog with
pan + pinch (`CropDialog.kt`), photo picker via `PickVisualMedia` (no permission needed).

**Bug I introduced and fixed:** the release of the first-frame graphs sat in the `else`
branch, so in I2V mode `ssd1bunet` + `clipg` + friends stayed resident and the MMDiT then
mapped 4.5 GB on top — ~7.5 GB in one process, killed mid-loop with no tombstone and no
FATAL (the lmkd signature). The release is now unconditional before the MMDiT loop.

**Two Compose traps hit while building the crop UI**, both producing a wrong aspect ratio:
`graphicsLayer`-scaling a `ContentScale.Fit` image, then `Modifier.size()` — **which is
clamped by the parent's constraints**, so a zoomed photo got squashed back to the window.
The preview is now a `Canvas` `drawImage` with an explicit dst rect evaluating the SAME
expressions `crop()` inverts, so preview and result cannot diverge.

Also fixed: the picker decoded at full resolution (a 108 MP photo is ~400 MB as
ARGB_8888) — now downsamples on decode via `inSampleSize`, capped at 2048.

## QuickSRNet (Phase 2) — BUILT, untested on device

`work/device/quicksrm2x_v79.bin`, **377,880 bytes**. QuickSRNet **Medium, 2x**,
320x512 -> 640x1024. Large was ruled out by arithmetic (67.9 GMAC cannot be the paper's
6.5 ms); medium is 8.26 GMAC ~ 4 ms/frame ~ 200 ms per video.

* `export_quicksrnet.py` — pulls the checkpoint from AI Hub (`qai-hub-models`), asserts
  determinism, exports ONNX.
* `make_quicksr_calib.py` — 392 real decoded frames. The kept `vae_dec` latents are
  per-frame slices of 8 videos and had to be **re-stacked into [1,16,7,40,64]** before
  decoding: decoding them individually gives ONE frame each (causal decoder needs the
  temporal context), which is why the first attempt produced 56 frames instead of 392.
  They are already scale/shift-corrected — do NOT re-apply the factors.
* **Video 8 (49 frames) is held out**; calibration used 343. Scoring on calibration data
  would be optimistic and could not fail.
* Quantised graph: 10 nodes (the 7 Clips **fused into the convs**), zero float fallback,
  W8A16 `u16 x s8* x s32* -> u16`, transpose/compute **0.07**, no trap #27 issue.

**Expect BELOW the paper's 48 dB.** Table 9 gives QuickSRNet W8A16 + **AdaRound** and
credits AdaRound with "7+ dB SQNR"; we apply `--algorithms cle` only. A shortfall
identifies the AdaRound gap, not a broken conversion. `compare_quicksr.py` is ready.

**Calibration lists must use WSL paths** (`/mnt/c/...`). Windows paths make the converter
report every entry as a missing file and refuse to start.

## MP4 export — built, untested

`VideoWriter.kt`: H.264 via MediaCodec with an **input Surface** (the driver does
RGB->YUV; hand-packing YUV produces bugs that look like model artefacts). Lands in
**Movies/Neodragon** via MediaStore so it reaches the gallery. 24 fps regardless of the
preview toggle.

## HF — the three `*fs` builds are UP

Repo is now **15 binaries, 12.28 GiB**: both MMDiT generations coexist. All byte-verified
against local.

**Still to do:** delete `mmdit_s0g` / `s1f` / `s2f` from HF once the A/B confirms the new
set (back to ~8.3 GiB), upload `quicksrm2x`, and update `ModelStore.MANIFEST` +
`Video.STAGE_BIN` to the `*fs` names. **Until STAGE_BIN changes the app still runs the OLD
graphs** — every APK test so far used them.

## Method note, learned twice

**Verify background work by its OUTPUT, not by a process query.** `pgrep -f ab_stage`
matched its own command line and reported a dead run as RUNNING; the real evidence was a
0-byte log and a scratch dir untouched for three hours. `setsid` inside `adb shell` does
not reliably survive the connection either — run device work in the foreground over USB
and confirm log files appear.


## Shipping polish (end of session 7)

* **Device gating.** `DeviceCheck.kt` runs a 58 KB canary graph (`canary_v79.bin`, shipped
  in assets) before the download is offered. It proves the backend, skel and Hexagon
  revision agree, and checks fp16 correctness against a host fp32 reference -- a
  `Build.SOC_MODEL` allowlist would be a guess that goes stale with every chip. A
  confirmation dialog shows exactly what it found. Deliberately permissive: a check that
  cannot RUN returns Inconclusive and allows the download; only a real refusal blocks.
* **Download UX.** Progress notification (`DownloadNotifier`), immediate feedback on tap,
  4 MB repaint granularity. The old build set nothing until the first 16 MB landed, which
  read as a dead button.
* **Public repo.** `AbrahamPJ/neodragon-npu-s25u` is public, so the token field is gone.
  Licences checked at source: QAIRT grants distribution "solely in object code format and
  as incorporated in Your software application" (exactly the APK case) and contemplates
  app-store distribution; Neodragon is BSD-3-Clause-Clear. **QuickSRNet is the exception**
  -- the AIMET Model Zoo terms are non-exclusive, non-transferable and revocable with no
  patent grant, so redistributing that one derived binary deserves its own decision. The
  app runs at 320x512 without it.
* **Icon and title.** Adaptive icon (film frame + play triangle in the app accent),
  label "Neodragon Demo". Was a stock `ic_menu_gallery`.
* **MP4 fps.** Was 62 fps for a 24 fps request -- trap #47.
* **Layout.** The main column now scrolls; with the models panel open it silently
  overlapped the generate buttons.

## Device and disk left clean

The phone was wiped at the end of the session: `/data/local/tmp/nd` (9.3 GB desktop
harness) and the app's `ctx/` (12 GB of models) both removed, **21 GB reclaimed**. The app
now downloads its own models, so nothing is pushed by hand. `work/device/push_harness.ps1`
restores the harness if `qnn-net-run` is needed again.

Host: **36 GB free** after removing the superseded MMDiT generation (on HF for rollback),
an old 1.5 GB android lib build, QuickSRNet test I/O, and stale net.json files.


## The download had to become a foreground service

It ran in the Activity's `rememberCoroutineScope()`. **That scope is tied to the
COMPOSITION**, so backgrounding or recreating the Activity cancelled the transfer
mid-file. It looked like "it errors, and I can restart it when I'm in the app" -- the
restart worked because `Range` resume picked up the partial file.

Android also throttles network for processes that are not foreground, so surviving
cancellation would not have been sufficient on its own.

`DownloadService` is a foreground service with an ongoing notification. Progress is
published through a process-wide `State` object rather than by binding, because the UI may
not exist while the download runs and has to be able to **attach to a transfer already in
flight** when it comes back.

> **Trap: never run a multi-minute job in `rememberCoroutineScope()`.** Anything that must
> outlive the screen needs a foreground service (or WorkManager). The symptom is an
> operation that fails only when the user looks away.
