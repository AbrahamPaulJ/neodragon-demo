# Porting to other Snapdragon SoCs (8 Gen 3, 8 Gen 2, and older)

Everything in this repo was built and measured on **one** device: a Samsung S25 Ultra,
SM8750, Hexagon **HTP v79**. Nothing has been run on any other chip. This page covers what
is tied to v79, what you have to change, and which risks are likely to show up first on
older silicon. Anything marked *unverified* is a prediction, not a measurement.

## Why the shipped binaries will not load on your phone

A QNN **context binary** is graph preparation done ahead of time for one Hexagon revision
(`dsp_arch`) and one SoC (`soc_model`). A v79 binary does not deserialize on v75 or older.
The app finds this out with a 58 KB canary graph before it offers the 8 GB download
(`DeviceCheck.kt`), so on an 8 Gen 3 you will see the "would not load" message. That is
expected, not a bug.

The **quantisation is not SoC-specific.** The slow part of every conversion
(`qnn-onnx-converter` with calibration: up to 3 h for MMDiT stage 2, plus GPU time to
capture calibration data) produces a model library that works for any target. Only the
last step, `qnn-context-binary-generator`, has to be re-run per chip.

## Target table

`SOC_MODEL` comes from the `QNN_SOC_MODEL_*` enum in
`$QNN_SDK_ROOT/include/QNN/QnnTypes.h`. These values are from the QAIRT 2.49 header:

| Chip | SoC | `HTP_ARCH` | `SOC_MODEL` |
|---|---|---|---:|
| Snapdragon 8 Elite (reference) | SM8750 | `v79` | 69 |
| Snapdragon 8 Gen 3 | SM8650 | `v75` | 57 |
| Snapdragon 8 Gen 2 | SM8550 | `v73` | 43 |
| Snapdragon 8+ Gen 1 | SM8475 | `v69` | 42 |
| Snapdragon 8 Gen 1 | SM8450 | `v69` | 36 |

For any other chip, look up its SoC enum in `QnnTypes.h` and its Hexagon revision in
Qualcomm's QNN/QAIRT SoC support table. Also check that
`$QNN_SDK_ROOT/lib/hexagon-<arch>/unsigned/` exists, which tells you the SDK still ships
that revision.

## Step 1: context binaries

Every `work/qnn/convert_*.sh` that writes an HTP config reads `HTP_ARCH` and `SOC_MODEL`
(default `v79` / `69`) and names its output `<name>_<arch>.bin`.

**If you have the model libraries** (`$HOME/neodragon-build/<name>/lib/x86_64-linux-clang/lib<name>.so`,
left behind by any earlier conversion), re-target them in minutes:

```bash
export HTP_ARCH=v75 SOC_MODEL=57
for n in mmdit_s0fs mmdit_s1fs mmdit_s2fs quicksrm2x ctxadaptfp16 distilt5f \
         vaeenc vaedecsn clipl cliplp clipg ssd1bunet ssd1bvaedec canary; do
  wsl -d Ubuntu -e bash work/qnn/retarget_ctx.sh $n
done
```

**If you are starting from nothing,** follow `SETUP.md` (weights, exports, calibration
capture) and run each module's conversion script with the variables set:

| shipped binary | produced by |
|---|---|
| `mmdit_s0fs` / `s1fs` / `s2fs` | `export_mmdit_stage.py`, `capture_mmdit_calib.py`, `make_mmdit_io.py`, then `run_stage.sh full <stage> <name>` (see `chain_s01fs.ps1`, `chain_s2fs.ps1`) |
| `vaedecsn` | `export_vae_decoder_stream.py`, `make_stream_calib.py`, `convert_vae_stream_nhwc.sh vaedecsn` |
| `vaeenc` | `export_vae_encoder_2d.py`, `capture_first_frames.py`, `convert_vae_encoder_w8a16.sh` |
| `distilt5f` | `export_distilt5_folded.py`, `convert.sh distilt5_folded_qnn distilt5f` |
| `ctxadaptfp16` | `export_context_adapter.py --export`, `FLOAT=1 convert_ctxadapt_w8a16.sh` |
| `clipl` / `cliplp` / `clipg` | `export_clip.py`, `convert_clip_fp16.sh <name>` |
| `ssd1bunet` | `export_ssd1b_unet.py`, `capture_unet_calib.py`, `convert_ssd1bunet_w8a16.sh` |
| `ssd1bvaedec` | `export_ssd1b_vaedec.py`, `convert_ssd1bvaedec_w8a16.sh` |
| `quicksrm2x` | `export_quicksrnet.py`, `make_quicksr_calib.py`, `convert_quicksrnet.sh` |
| `canary` | `export_canary.py`, `convert_canary.sh` |

Read each script's header before running it. Several have a `--layout` mode that screens
the graph in about 2 minutes without calibration, and `docs/traps.md` explains why
that screen matters.

## Step 2: runtime libraries

`libQnnHtp.so` picks the stub that matches the device at runtime, so the APK has to
carry the stub and skel **for your revision**. Packaging several revisions side by side
is fine. For v75:

```bash
A=work/android/app/src/main/jniLibs/arm64-v8a
cp $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV75Stub.so  $A/
cp $QNN_SDK_ROOT/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so $A/
```

The skel comes from `lib/hexagon-<arch>/unsigned/`, **not** `lib/aarch64-android/`.
Taking it from the wrong folder shows up as `Failed to load skel, error: 4000`.

For the desktop harness: `.\work\device\push_harness.ps1 -HtpArch v75`.

## Step 3: the app

1. Copy `work/device/canary_<arch>.bin` into `work/android/app/src/main/assets/`.
2. Upload your `<name>_<arch>.bin` files to a Hugging Face model repo (or any HTTPS host
   that supports `Range` requests).
3. Update the byte sizes in `ModelStore.MANIFEST`. Graph preparation differs per revision,
   so the sizes will not match v79. The download check compares exact byte counts.
4. Build:

```bash
./gradlew assembleDebug -PhtpArch=v75 \
    -PsocLabel="Snapdragon 8 Gen 3 / SM8650" \
    -PmodelBaseUrl=https://huggingface.co/<you>/<repo>/resolve/main
```

These properties set `BuildConfig.HTP_ARCH`, `SOC_LABEL` and `MODEL_BASE_URL`. The
context file names, the canary asset name and the UI text all read from them.

## Step 4: measure before you trust it

Generation is seed-reproducible, so single-run A/B comparisons are valid. The device loop
per module is `make_*_io.py` → `push_*.ps1` → `run_*.sh` → `compare_*.py` (see
`CLAUDE.md`). Re-check at least:

* **Accuracy per module** against the v79 numbers in `HANDOFF.md`. The quantisation
  encodings are identical, so a large drop points at graph preparation or an op falling
  back, not at calibration.
* **`analyze_net.py` on the new `*_net.json`**, and a `--profiling_level detailed` run
  through `prof_by_optype.py`. Layout decisions happen before the quantiser and should
  not change, but kernel selection does.
* **Latency with trap #10 in mind:** interleave A/B runs in one session and take the
  minimum of N runs, never the average.

## What is likely to break first on older chips (*unverified*)

Ranked by how likely each one is to cost you a day:

1. **The 1.5 GB MMDiT stage binaries.** Older Hexagon revisions have less DSP-side address
   space and less VTCM, and large-model deployments on v73 and older are usually split
   into smaller graphs. If a stage fails at `contextCreateFromBinary` or at the first
   execute with a memory error, split each stage into two graphs at a transformer block
   boundary. `block_tensors.py` / `find_block_io.py` already list the block I/O names. The
   three stages are also co-resident in the app (`QnnRunner.kt`), about 4.5 GB mapped at
   once, so try releasing between stages before you split anything.
2. **The float graphs** (`distilt5f`, `ctxadaptfp16`, `clipl`, `cliplp`, `clipg`). HTP
   runs them in fp16 with no fp32 upcast (trap #3). The canary's fp16 check exists for
   exactly this case. If it reports `Fp16Suspect`, switch those modules to W8A16
   (`WBITS=16 convert_ctxadapt_w8a16.sh` builds `ctxadaptw16`, `QUANT=1 convert_clip_fp16.sh <name>`
   builds a W8A16 CLIP, and `convert_distilt5_scaled.sh` is the residual-scaled DistilT5 path).
3. **Latency.** MMDiT stage 2 is memory-bandwidth bound (trap #41,
   `docs/phase5-stage2-bandwidth.md`), so expect it to scale with LPDDR generation more
   than with NPU TOPS. It is already 47% of the video on v79.
4. **RAM.** The app's peak RSS is 238–268 MB because binaries are `mmap`ed (trap #35), but
   the DSP-side sessions still need physical memory. An 8 GB phone is the most likely
   place to see low-memory kills.
5. **16-bit activation kernels.** Every quantised graph is W8A16. It is supported well
   back, but it gets less kernel coverage on older revisions. `matmul_bits.py` and
   `prof_by_optype.py` will show whether an op fell back.

If you get this running on another chip, please add a row to the table above and a note
under `notes/` with the numbers. That is the part of this document nobody has measured yet.
