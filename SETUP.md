# Setup — restoring everything the repo deliberately does not carry

This repo holds the **port**: the export scripts, the conversion recipes, the device
tooling, the Android app, and the write-ups. It does not hold model weights, converted
binaries, vendor SDK material, or the upstream clone — about 26 GB in total, none of which
is ours to redistribute or useful to version.

Restore in this order.

## 1. Toolchain

* **QAIRT SDK 2.49.0.260730** (Qualcomm AI Engine Direct). Everything here targets
  **HTP v79 / SM8750** (`soc_model: 69`). Conversion runs under **WSL**
  (`x86_64-linux-clang`); adb runs from **PowerShell** — Git Bash mangles `/data/...`
  paths.
* **Python 3.10** on Windows with torch 2.8+cu129, onnx, onnxruntime, transformers 4.55.3,
  diffusers, timm, sentencepiece, onnxconverter_common.
* A **QNN venv under WSL** on Python 3.10 with numpy 1.26.4 (the SDK's tested pin) — the
  SDK's `libPyIrGraph310` requires 3.10.
* **Android NDK r26d or newer** (Linux, inside WSL) for `qnn-model-lib-generator`, plus the
  Android SDK with **NDK 27** and **CMake 3.22.1** for the app.
* **adb** on `PATH`, or set `$env:ADB` to its full path. The PowerShell device scripts use it.

Then point the scripts at your toolchain:

```bash
cp work/qnn/qnn_env.example.sh work/qnn/qnn_env.sh   # gitignored; edit QNN_SDK_ROOT, VENV
QNN_SDK_ROOT=/mnt/c/path/to/qairt/2.49.0.260730 bash work/qnn/setup_env.sh   # builds the WSL venv
```

Every conversion script sources `work/qnn/qnn_env.sh` (or `$QNN_ENV`) and finds the repo
root from its own location, so nothing depends on where you cloned it. On Windows, also
set `QNN_SDK_ROOT` for the PowerShell scripts.

> **Targeting a chip other than the SM8750?** Set `HTP_ARCH` and `SOC_MODEL` before
> converting. See [`docs/porting-other-socs.md`](docs/porting-other-socs.md).

## 2. Vendor libraries the app needs

The app links nothing at build time — it `dlopen`s the backend at runtime — but the `.so`
files must be packaged, and the C++ needs the SDK headers:

```bash
QAIRT=/path/to/qairt/2.49.0.260730
A=work/android/app/src/main/

# headers
mkdir -p $A/cpp/include && cp -r $QAIRT/include/QNN $A/cpp/include/

# runtime libraries (arm64)
mkdir -p $A/jniLibs/arm64-v8a
cp $QAIRT/lib/aarch64-android/libQnnHtp.so \
   $QAIRT/lib/aarch64-android/libQnnHtpNetRunExtensions.so \
   $QAIRT/lib/aarch64-android/libQnnHtpPrepare.so \
   $QAIRT/lib/aarch64-android/libQnnHtpV79.so \
   $QAIRT/lib/aarch64-android/libQnnHtpV79Stub.so \
   $QAIRT/lib/aarch64-android/libQnnSystem.so \
   $A/jniLibs/arm64-v8a/
cp $QAIRT/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so $A/jniLibs/arm64-v8a/
```

The skel is found at runtime through `ADSP_LIBRARY_PATH`, which the app sets to its own
`nativeLibraryDir` before the first `dlopen` (see `ndqnn.cpp`).

> Two things here are not optional and cost a day each to rediscover — see traps #33 and
> #34 in `docs/traps.md`. An app **cannot exec a helper binary onto the DSP**, and
> `libcdsprpc.so` must be declared with `<uses-native-library>` in the manifest.

## 3. Upstream model code

```bash
git clone https://github.com/qualcomm-ai-research/neodragon src/neodragon
```

BSD-3-Clause-Clear. The port reads it for the reference implementation and imports it
directly in the export scripts.

## 4. Weights

`Qualcomm-AI-Research/Neodragon` on HuggingFace (BSD-3-Clause-Clear, 17.5 GB total). The
AR text-to-video path needs only ~8.6 GB of it, into `work/models/neodragon/`:

```
text_encoder_3 + tokenizer_3   DistilT5, 259 MB
diffusion_transformer_320p     3.1 GB
context_adapter                520 MB
causal_video_vae               239 MB
ssd_1b_unet                    2.6 GB
ssd_1b_vae (fp16)              167 MB
ssd_1b_text_encoder + _2       1.6 GB
text_encoder + text_encoder_2  the video path's pooled embeddings — NOT the ssd_1b copies
tokenizer, tokenizer_2, tokenizer_3
```

The `_multistep_t2v` variants (3.7 GB) are a separate path and are not needed.

> `text_encoder` and `ssd_1b_text_encoder` are **different checkpoints for different
> purposes** — trap #37. The video path needs the projection head; the first-frame path
> does not.

## 5. Build the app assets

Two scripts turn the weights into the ~57 MB of host-side tables the APK carries:

```bash
py -3.10 work/export/export_app_assets.py        # add_embedding, CLIP vocab, LCM constants
py -3.10 work/export/export_video_structure.py   # time_text_embed + all 18 RoPE/layout tables
```

## 6. Convert the models

Each module has an export script in `work/export/` and a conversion script in `work/qnn/`.
Run conversions under WSL. Rough wall times measured on this machine:

| module | conversion |
|---|---|
| MMDiT stage 0 / 1 / 2 | 38 min / 49 min / **2 h 52 min** |
| SSD1B UNet, CLIP G | tens of minutes |
| VAE encoder/decoder, DistilT5, ContextAdapter | minutes |

Stage 2 is superlinear in sequence length — a single `[24,1728,1728]` score matrix is
143 MB in fp32, so the quantiser's min/max pass becomes memory-bandwidth bound.

**Run `work/device/analyze_net.py` on every `*_net.json` before spending device time.** It
is free, static, and it is what caught a 53.5× latency penalty.

## 7. Build and install the app

```bash
echo "sdk.dir=/path/to/Android/Sdk" > work/android/local.properties
cd work/android && ./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

The app downloads its context binaries itself from the URL in its settings. The default is
the v79 set on Hugging Face; override it at build time with `-PmodelBaseUrl` (see
`work/android/app/build.gradle.kts`). To side-load binaries you built yourself instead:

```
adb push work/device/<name>_v79.bin \
  /sdcard/Android/data/com.neodragon.demo/files/nd/ctx/
```

**Always verify a push by byte length.** Large transfers to this device silently truncated
twice (1.5 GB arrived as 113 MB, 1.4 GB as 577 MB, both reporting success). The app checks
for you — it compares each file against `ModelStore.MANIFEST` and reports what is missing.
