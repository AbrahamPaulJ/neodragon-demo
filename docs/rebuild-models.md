# Rebuilding every model from the weights

This is the complete, ordered recipe for regenerating all 14 context binaries the app
uses, starting from nothing but the public weights. It is the path to take when porting
to another SoC (set `HTP_ARCH` / `SOC_MODEL` first, see `docs/porting-other-socs.md`)
or when changing a graph.

No intermediate artefacts are published: no ONNX files, no calibration sets, and no
converter model libraries. Everything below regenerates them. Budget **one long day**:
about 8–10 h of mostly unattended compute on a 6 GB laptop GPU and a 16 GB RAM host, of
which MMDiT stage 2 alone is about 3 h.

## Before you start

1. **`SETUP.md` §1–4**: toolchain, the upstream clone **at the pinned commit**, and the
   weights (`py -3.10 work/pipeline/fetch_hybrid.py`).
2. **Read `docs/traps.md`**, or at least the triage table in `CLAUDE.md`. Almost every trap
   there is something that *runs and gives a wrong answer*.
3. **Disk.** Each MMDiT stage exports a 5.7 GB fp32 ONNX, and the converter stages a
   second copy onto the WSL ext4 disk. Keep **≥ 14 GB free per stage** and build stages
   one at a time (the `chain_*.ps1` scripts check this and delete intermediates as they
   go). Deleting files inside WSL does not return space to Windows. See the vhdx note in
   `HANDOFF.md`.
4. **WSL memory.** Stage 2's converter needs about 11 GB RAM + swap. Give WSL at least
   10 GB and a large swap file in `.wslconfig`.

Conventions below: `py` commands run on the **host** (Windows, from the repo root);
`wsl` commands run the converter under **WSL Ubuntu**. Launch WSL scripts from
PowerShell as `wsl -d Ubuntu -e bash work/qnn/<script>.sh ...`. Git Bash mangles the path
and exits 0 having done nothing. Every convert script writes
`work/device/<name>_<arch>.bin` plus `<name>_net.json`.

**After every conversion, run `py -3.10 work/device/analyze_net.py work/device/<name>_net.json`.**
It is static, it takes seconds, and it is what caught a 53.5× latency penalty (trap #11).

## 0. Reference latents (shared by two modules)

```
py -3.10 work/pipeline/run_reference.py --num-prompts 8     # -> work/calib/vae_dec/  (56 latents, ~5 min)
```

Both the streaming VAE decoder calibration and the QuickSRNet calibration are built from
these.

## 1. Text path

| step | command | output |
|---|---|---|
| DistilT5 export (residual scaling folded, trap #3) | `py -3.10 work/export/export_distilt5_folded.py` | `work/onnx/distilt5_folded_qnn.onnx` |
| DistilT5 convert (float → fp16 on HTP) | `wsl ... work/qnn/convert.sh distilt5_folded_qnn distilt5f` | `distilt5f` |
| ContextAdapter export + calibration | `py -3.10 work/export/export_context_adapter.py --prompts 300 --export` | `work/onnx/ctxadapt/`, `work/calib/ctxadapt/` |
| ContextAdapter convert (float) | `wsl ... env FLOAT=1 bash work/qnn/convert_ctxadapt_w8a16.sh` | `ctxadaptfp16` |
| CLIP L / G / L-with-projection export | `py -3.10 work/export/export_clip.py --export` | `work/onnx/clipl,clipg,cliplp/` |
| CLIP convert (float), one per graph | `wsl ... work/qnn/convert_clip_fp16.sh clipl` (then `clipg`, `cliplp`) | `clipl`, `clipg`, `cliplp` |

`clipl` / `clipg` feed the first-frame UNet. `cliplp` is the *video* path's pooled
embedding and comes from a **different checkpoint** (trap #37).

Environment variables do not pass through `wsl -e bash script` cleanly. Either put them
in the command with `env`, as shown above, or export them in `work/qnn/qnn_env.sh`.

## 2. First-frame path (SSD1B)

| step | command | output |
|---|---|---|
| UNet export | `py -3.10 work/export/export_ssd1b_unet.py --export` | `work/onnx/ssd1bunet/` |
| **UNet layout screen, before any GPU capture** | `wsl ... work/qnn/convert_ssd1bunet_w8a16.sh --layout`, then `analyze_net.py` | transpose/compute ratio |
| UNet calibration (125 prompts × 4 LCM steps = 500 calls) | `py -3.10 work/pipeline/capture_unet_calib.py --prompts 125 --split calib` | `work/calib/ssd1bunet/` |
| UNet convert W8A16 | `wsl ... work/qnn/convert_ssd1bunet_w8a16.sh` | `ssd1bunet` |
| SDXL VAE decoder export + calibration | `py -3.10 work/export/export_ssd1b_vaedec.py --prompts 128 --export` | `work/onnx/ssd1bvaedec/`, `work/calib/ssd1bvaedec/` |
| SDXL VAE decoder convert W8A16 | `wsl ... work/qnn/convert_ssd1bvaedec_w8a16.sh` | `ssd1bvaedec` |

## 3. Video VAE

| step | command | output |
|---|---|---|
| Encoder export (the exact T=1 collapse to 2-D) | `py -3.10 work/export/export_vae_encoder_2d.py` | `work/onnx/vaeenc2d_qnn.onnx` |
| Encoder calibration: real SSD1B first frames | `py -3.10 work/pipeline/capture_first_frames.py --split calib --num 64` | `work/calib/vae_enc/` (converter uses 50) |
| Encoder convert W8A16 | `wsl ... work/qnn/convert_vae_encoder_w8a16.sh vaeenc` | `vaeenc` |
| Streaming decoder export (explicit MemBlock state) | `py -3.10 work/export/export_vae_decoder_stream.py` | `work/onnx/vaedec_stream_qnn.onnx` |
| Decoder calibration with **real** states (trap #4) | `py -3.10 work/device/make_stream_calib.py` | `work/calib/vae_dec_stream/` (~3 GB) |
| **Permute state raws to NHWC** | `py -3.10 work/device/permute_states_nhwc.py` | sentinel `.layout_nhwc` |
| Decoder convert W8A16, NHWC states | `wsl ... work/qnn/convert_vae_stream_nhwc.sh vaedecsn` | `vaedecsn` |

Skipping the permute step does not raise an error. It calibrates the states in the wrong
layout.

## 4. QuickSRNet 2×

| step | command | output |
|---|---|---|
| Export (weights from `qai_hub_models`) | `py -3.10 work/export/export_quicksrnet.py --variant medium --scale 2` | `work/onnx/quicksrm2x/` |
| Calibration: decoded real frames, in [0,1] | `py -3.10 work/pipeline/make_quicksr_calib.py` | `work/calib/quicksr/` |
| Convert W8A16 | `wsl ... work/qnn/convert_quicksrnet.sh` | `quicksrm2x` |

## 5. Pyramidal MMDiT: three stages

Calibration is captured once, as **raw DiT calls**, and the per-stage graph inputs are
derived from those calls. This keeps the capture reusable if the graph rewrite changes.

```
py -3.10 work/pipeline/capture_mmdit_calib.py --num-prompts 50      # ~30 min GPU -> work/calib/mmdit/ (900 calls)
```

Then, **one stage at a time** (N = 0, 1, 2):

```
py -3.10 work/device/make_mmdit_io.py --stage N --split calib --out-name mmdit_sNfs
py -3.10 work/export/export_mmdit_stage.py --stage N --envelope --export --tag mmdit_sNfs_env   # ~15 min, 5.7 GB
wsl -d Ubuntu -e bash work/qnn/run_stage.sh layout N mmdit_sNfs     # optional ~2 min layout screen
wsl -d Ubuntu -e bash work/qnn/run_stage.sh full   N mmdit_sNfs     # 38 min / 49 min / ~3 h
```

After each stage finishes, delete `work/onnx/mmdit_sNfs_env/`, `work/calib/mmdit_sNfs/`
and `~/neodragon-build/onnx/mmdit_sNfs_env` inside WSL before starting the next.
`work/qnn/chain_s01fs.ps1` and `chain_s2fs.ps1` automate exactly this sequence,
including the disk guard and resuming after an interruption.

The fused score (`fuse_score=True`, trap #40) is the default in the export, and it is
what `fs` means in the names. `--split-score` rebuilds the older, 11–12 dB worse form
and exists only for A/B comparisons.

**Keep `~/neodragon-build/<name>/lib/x86_64-linux-clang/lib<name>.so`.** That file holds
the quantised model. With it, `work/qnn/retarget_ctx.sh` rebuilds for another chip in
minutes instead of repeating this section.

## 6. Canary and app assets

```
py -3.10 work/export/export_canary.py                # ONNX + canary_in/ref.raw into the app assets
wsl -d Ubuntu -e bash work/qnn/convert_canary.sh     # -> work/device/canary_<arch>.bin
copy work\device\canary_<arch>.bin work\android\app\src\main\assets\

py -3.10 work/export/export_app_assets.py            # add_embedding, CLIP vocab, LCM constants
py -3.10 work/export/export_video_structure.py       # time_text_embed + the 18 RoPE/layout tables
```

## 7. Verify before shipping

The numbers to reproduce, on v79, are in `HANDOFF.md`. For each module the device loop is
`make_*_io.py` (test split) → `push_*.ps1` → `run_*.sh` → `compare_*.py`, after staging
the harness with `work/device/push_harness.ps1 [-HtpArch vNN]`.

| module | v79 accuracy vs fp32 | compare script |
|---|---:|---|
| DistilT5 | 49.04 dB | `compare.py` |
| ContextAdapter | 39.36 dB | `compare_ctxadapt.py` |
| CLIP L / G | 60.75 / 14.71 dB (fp16) | `compare_clip.py` |
| SSD1B UNet | 32.51 dB | `compare_unet.py` |
| SSD1B VAE decoder | 32.43 dB | `compare_ssd1bvaedec.py` |
| VAE encoder | 41.60 dB | `compare_enc.py` |
| VAE decoder (streaming) | 34.27 dB | `compare_stream.py` |
| QuickSRNet 2× | 28.06 dB | `compare_quicksr.py` |
| MMDiT stage 0 / 1 / 2 | 39.67 / 37.80 / 33.89 dB | `compare_mmdit.py` |

Calibration is seed-reproducible, but a rebuild on a different SDK version or with a
different calibration count can move these numbers. A drop of more than ~1 dB is a sign
that something in the chain differs, not noise. Finally, upload the binaries, update the
byte sizes in `ModelStore.MANIFEST`, and build the app (`docs/porting-other-socs.md`
step 3).
