# Session 5 — getting the MMDiT layerwise dump

Continues `notes/2026-08-22-HANDOFF.md`, whose single instruction was *"get the
layerwise dump; everything else is eliminated."*

## The blocker is gone

The handoff listed the Android NDK as an uninstalled prerequisite. Installed
`android-ndk-r26d` (Linux) into WSL at `$HOME/ndk/android-ndk-r26d` — 637 MB download,
and the *Linux* NDK is what matters: the Windows NDK already present under
`~/Downloads` and the Android SDK is useless here because `qnn-model-lib-generator`
runs under WSL and shells out to `ndk-build`.

| step | wall | result |
|---|---|---|
| NDK download + unzip | 1:20 | `$HOME/ndk/android-ndk-r26d` |
| `qnn-model-lib-generator -t aarch64-android` | **5:52** | `libmmdit_s0.so`, 1,519,807,184 B |
| `adb push` over USB | ~50 s | verified by byte length |
| **online graph prep + 16-case run on device** | **65 s total** | 16 results |

`work/qnn/build_android_lib.sh` and `work/device/push_mmdit_lib.ps1` /
`run_mmdit_lib.sh` do all of it.

**Online prep of a 1.5 GB graph was not slow.** The handoff warned it might be; the
whole run — compose, finalize, 16 inferences — took 65 seconds. That removes the last
reason to prefer the context binary for diagnostics.

### One trap worth recording

`qnn-model-lib-generator` builds in a temp directory under **the current working
directory**, and it explodes the 1.5 GB weight blob into ~900 individual `.raw` files,
`llvm-objcopy`s each into a `.o`, then links. Run it with cwd on `/mnt/c` and every one
of those files crosses the 9p mount. It still finished in 5:52, but the peak on-disk
footprint was **4.2 GB of scratch on C:** — `cd` to ext4 first, or at least know where
the space went.

## The control: is an online-prepared graph the same graph?

`--debug` and `--set_output_tensors` only work against `--model`, never against a
retrieved context binary. That makes the whole dump worthless unless the online-prepared
graph computes what the shipping artefact computes. It does:

| build | deploy SNR (16 held-out cases) |
|---|---|
| context binary `mmdit_s0_v79.bin` (`--retrieve_context`) | 19.48 dB |
| **aarch64 model lib, online prep, same htp config** | **19.48 dB** |

Case for case, not just in the mean (14.51 / 19.45 / 19.98 / … / 20.00; min 12.16).
So anything the dump shows is a property of the shipping graph.

The htp config is passed explicitly on the online path
(`vtcm_mb: 8`, `O: 3.0`, `dsp_arch v79`, `soc_model 69`, burst) so that the comparison
isolates *quantisation* rather than confounding it with graph-prep defaults.

## The reference: onnxruntime, not torch

`block_snr.py` uses a torch reference and 35 hand-derived boundary tensors. For a full
layerwise diff that is the wrong tool, because the join has to be by name and torch has
no names. The converter's tensor names are the **ONNX names with every non-identifier
character replaced by `_`**:

```
/norm1/norm.12/Constant_1_output_0   ->   _norm1_norm_12_Constant_1_output_0
/Add_10_output_0                     ->   _Add_10_output_0
```

so device raws join to ONNX intermediates mechanically, for all 2374 dumpable tensors.
`work/device/layer_snr.py` does this: load the ONNX proto with
`load_external_data=False` (2.3 MB, instant), append the wanted intermediates to
`graph.output`, re-save **into the same directory** so the external-data path still
resolves, and run onnxruntime.

Measured: proto load 0.002 s, session build 11 s, **inference 1.5 s**, and the fp32
`noise_pred` reproduces the stored `nref` at **121.73 dB**. That is a far cheaper
reference than the 1.5 B-parameter torch model, and it is the *same graph that was
converted*, so an export discrepancy cannot hide in it.

`--debug` for stage 0 is **4.33 GB / 2374 tensors** (computed from `net.json`; the 36
`[24,408,408]` softmax outputs are 16 MB each).

## `--debug` does not work on this graph — `--set_output_tensors` does

| forced client-readable tensors | HTP finalize |
|---|---|
| 2374 (`--debug`) | ❌ `Finalize Graph for Idx = 0 failed with error = 1002` after 1:55 |
| 314 (one block) | see below |
| 35 (block boundaries) | ✅ finalize + execute in 71 s, 42 MB |

So the dump has to be taken a slice at a time. `work/device/block_tensors.py` emits the
names for one block; `run_mmdit_lib.sh 0 names <file>` runs an arbitrary list.

## THE RESULT: the error is there after ONE block

Per-block SNR, device vs onnxruntime fp32, case 0 (whose end-to-end SNR is 14.51 dB):

| block | image stream | text stream |
|---:|---:|---:|
| 0 | **23.49** | 19.31 |
| 1 | 24.39 | 16.88 |
| 2 | 23.99 | **2.93** |
| 3 | 21.73 | 5.06 |
| 4 | 22.95 | 7.31 |
| 5 | 22.41 | 8.31 |
| 6 | 21.94 | 9.54 |
| 7 | 22.05 | 10.86 |
| 8 | 21.54 | 9.75 |
| 9 | 20.74 | 11.62 |
| 10 | 19.69 | 10.35 |
| 11 | 17.71 | 10.24 |
| 12 | 18.55 | 11.14 |
| 13 | 16.49 | 13.98 |
| 14 | 12.64 | 13.98 |
| 15 | 11.30 | 13.71 |
| 16 | 11.32 | — |
| 17 | 27.63 | (no text output: `context_pre_only`) |
| **noise_pred** | **14.51** | |

Read against `block_snr.py`'s own decision rule, this is the third case:
**"SNR is already bad after block 0 → the problem is in patchify / conditioning, not
the blocks at all."**

Three things follow, and they change the shape of the problem:

1. **There is no smooth accumulation.** The image stream loses everything it is going to
   lose in block 0 (23.49 dB) and then sits between 21 and 24 dB for six more blocks.
   Trap #3's residual-accumulation argument is dead a second time, now from the
   intermediate values rather than from a host replay.
2. **The text stream collapses at block 2** — 16.88 → **2.93 dB** in one block — and
   then *climbs* monotonically to ~14 dB. A rising SNR down a residual chain does not
   mean the error is being repaired: the text residual grows 197 → 27,922 (140×), so a
   roughly fixed absolute error shrinks in relative terms as the signal grows around it.
   The absolute damage is injected once, in block 2, and never removed.
3. **The last block's +16.3 dB jump** (11.32 → 27.63) is the same effect: `_Add_214` has
   `max|ref|` 1.78e4 against the previous block's 7.3e3.

### Clipping: real, measured, and too small to matter

Checked first because it was the one thing the host ablations structurally could not
see — `quant_ablate.py` derives each range from the tensor it is quantising, so it can
never saturate. `work/device/clip_scan.py` compares every activation's true fp32 range
on the held-out cases against the interval `net.json` actually calibrated:

```
tensors whose real range LEAVES the calibrated encoding: 443 of 2123
total clipped elements across 3 cases:                   184,944
worst overflow:                                          0.299 x span
```

and the worst offenders are all in blocks 9–17 (`_ff_context_net_0_11_...`,
`_norm_add_k_13_...`), with a handful of elements each — nothing in block 0. Saturation
is real but second-order, and it cannot explain a block-0 loss. **Hypothesis 7, dead.**
