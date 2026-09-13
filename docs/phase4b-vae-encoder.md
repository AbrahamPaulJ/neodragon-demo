# Phase 4b — the VAE encoder

*Started 2026-08-21, the session after Phase 4 closed.*

The encoder was never in the phase plan. The 2026-08-21 session found it on the critical
path at the very end and left this warning:

> It is the hard half of the "asymmetric" VAE: `modeling_causal_ops.py`'s `CausalConv3d`
> carries a stateful `deque` feature cache and is **not exportable as written**.
> Paper Table 7 puts it at **1206.5 ms** — the most expensive component in the pipeline.
> **Do not quote an E2E timeline without scoping this first.**

Scoped. **The `deque` is not on the critical path, and the graph is far cheaper than the
paper's number implies.** Two findings, both static, both verified before any device time.

## Finding 1 — the feature cache is never used on the E2E path

`CausalConv3d.forward` has two branches:

```python
if not temporal_chunk:
    x = F.pad(x, self.time_causal_padding, mode=pad_mode)     # <-- stateless
else:
    ...
    self.cache_front_feat.append(x[:, :, -2:].clone().detach())   # <-- the deque
```

The `deque` lives only on the `temporal_chunk=True` branch, which is reached from
`AsymmetricCausalVideoVAE.temporal_chunk_encode`. The AR t2v path never calls it:

```python
# generation_utils.py:526-531
image = image.resize((width, height), resample=Image.LANCZOS)
image = _pil_to_numpy(image)
image = rearrange(image, "(b t) h w c -> b c t h w", b=1, t=1)   # <-- t = 1
image_latent = vae.encode(image).latent_dist.sample()            # <-- all defaults
```

`encode()`'s defaults are `is_init_image=True, temporal_chunk=False`, and `use_tiling` is
`False` unless something calls `enable_tiling()` (nothing in the pipeline does). So the
encoder runs its stateless branch, once, on a **single 320×512 frame**.

The `deque` remains genuinely unexportable. It is simply not something Phase 4b has to
solve. If a future phase ever needs full-video encoding, the explicit-state pattern from
`export_vae_decoder_stream.py` is the template — but nothing on the E2E path needs it.

## Finding 2 — at T=1 the whole encoder collapses to a 2-D convnet, exactly

Every `CausalConv3d` front-pads time by `time_kernel_size - 1 == 2` with **constant zeros**
(`pad_mode` is `"constant"`, and the `self.time_pad < x.shape[2]` guard forces `"constant"`
at T=1 regardless), giving `T_pad == 3`, then convolves with a time-3 kernel and no time
padding. One output frame, and:

```
out[0] = w[0]·0 + w[1]·0 + w[2]·x_real  ==  w[2]·x_real
```

The two discarded terms are multiplications by zero, so this is **exact, not approximate**.
It holds for both temporal strides, because with `T_pad == 3` and `k == 3` the stride-2 case
also produces exactly one output frame covering the same three planes:

| upstream module | stride | at T=1 becomes |
|---|---|---|
| `conv_in`, resnet `conv1`/`conv2`, `conv_out` | (1,1,1) | `Conv2d(k=3, stride=1, pad=1)` on `w[:, :, -1]` |
| `CausalDownsample2x` | (1,2,2) | `Conv2d(k=3, stride=2, pad=1)` on `w[:, :, -1]` |
| `CausalTemporalDownsample2x` | (2,1,1) | `Conv2d(k=3, stride=1, pad=1)` on `w[:, :, -1]` |
| `conv_shortcut`, `quant_conv` | k=1 | `Conv2d(1×1)`, `time_pad == 0`, pointwise |

`CausalGroupNorm` does `rearrange(x, "b c t h w -> (b t) c h w")` — **trap #11's exact
shape**, a time-into-batch fold. At T=1 it is a squeeze, so it is replaced with a plain 2-D
`GroupNorm` rather than left for the converter to notice. Same for the mid block's
attention rearranges, which only existed to fold `t` into batch before calling a 2-D
`Attention`.

**Verified host-side** (`work/export/export_vae_encoder_2d.py`), two input distributions,
against the stock `vae.encoder` + `vae.quant_conv`:

```
[collapse] 30 Conv3d in the encoder -> 31 Conv2d (incl. quant_conv), 0 five-D ops
[check] trial 0: (1, 32, 40, 64)  max|diff|=4.578e-05  SNR=129.6 dB
[check] trial 1: (1, 32, 40, 64)  max|diff|=4.578e-05  SNR=129.6 dB
[check] PASS -- the T=1 collapse is exact
```

129.6 dB is fp32 accumulation-order noise, not approximation error.

### What that costs, and what the paper's 1206.5 ms probably is

`work/device/enc_flops.py`:

| | MACs |
|---|---|
| collapsed 2-D encoder (conv 353.94 + attn 6.71 + gemm 2.68) | **363.33 GMAC** |
| naive T=1 5-D export (3× the conv work, two planes of zeros) | 1071.2 GMAC |
| Phase 4 decoder, for scale — ran in 100.18 ms | 205.07 GMAC |

The decoder sustained ~2.05 TMAC/s on this chip. At that rate 363 GMAC is ~180 ms.
The paper's 1206.5 ms divided by the **uncollapsed** 1071.2 GMAC is ~1.13 ms/GMAC — the
same order as an unoptimised 3-D build. So the reference deployment most likely convolved
the zero planes.

## Calibration and the held-out set

Trap #4: calibration must use real pipeline tensors. The encoder's input depends only on
SSD1B and the resize — the DiT is irrelevant to it — so
`work/pipeline/capture_first_frames.py` reproduces exactly the chain at
`generation_utils.py:526-530` and skips the 35 s/video AR loop, at ~2 s/image:

```
SSD1B(prompt + DEFAULT_PROMPT_MODIFIER) -> 1024x640 PIL
  -> resize((512, 320), LANCZOS) -> /255*2-1 -> [1, 3, 320, 512] in [-1, 1]
```

Two **disjoint** prompt sources so the deploy number is honest:

| split | prompts | count | destination |
|---|---|---|---|
| calib | `vbench_prompts.txt` | 64 (50 used, Table 9's count) | `work/calib/vae_enc/` |
| test | `showcase_prompts.txt` | 16, never seen by the quantiser | `work/device/eio/` |

The input range is exactly [-1, 1] by construction, which is the best case for the input
encoding.

## The measurement trap in this module

`moments` is `[mean | logvar]` concatenated on the channel axis. Measured over the 16
held-out frames:

- `mean` half: std ≈ 1.85, range ≈ [-8.3, +7.1]
- `logvar` half: range ≈ [-60, -12] — i.e. `std = exp(0.5·logvar)` ≈ 3e-7 at most

So **whole-tensor SNR is dominated by the logvar half**, which is a near-constant that any
quantiser reproduces trivially. It would flatter the build. What actually reaches the DiT is

```python
image_latent = mean + exp(0.5*logvar)*noise      # the noise term is ~1e-7, negligible
image_latent = (image_latent - VAE_SHIFT_FACTOR) * VAE_SCALE_FACTOR
```

`work/device/compare_enc.py` therefore reports **mean-half SNR as the headline**, with the
latent-space and whole-tensor figures alongside for transparency.

For the same reason the mixed output range is *not* a quantisation problem: a single A16
encoding over a span of ~68 gives a step of 1.04e-3, i.e. ~76 dB SQNR against a signal of
std 1.85. The output encoding is not the limiter, so the graph keeps one `moments` output
rather than splitting `quant_conv` into two branches.

## Conversion and the static gate

`work/qnn/convert_vae_encoder_w8a16.sh vaeenc`, W8A16 PTQ, per-channel weights, 50
calibration samples, `--preserve_io layout image moments`. Converter reports **361.03 GMAC,
37.36 M params**; context binary 41.67 MB.

Then trap #12, before any device time — `analyze_net.py work/device/vaeenc_net.json`:

```
op histogram: Eltwise_Binary 34, Conv2d 31, GroupNorm 22, ElementWiseNeuron 21,
              Transpose 10, Reshape 10, FullyConnected 4, Convert 2, MatMul 2, Softmax 1
float tensors (unquantised fallback): 0

Transpose output bytes :      22.12 MB
compute  output bytes  :     451.22 MB
ratio transpose/compute:       0.05
```

**0.05 — cleaner than the decoder's shipping build (0.09).** Two things worth recording:

1. **The converter fuses GroupNorm natively.** PyTorch's ONNX exporter never emits opset-18
   `GroupNormalization` — at opsets 17, 18 and 20 alike it lowers `nn.GroupNorm` to
   Reshape → InstanceNormalization → Reshape → Mul → Add, and that first reshape folds
   channels into a flattened axis, which is trap #11's shape. It does not matter here:
   `op_graph_optimizations.py`'s `OptimizeGroupNormTranslation` matches that exact pattern
   and rewrites it to a single `GroupNorm`, which `htp.json` lists as natively supported.
   All 22 arrived as `GroupNorm`. **Phase 3's SSD1B UNet is full of GroupNorms and can rely
   on this** — no pre-emptive fusion pass needed.
2. **The `vaeencn` NHWC-image variant is not worth building.** The whole point of the
   `vaedecs → vaedecsn` A/B was that boundary transposes were 63.00 MB of 63.00 MB. Here
   `image_nhwc` is 0.98 MB and `moments_nchw` is 0.16 MB out of 22.12 MB — the remaining
   20.96 MB is all inside the mid-block attention. Pinning the image channel-last would move
   ~0.2% of total traffic. The variant stays in the convert script, unbuilt, and this is the
   measurement that says why.

## On device — S25 Ultra, HTP V79

Accuracy on the 16 held-out showcase frames the quantiser never saw
(`work/device/compare_enc.py`):

```
  mean-half SNR:  mean  41.60 dB   min  39.78 dB
  latent    SNR:  mean  41.61 dB   min  39.68 dB
  whole     SNR:  mean  46.24 dB   (inflated by the logvar half)
  nan+inf: 0
  paper deploy SNR target for VAE Enc: 40.0 dB  -> MET
```

The predicted metric trap showed up exactly as expected: the whole-tensor figure reads
4.6 dB better than the number that matters. Mean pixel-space error on the latent is
~0.08 in a signal of std 1.85.

Latency, min-of-N across 3 interleaved rounds in one thermal session (trap #10):

| round | min | average |
|---|---|---|
| Q1 | **166.83 ms** | 177.90 ms |
| Q2 | 179.92 ms | 194.26 ms |
| Q3 | 166.87 ms | 201.31 ms |

| | this build | paper (8 Elite Gen4) |
|---|---|---|
| deploy SNR | **41.60 dB** | 40 dB |
| latency | **166.83 ms** | 1206.5 ms |

**7.2× faster than the reference figure, and +1.60 dB, on the first conversion.**

The throughput cross-check says the collapse is the whole story: 361.03 GMAC in 166.83 ms is
**2.16 TMAC/s**, against the decoder's 205.07 GMAC in 100.18 ms = 2.05 TMAC/s. Same chip,
same efficiency. A naive 1071.2 GMAC build at that rate would be ~496 ms before counting any
5-D layout traffic, which is the right order for the paper's 1206.5 ms.

## Files

| File | What it is |
|---|---|
| `work/export/export_vae_encoder_2d.py` | the T=1 collapse + ONNX export, with the equivalence asserted per-conv |
| `work/pipeline/capture_first_frames.py` | real SSD1B first frames, calib and test splits |
| `work/device/enc_flops.py` | MAC count, collapsed vs naive |
| `work/qnn/convert_vae_encoder_w8a16.sh` | W8A16 PTQ; `vaeenc` (both I/O NCHW) and `vaeencn` (image left channel-last) |
| `work/device/make_enc_io.py` | fp32 ONNX references, device `input_list.txt`, optional NHWC copies |
| `work/device/push_enc.ps1` / `run_enc.sh` / `compare_enc.py` / `eperf_report.sh` | device loop |

## Status

- 4b-i source audit — **done**: the `deque` is off the critical path
- 4b-ii T=1 collapse + ONNX export — **done**: 129.6 dB, 0 five-D ops, 363.33 GMAC
- 4b-iii calibration capture — **done**: 64 vbench + 16 disjoint showcase frames
- 4b-iv W8A16 conversion + `analyze_net.py` — **done**: ratio 0.05, 0 float fallback
- 4b-v device accuracy and latency — **done**: **41.60 dB, 166.83 ms**
- 4b-vi `vaeencn` NHWC-image A/B — **dropped**, and the static measurement says why (above)

**Phase 4b is done.** Shipping artefact: `work/device/vaeenc_v79.bin` (41.67 MB).
Input `image` [1,3,320,512] NCHW in [-1,1]; output `moments` [1,32,40,64] NCHW, split
`[mean | logvar]` on channels. The caller does the sampling and the shift/scale.

| open question | status |
|---|---|
| **Is the `deque` feature cache exportable?** | **moot on the E2E path.** `temporal_chunk=False` never touches it. Still unsolved for full-video encoding, which nothing in the AR t2v flow needs. |
| **The 1206.5 ms budget** | **beaten by 7.2×.** Cause of the gap was almost certainly convolving two planes of zero padding at T=1. |
| **40 dB target** | **met at 41.60 dB** with plain PTQ. No AdaRound or AIMET encodings needed. |
| Attention transposes | 20.96 MB of the graph's 22.12 MB, ~5% of compute traffic. Only worth revisiting if the encoder ever becomes the E2E bottleneck, which at 166.83 ms it is not. |
| `mid_attentions.0` 16-bit dynamic weights | The converter warned it could not honour a 16-bit override on two dynamic attention tensors (`--use_dynamic_16_bit_weights` would). Accuracy met the target anyway, so left alone; it is the first knob to try if attention-heavy Phase 5 modules fall short. |

### What carries to the rest of the pipeline

1. **Check what the deployment path actually calls before believing a module is hard.** The
   `deque` was recorded as the blocker for this phase for a week. It is dead code at T=1.
2. **Constant padding against a size-1 axis is free work you can delete.** Three quarters of
   this graph's arithmetic was multiplying zero-padded planes. The collapse is exact, and it
   is the same family as trap #11: the cheapest wins keep coming from removing structure,
   not from tuning the backend.
3. **The converter's GroupNorm fusion works**, so PyTorch's InstanceNorm-based lowering is
   safe to export. Confirmed with a real `analyze_net.py` histogram, not assumed.
4. **Pick the SNR metric before running the comparison.** Whole-tensor SNR on `moments` reads
   46.24 dB where the number that governs quality is 41.60 dB. A concatenated output with
   mixed dynamic ranges will flatter itself by default.
