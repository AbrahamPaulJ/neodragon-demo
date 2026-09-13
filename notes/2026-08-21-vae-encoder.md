# Session 3 — 2026-08-21 — Phase 4b, the VAE encoder

Continuation of the session that closed Phase 4 the same day. That session's last act was to
find an entire missing phase and warn:

> It is the hard half of the "asymmetric" VAE … **not exportable as written** … Paper puts it
> at **1206.5 ms**, the most expensive component in Table 7.
> **Do not quote an E2E timeline without scoping this first.**

Scoped and shipped in one session: **41.60 dB, 166.83 ms**, both better than the paper.

## What actually happened

I expected to spend the session porting the `deque` feature cache to explicit state, reusing
`export_vae_decoder_stream.py`. Two greps killed that plan before any code was written.

**First**, `CausalConv3d.forward` keeps the `deque` entirely inside the `temporal_chunk=True`
branch. **Second**, the only call site that matters —

```python
# generation_utils.py:526-531
image = rearrange(image, "(b t) h w c -> b c t h w", b=1, t=1)
image_latent = vae.encode(image).latent_dist.sample()
```

— takes `encode()`'s defaults, i.e. `temporal_chunk=False`, on a tensor with **T = 1**. The
stateful branch is never executed on the E2E path. The cache is still unexportable; it is
simply dead code here.

That reframed the whole phase, and the second finding fell straight out of it: at T = 1,
`F.pad(x, time_causal_padding)` prepends **two planes of zeros**, so a time-3 causal conv
computes `w[0]·0 + w[1]·0 + w[2]·x`. The encoder is a 2-D convnet wearing a 3-D costume.
Collapsing it is exact — the discarded terms are multiplications by zero — and it holds for
the stride-2 temporal downsamplers too, because `(3-3)/2+1 == 1` output frame covering the
same three planes.

Measured 129.6 dB against the stock encoder on two input distributions. That is fp32
accumulation-order noise.

## The numbers

| | naive T=1 5-D | collapsed 2-D | paper |
|---|---|---|---|
| MACs | 1071.2 GMAC | **363.33 GMAC** | — |
| Conv3d / 5-D ops | 30 / many | **0 / 0** | — |
| transpose ÷ compute bytes | — | **0.05** | — |
| deploy SNR (16 held-out frames) | — | **41.60 dB** | 40 dB |
| latency | — | **166.83 ms** | 1206.5 ms |

361.03 GMAC in 166.83 ms is **2.16 TMAC/s**, against the decoder's 205.07 GMAC in 100.18 ms
= 2.05 TMAC/s. Same chip, same efficiency — which says the collapse is the entire story and
there is no hidden inefficiency left to hunt. A 1071.2 GMAC build at that rate is ~496 ms
before any 5-D layout traffic, the right order for the paper's figure.

## Things that went right, deliberately

- **`analyze_net.py` before device time** (trap #12). Ratio 0.05, 0 float fallback. It also
  answered a question I would otherwise have burned a 20-minute conversion on: an NHWC-image
  `vaeencn` variant would move 0.98 MB of 22.12 MB of transpose traffic — 0.2% of total. The
  decoder's equivalent A/B was worth 10%; this one is not worth building. The variant stays
  in the convert script, unbuilt, with the measurement recorded next to it.
- **The metric was chosen before the comparison.** `moments` is `[mean | logvar]` on the
  channel axis, and logvar sits near −30 while mean has std 1.85. Whole-tensor SNR reads
  46.24 dB; the mean-half figure that actually governs quality is 41.60 dB. Had I reported
  the default I would have claimed +6.2 dB over the paper instead of +1.6.
- **Calibration is real and the test set is disjoint** (trap #4). 64 SSD1B first frames from
  vbench prompts calibrate; 16 from showcase prompts are held out. The encoder's input
  depends only on SSD1B and the LANCZOS resize, so `capture_first_frames.py` skips the
  35 s/video DiT loop entirely — ~2 s/image instead.
- **trap #8 guards are in the tooling, not in my head.** `make_enc_io.py` asserts each case
  differs from the previous one, and `compare_enc.py` re-checks it on the device outputs.

## Things that bit

- **Git Bash mangles WSL paths too.** `wsl -d Ubuntu -e bash /mnt/c/...` run from the Bash
  tool became `C:/Program Files/Git/mnt/c/...` and exited 0 having done nothing. It only
  looked wrong because the log was one line long. Run `wsl` from PowerShell — this is the
  same lesson CLAUDE.md already records for adb, and it applies to `wsl` as well.
- **Heredocs, again.** A `cat <<'PYEOF'` of the export script died with "unexpected EOF".
  Third session running that heredocs have cost time; large files go through the Write tool.
- **Assertions on scripted edits keep paying.** The first CLAUDE.md patch asserted and failed
  instantly because I had typed `--` and `x` where the file has `—` and `×`. Ten seconds to
  find, versus a silent no-op replace.

## Converter note worth keeping

`torch.onnx.export` never emits opset-18 `GroupNormalization` — at 17, 18 and 20 alike it
lowers `nn.GroupNorm` to Reshape → InstanceNormalization → Reshape → Mul → Add, and that
first reshape folds channels into a flattened axis, i.e. trap #11's shape on 22 of the
graph's largest tensors. I probed this expecting to have to write a fusion pass. Not needed:
`op_graph_optimizations.py`'s `OptimizeGroupNormTranslation` matches the pattern and emits a
native `GroupNorm`, which `htp.json` lists as supported. All 22 arrived fused. **Phase 3's
SSD1B UNet is full of GroupNorms and can rely on this.**

The converter also warned it could not honour a 16-bit override on two dynamic tensors inside
`mid_attentions.0` (`--use_dynamic_16_bit_weights` would). Accuracy met target regardless, so
it was left alone — but it is the first knob to try if an attention-heavy Phase 5 graph falls
short.

## State at end of session

- Phases 1, 4 and **4b** done on device. Phases 2, 3, 5, 6 untouched.
- **Shipping artefact: `work/device/vaeenc_v79.bin`** (41.67 MB). Input `image`
  [1,3,320,512] NCHW in [−1,1]; output `moments` [1,32,40,64] NCHW, `[mean | logvar]` split
  on channels. Sampling and the shift/scale stay on the caller.
- `work/calib/vae_enc/` is ~252 MB including the `nhwc/` copies. **Deletable** — regenerate
  with `capture_first_frames.py` in ~3 min. The `nhwc/` copies are only needed if anyone ever
  builds `vaeencn`, which the transpose measurement says not to.
- Two of the three E2E VAE questions are now closed. **Phase 5 (Pyramidal MMDiT) is the only
  remaining structural risk**, and trap #1's RoPE rank-6→4 rewrite should be applied *before*
  its first conversion, not after.
