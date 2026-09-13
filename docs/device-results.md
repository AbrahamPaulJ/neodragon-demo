# DistilT5 on S25 Ultra — first on-device results

2026-08-14. Device `SM-S938B` (S25 Ultra), platform `sun` = SM8750, HTP **V79**,
QAIRT 2.49.0.260730. Build is **float (HTP executes it in fp16)** — *not* W8A16.

This is the project's first execution on real hardware.

## Headline

Residual scaling is the difference between total failure and a usable graph.

| build | NaN+Inf / 524,288 | mean SNR | cosine |
|---|---|---|---|
| **A** — float, no residual scaling | **361,402 – 374,781** | — | — |
| **B** — residual scaling S=16 | **0** | **49.07 dB** (min 47.30) | 0.999993+ |

Build A's finite values saturate at **±65408**, an fp16 value just under the 65504 max.
That is the direct on-device signature of trap #3: the residual stream peaks at 329,509 =
5.03× fp16 max, HTP runs the float graph in fp16, and there is no fp32 upcast in the layer
norm to rescue it the way PyTorch has.

Build B holds the residual stream 16× smaller (peak 20,594, 3.2× headroom). The transform is
mathematically exact — T5LayerNorm is RMS-style and scale-invariant, so dividing the input
embedding and every branch contribution by S changes nothing, and the encoder ends in a
layer norm so no compensation is needed. Verified `rel_err` 1.2e-06 in fp32 and 108–118 dB
through onnxruntime.

Input-dependence sanity check: device case0-vs-case1 `max|d| = 0.3280`, against the CPU
ONNX's `0.328051`. The graph is genuinely reading its inputs.

## Latency

60 inferences, `--perf_profile burst`, `--profiling_level detailed`:

| metric | value |
|---|---|
| Accelerator execute (pure NPU compute) | **18.7 ms** |
| Accelerator execute, cycles | 14,481,856 |
| NetRun average (incl. RPC + IO) | 46.3 ms |
| NetRun min | 37.7 ms |
| Throughput | 20.3 inf/sec |
| HVX threads | 6 |
| VTCM acquire | 0.9 ms |

**The paper reports DistilT5 at 3.5 ms on the same 8 Elite Gen4 silicon.** We are at 18.7 ms
of accelerator time — **5.3× slower**.

> **CORRECTION (after reading the paper at source, see `paper-notes.md`).** An earlier draft
> of this file said the gap was "expected" because the paper's figure was W8A16 and ours is
> fp16. **That was wrong.** Table 9 deploys DistilT5 in **FP16**, exactly as we have. The
> comparison is like-for-like and the 5.3× gap is real and unexplained.
>
> Prime suspect: our graph leaves T5's RMSNorm decomposed — 37 `Pow`, 25 `ReduceMean`,
> 25 `Sqrt`, 25 `Div` — rather than matched to a fused HTP op. The context binary is 260 MB
> for ~130 M params, i.e. weights are already fp16, so weight precision is *not* the cause.
> `vtcm_mb: 8` / `O: 3.0` were inherited from the 8gen2 SD config and never tuned for V79.

The 46.3 ms NetRun figure is dominated by RPC (44.4 ms); it reflects `qnn-net-run`'s
per-inference host round-trip, not the model. Use the 18.7 ms accelerator number when
comparing against the paper.

## Latency investigation — where the 18.66 ms goes

Per-op cycles from `--profiling_level detailed`, 60 inferences, aggregated by
`work/device/analyze_prof.py`. Derived accelerator clock **776 MHz**; per-op cycles sum to
14,481,857 = 18.66 ms, matching the reported accelerator time exactly.

| op | ms | % |
|---|---|---|
| `block_N_layer_0_dropout_Mul` | **3.880** | **20.8** |
| `SelfAttention_Softmax` | 3.471 | 18.6 |
| `SelfAttention_MatMul` | 2.859 | 15.3 |
| `final_projection_3_MatMul` | 1.565 | 8.4 |
| `DenseReluDense_act_Mul` | 1.041 | 5.6 |
| `block_N_layer_1_dropout_Mul` | **0.894** | **4.8** |
| `DenseReluDense_wi_1_MatMul` | 0.703 | 3.8 |
| `SelfAttention_Add` (pos bias + mask) | 0.631 | 3.4 |
| `SelfAttention_o_MatMul` | 0.622 | 3.3 |
| `rms_norm` (both, all blocks) | 1.234 | 6.6 |

By kind: MatMul 33.9%, Mul 33.5%, Softmax 18.6%, rms_norm 6.6%.

**The RMSNorm-fusion hypothesis was wrong.** The converter already emits a fused
`rms_norm` op (visible as `rms_norm__enc_encoder_encoder_block_N_layer_M_layer_norm_`), and
it costs only 6.6% of runtime. Nothing to win there.

**The actual top cost was self-inflicted.** `block_N_layer_0_dropout_Mul` and
`block_N_layer_1_dropout_Mul` are the residual-scaling multiplies introduced in v2 —
together **4.77 ms, 25.6% of inference**, for arithmetic that belongs at build time.

### Fix: fold the scale into weights (v3)

Each branch ends in a bias-free linear layer — `SelfAttention.o` and `DenseReluDense.wo` —
so scaling those weights by 1/S scales the branch output identically, at zero runtime cost.
Same for the input embedding. Implemented in `work/export/export_distilt5_folded.py`.

- ONNX nodes 547 → **522**, `Mul` 111 → **86** (25 removed)
- Peak residual unchanged at 20,594; `rel_err` vs fp32 unchanged at 1.21e-06
- v2 vs v3 on CPU: **`max|diff| = 0.000e+00`, bit-identical** — pure performance change
- Context binary built and staged: `work/device/distilt5f_v79.bin`, compiled clean for V79,
  `spill_bytes=0`

### Measured result

Per-op profile confirms both `dropout_Mul` ops are **gone** — `Mul` drops to 6,205 cycles
(0.0% of runtime, from 25.6%).

Latency needs care: **the DSP throttles**. A standalone v3 run clocked at 596 MHz against
the earlier v2 run's 776 MHz, which inflated v3's cycle counts across unrelated ops and made
a naive comparison show v3 as *slower*. Averages swing 15–26 ms between rounds. The valid
comparison is interleaved in one thermal session, using **min-of-100** per round
(`work/device/ab_latency.sh`):

| round | A: runtime Mul | B: folded | delta |
|---|---|---|---|
| 1 | 14,960 µs | 13,186 µs | −11.9% |
| 2 | 14,797 µs | 12,498 µs | −15.5% |
| 3 | 15,020 µs | 12,835 µs | −14.5% |
| **mean** | **14.93 ms** | **12.84 ms** | **−14.0%** |

**Real, reproducible −2.09 ms / −14.0%**, consistent across all three rounds, at unchanged
accuracy (device SNR 49.04 dB vs 49.07; CPU bit-identical).

Note the gap between prediction and result: the profiler attributed 4.77 ms (25.6%) to the
two multiplies, but removing them saved 2.09 ms (14.0%). Per-op attribution overstates
*marginal* cost — deleting an op lets the compiler re-fuse and does not remove the associated
memory traffic proportionally. Treat profile attribution as a ranking, not a budget.

Against the paper's 3.5 ms, on the same min metric: **4.27× → 3.67×**.

## fp16 activation graph — hypothesis REJECTED

The remaining candidate after the weight fold was: the ONNX declares fp32 tensors while HTP
executes in fp16 and stores fp16 weights, so maybe conversions were being paid at runtime.
Tested and **decisively wrong — the fp16 graph is 3.7× slower.**

Interleaved, min-of-100 per round (`work/device/ab_fp16.sh`):

| round | F: fp32-declared | H: fp16-declared |
|---|---|---|
| 1 | 12,471 µs | 47,991 µs |
| 2 | 13,166 µs | 48,943 µs |
| 3 | 13,508 µs | 48,753 µs |
| **mean** | **13.05 ms** | **48.56 ms** |

And it costs accuracy too: device SNR **38.87 dB vs 49.04 dB**, a ~10 dB regression.

Tell-tale sign, visible before the run: the fp16 context binary is **260,595,368 bytes**
against the fp32-declared build's **260,050,504** — essentially identical. The ONNX was half
the size (247.7 MB vs 495.1 MB), so the converter is clearly normalising to its own internal
representation regardless of the declared ONNX dtype. Declaring fp16 does not buy a
narrower deployment; it only adds conversion work.

**Conclusion: let the QNN converter own precision.** The ONNX should stay fp32 and describe
the maths; the backend decides the execution dtype. This likely generalises to every module
in the pipeline, so it is worth not repeating for the MMDiT.

### Two numerical facts learned on the way

1. **Residual scaling is bounded above by the layer-norm epsilon.** T5LayerNorm is
   scale-invariant only while `variance >> variance_epsilon`. Scaling the stream by 1/S
   scales the variance by 1/S², so with the stock `layer_norm_epsilon: 1e-6`, S=4096 pushed
   `rel_err` from 1.2e-06 to **5.6e-02**. Scaling each `variance_epsilon` by 1/S² alongside
   restores exactness — `rel_err` then becomes **0.00e+00** in torch, and it improved the
   S=16 build too (1.21e-06 → 0.00e+00). This fix is now in
   `export_distilt5_folded.py::fold_residual_scaling` and should carry to any module where
   this trick is reused.

2. **A genuine fp16 interior has a narrow viable window for S.** `mean(h²)` must stay in
   fp16 range, which needs S ≥ 4096 (at S=16, `sum(h²)` ≈ 2.1e9); but larger S pushes typical
   `h²` toward fp16 subnormals. Only S≈4096–8192 satisfies both, and it is tight. Moot given
   the result above, but it explains the 10 dB loss.

## Postscript 2026-08-21 — DistilT5 as the layout control

When the Phase 4 VAE decoder turned out to be 24x off because of layout transposes, this
graph became the control that made the finding readable. `work/device/analyze_net.py` on
`distilt5_net.json`:

| | DistilT5 (3.7x off) | VAE dec T=1 (24x off) |
|---|---|---|
| Transpose nodes | 48 | 47 |
| **Transpose output bytes** | **18.87 MB** | **779.80 MB** |
| transpose / compute | **0.28** | **1.16** |

Near-identical Transpose *count*, 41x the bytes. DistilT5's are attention head permutes on
0.39 MB tensors; the VAE's were full-resolution feature maps. **Node count is not the
discriminator — bytes moved is.** 0.28 is the number to treat as "livable" when screening
future modules; the fixed streaming decoder reaches 0.09.

This also means the remaining 3.67x on DistilT5 is *not* a layout problem, and stays
unexplained. At 12.84 ms against a multi-second pipeline it is not worth chasing.

## Compile telemetry (host, WSL)

Both builds compiled clean for V79, `spill_bytes=0` / `fill_bytes=0` — no VTCM thrashing.

| stage | build A | build B |
|---|---|---|
| Graph Optimizations | — | 2,544 ms |
| Graph Sequencing for Target | 391 ms | 393 ms |
| VTCM Allocation | 45 ms | 32 ms |
| Parallelization Optimization | 240 ms | 161 ms |
| context binary | 260,050,504 B | 260,085,072 B |

## Gotcha worth recording

`qnn-net-run` parses input raw files as **float32 by default**, regardless of the graph's
declared input dtype. Feeding int32 `input_ids` without `--use_native_input_files` silently
produces garbage — every case reads as near-zero, so all outputs come back *identical* and
plausible-looking (SNR 2.2 dB, cosine 0.69). It does not error. Always pass
`--use_native_input_files` for integer inputs, and always assert that two different prompts
produce different outputs.

## Reproduce

```powershell
py -3.10 work\export\export_distilt5_scaled.py --scale 16
wsl -d Ubuntu -e bash work/qnn/convert_distilt5_scaled.sh
py -3.10 work\device\make_inputs.py
# push, then:
#   ./qnn-net-run --backend lib/libQnnHtp.so \
#     --retrieve_context ctx/distilt5s_v79.bin --input_list io/input_list.txt \
#     --output_dir outs --use_native_input_files --perf_profile burst
py -3.10 work\device\compare2.py
```

## Next

> **Superseded.** An earlier draft listed "W8A16 quantise DistilT5" as step 1. Table 9 of the
> paper deploys DistilT5 in **FP16** — quantising it would deviate from the reference design,
> and the a16 resolution concern does not apply to this module at all. See `paper-notes.md`.

**Phase 1 is architecturally complete.** FP16 + residual scaling *is* the reference
configuration for DistilT5. What remains is optimisation, not correctness:

1. **Close the 5.3× latency gap** — start by checking whether T5's RMSNorm is being matched
   to a fused HTP op, since the graph currently carries 112 loose elementwise norm ops.
   Cheap, and the lesson transfers directly to the 938 ms MMDiT-high stage.
2. **Phase 2 (QuickSRNet)** — pre-compiled QNN binaries exist on AI Hub; paper says 6.5 ms
   and W8A16+AdaRound at 48 dB.
3. **Phase 4 (TinyAEHV / VAE Dec)** — W8A16 PTQ, only **50** calibration samples, 35 dB
   target. The small calibration set makes this far cheaper than assumed.
