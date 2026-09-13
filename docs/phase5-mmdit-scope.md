# Phase 5 scoping — the Pyramidal MMDiT

*Scoped 2026-08-21, immediately after Phase 4b closed. Entry cost was an open decision.*

Two corrections to what CLAUDE.md and the brief have said about this phase, both from
reading the call site rather than the module.

## Correction 1 — `num_stages` is always 1 at the call site

CLAUDE.md records:

> `PyramidMMDiT.forward` takes `List[List[Tensor]]`, loops stages in Python and
> boolean-scatters (`__init__.py:299`). A single-stage forward must be written from scratch.

The signature is real, but the AR t2v path never uses its generality. `_generate_one_unit`
loops the pyramid stages **outside** the model, one DiT call per stage
(`generation_utils.py:328-334`):

```python
latent_model_input = past_conditions[stage] + [latent_model_input]
noise_pred = dit(sample=[latent_model_input], ...)      # <-- outer list, length 1
```

`merge_input` sets `num_stages = len(sample)`, so **`num_stages == 1` on every invocation**.
Everything that looked like Python stage machinery degenerates:

| construct | at `num_stages == 1` |
|---|---|
| `for i_p, length in enumerate(hidden_length)` in `VarlenSelfAttentionWithT5Mask.__call__` | one iteration, straight line |
| `encoder_qkv[i_p::num_stages]`, `text_ids[i_p::num_stages]` | `[0::1]` — the whole tensor |
| `torch.stack(output_encoder_hidden_list, dim=1)` + `rearrange("b n s d -> (b n) s d")` | identity at `n == 1` |
| `torch.cat(output_hidden_list, dim=1)`, `torch.split(..., hidden_length, dim=1)` | identity |
| `text_ids[encoder_attention_mask == 0] = 0` (the boolean scatter at `__init__.py:299`) | **host-side constant** — see below |

The inner `List[Tensor]` is the one that is real: `past_conditions[stage] + [current]`, a
handful of latents at different pyramid resolutions that `PatchEmbed3D.forward` patchifies
and concatenates into one token sequence. That is a fixed concatenation for a fixed shape.

**The whole of `merge_input` is shape-and-mask arithmetic.** `attention_mask`,
`image_rotary_emb` and the cropped `pos_embed` depend only on the token layout and on
`encoder_attention_mask` — never on the latent *values*. For a static graph they are host-side
constants, computed once and either passed in as inputs or folded. Nothing needs a scatter on
device. This also disposes of the 226 MB `pos_embed` buffer: it is cropped and interpolated to
the stage's grid on the host, so only the crop ships.

## Correction 2 — "3 static graphs" is 18 distinct shapes, collapsible to 3 *padded* ones

The brief's "3 static graphs" comes from paper Table 7's three rows, `[7×10×16]`,
`[7×20×32]`, `[7×40×64]` — which are `T × latent_H × latent_W`, matching this config's three
pyramid resolutions exactly (320×512 → latent 40×64 → stages 10×16, 20×32, 40×64).

But the token count is not fixed within a stage. `_prepare_past_condition_latents` grows the
clean history each unit, so every `(unit, stage)` pair has its own sequence length.
`work/audit/mmdit_shapes.py` replays the arithmetic — no weights, no GPU:

```
  unit stage  tokens                 input latents [(T,H,W), ...]     n_in
     1     0      80                              1x10x16 1x10x16        2
     1     1     320                              1x20x32 1x20x32        2
     1     2    1280                              1x40x64 1x40x64        2
     ...
     6     0     280                      5x10x16 1x10x16 1x10x16        3
     6     1     520                      5x10x16 1x20x32 1x20x32        3
     6     2    1600              4x10x16 1x20x32 1x40x64 1x40x64        4

DISTINCT (stage, token count) combinations: 18
  stage 0: [ 80, 120, 160, 200, 240, 280]
  stage 1: [320, 360, 400, 440, 480, 520]
  stage 2: [1280, 1440, 1480, 1520, 1560, 1600]
```

Note the per-stage **maximum** is reached at the last unit and is exactly 7 latent frames'
worth — 280 = 7·(10/2)·(16/2). So paper Table 7's `[7×…]` rows are the *max* shape per stage,
which strongly suggests the reference deployment ran **one graph per stage padded to its
maximum**, and quoted that. Its 104.7 / 218.3 / 938.3 ms are therefore already
padded-to-max numbers, and matching them needs no cleverness about the tail.

Padding cost, worst case (unit 1, the shortest sequence):

| stage | pad to | worst-case wasted tokens | paper latency |
|---|---|---|---|
| 0 | 280 | 71% | 104.7 ms |
| 1 | 520 | 38% | 218.3 ms |
| 2 | 1600 | 20% | 938.3 ms |

The waste is concentrated where it is cheap. Stage 2 costs 74% of the MMDiT budget and wastes
only 20%; stage 0 wastes 71% of 104.7 ms. **Three padded graphs is the right design** — 18
context binaries would each embed the full DiT weights and is not a real option.

Two requirements that come with padding:

1. **Pad at the front.** `split_output` slices `hidden_states[:, -trainable_token_num:]`, so
   the *trailing* tokens must stay the current latent. Padding the history side keeps the
   output slice fixed.
2. **Mask the padding, with a finite value.** `merge_input` already builds a boolean
   `attention_mask`; padded keys simply join the masked set. A fully-masked query row would
   make softmax produce NaN, and SDPA lowers a boolean mask to literal `-inf` — this is
   **trap #2 exactly**, already solved by `work/export/graph_fixes.py`'s
   `retarget_extreme_constants(fill=-100)`, which Phase 1 verified at 2.8e-42 leakage.

## Correction 3 — it traces. The problem is *what* it traces.

`work/audit/mmdit_trace_probe.py` builds the real 1512.2 M-param DiT, feeds it exactly what
`generation_utils.py:328-334` feeds it, runs it, and then calls `torch.onnx.export` on the
**unmodified** module:

```
[input] 3 latents 5x10x16 1x10x16 1x10x16  -> 280 image tokens + 128 text = 408
[run]   OK -- output list of 1, tensor (1, 16, 1, 10, 16)
[export] SUCCEEDED -- 1.9 MB  (+ 4.1 GB of external weight data)
[ops]    Constantx2981, Mulx999, Addx665, Unsqueezex415, Slicex385, Reshapex314,
         Expandx259, MatMulx250, Concatx231, Castx229, ConstantOfShapex201, Wherex196,
         Shapex186, Gatherx184, Equalx179, Divx127, Sqrtx126, Transposex80
```

So "not exportable as written" was wrong. The `List[List[Tensor]]` signature just needs a
positional wrapper, and the model exceeds 2 GB so torch switches to external-data format
automatically (which scatters ~680 sidecar files next to the `.onnx` — export into a
dedicated directory).

The real problem is the op histogram. `Shape`×186, `Gather`×184, `Equal`×179, `Where`×196,
`ConstantOfShape`×201, `Expand`×259 are **`merge_input`'s mask construction traced into the
graph**. `do_constant_folding=True` could not remove them because they descend from
`encoder_attention_mask`, which is a graph *input* — the prompt's padding pattern genuinely
varies per prompt, so the folder is right to keep them.

That makes the rewrite concrete, and much smaller than "write a single-stage forward from
scratch":

> **Hoist `merge_input` to the host.** Compute `attention_mask` and the RoPE tables on the
> host and pass them in as ordinary tensor inputs. They depend only on the token layout and
> the prompt mask — never on the latent values — so the graph loses every `Shape`/`Gather`/
> `Where`/`Equal`/`ConstantOfShape` node with it, and the 226 MB `pos_embed` buffer never
> ships because its crop-and-interpolate is host-side too.

`Transpose`×80 is the attention q/k/v path and is the number to watch in `analyze_net.py`.

## Model size, and a deployment constraint worth flagging early

**1512.2 M parameters** — CLAUDE.md's "3.1 GB" is the bf16 checkpoint, i.e. 6.05 GB in fp32.
At W8 that is ~1.5 GB of weights *per graph*, and QNN context binaries embed their own
weights, so three padded stage graphs is **~4.5 GB of context binaries**. Storage is fine
(46 GB free on the device). What needs checking before promising an E2E number is residency:
the AR loop visits stage 0, 1, 2 in sequence every unit, so either all three stay resident or
every unit pays three context loads. This is the first RAM-budget question the port has
actually faced — Phase 1's DistilT5 is 260 MB and Phase 4's VAE binaries are 12–42 MB.

## The latency picture this implies

18 DiT invocations per 49-frame video (6 units × 3 stages, 1 step per stage):

```
6 × (104.7 + 218.3 + 938.3) = 7567.8 ms
```

Against the measured VAE encoder (166.83 ms, once) and decoder (0.70 s for the whole 49-frame
video), **the MMDiT is essentially the entire E2E budget** — about 90% of it. Every remaining
optimisation opportunity on this port lives in Phase 5.

## Traps that apply here, before the first conversion

- **Trap #1, RoPE rank-6 tiling.** `rope()` returns `[bs, seq, 1, head_dim//2, 2, 2]` and
  `apply_rope` contracts it against a `[..., -1, 1, 2]` reshape of q and k. Rank 6. The
  rank-4 replacement was verified bit-exact during the Phase 1 audit and must be applied
  **before** the first conversion, not after — this is the same failure mode Phase 4 paid
  53× for. Since the sequence layout is static, `cos`/`sin` are constants: pass them as two
  `[1, seq, 1, head_dim//2]` tensors and compute
  `out_even = cos·x_even − sin·x_odd`, `out_odd = sin·x_even + cos·x_odd`.
- **Trap #3, residual stream in A16.** Unlike DistilT5 — which ships FP16 and so escapes it —
  the MMDiT *is* W8A16 and *does* carry a residual stream through 18 blocks. `trap-audit.md`
  §3's effective-bits finding applies here for real. Expect this to be the accuracy fight;
  paper Table 9 only claims 22–29 dB for these three rows, the worst in the pipeline.
- **Trap #12, `analyze_net.py` first.** The attention path transposes q/k/v per block. At
  stage 2, 1728 tokens × 24 heads × 64 dim is a 143 MB A16 score matrix per block.

## 5c — the rewrite, built and verified bit-exact

`work/export/export_mmdit_stage.py`. The principle is **patch the plumbing, keep the math**:
every weight-bearing module (blocks, feed-forwards, norms, projections) is the stock object,
untouched. Four things are replaced:

| replaced | with |
|---|---|
| `merge_input` | host-side `stage_conditioning()` returning `attn_mask`, `rope_cos`, `rope_sin` as ordinary tensors |
| `PatchEmbed3D.forward` | precomputed per-latent `pos_embed` crops as buffers — the 226 MB table never reaches the graph |
| `AdaLayerNorm*.forward` | the plain broadcast path; `hidden_length` dropped |
| `apply_rope` | **rank 3** (trap #1): `cos`/`sin` at `[S, 1, 32]`, rotation computed directly. Rank 4 was the first form — see §5d for why it dropped to 3 |

Verified against the stock `PyramidMMDiT` at **every one of the 18 shapes**:

```
[check] unit 1 stage 0     80 img tok  (1, 16, 1, 10, 16)   max|diff|=0.000e+00  SNR=inf dB
...
[check] unit 6 stage 2   1600 img tok  (1, 16, 1, 40, 64)   max|diff|=0.000e+00  SNR=inf dB
[check] PASS -- worst max|diff| over 18 shape(s): 0.000e+00
```

**Bit-exact, including the RoPE replacement.** Trap #1's rewrite was verified bit-exact
during the Phase 1 audit in isolation; this is the same result inside the real model.

> **Superseded in one respect.** The numbers above are the rank-4 attention form. §5d drops
> the attention to rank 3 to fix the layout traffic, and bit-exactness goes with it:
> reordering the attention changes SDPA's accumulation blocking, so the **shipping** rewrite
> scores **124–128 dB**, not `inf`. That is the fp32 floor (~1e-5 on O(1) tensors), ~95 dB
> below the 29 dB quantisation target — but it is no longer bit-exact and should not be
> described as such.

### What left the graph

| op | stock trace | after 5c |
|---|---|---|
| `ConstantOfShape` | 201 | **0** |
| `Where` | 196 | **0** |
| `Equal` | 179 | **0** |
| `Expand` | 259 | **0** |
| `Shape` | 186 | 55 |
| `Gather` | 184 | 37 |
| `Transpose` | 80 | 79 |

The `ConstantOfShape`/`Where`/`Equal`/`Expand` block is `merge_input`'s mask construction
plus `AdaLayerNorm*.forward_with_pad`'s `zeros_like(x).repeat(1,1,6)`-and-scatter, which ran
once per block. The first `Shape` reduction (186 → 112 → 55) came from never asking a tensor
for a dimension the shape list already knows — `t.view(t.shape[0], ...)` emits Shape+Gather
that constant folding cannot remove. The remainder is SDPA's own lowering and folds in the
ORT pass.

## Padding: 18 shapes through 3 graphs, verified

`--pad-check` runs every short unit through its stage's envelope graph and compares against
the stock model at that unit's true shape:

```
  stage 0 envelope 5x10x16 1x10x16 1x10x16  (280 image tokens)
    unit 1  + 200 padded tokens  max|diff|=8.821e-06  SNR=123.1 dB
    ...
  stage 2 envelope 4x10x16 1x20x32 1x40x64 1x40x64  (1600 image tokens)
    unit 1  + 320 padded tokens  max|diff|=1.192e-05  SNR=126.9 dB
    ...
[pad]   PASS -- the envelope graph reproduces every short unit
```

123–128 dB is the fp32 accumulation-order floor: `MASK_FILL = -100` leaks `e^-100 ≈ 4e-44`,
which is zero in fp32, so the padded tokens contribute nothing.

**The subtlety that makes this work is a direct consequence of hoisting RoPE to the host.**
Prepending frames naively would shift every image token's position id — and the text tokens
sit at id 0 and are *not* shifted, so every text-to-image relative position would change and
every score with it. Because the host now chooses the id of every slot, real tokens keep the
ids they would have had at their true shape and only the padded slots get filler. The
mask hides them; a fully-masked query row is finite rather than NaN precisely because
`MASK_FILL` is −100 and not −inf (trap #2).

## 5d — the transpose fight, and how it was won

`analyze_net.py` on the first stage-0 conversion said **ratio 1.35** — worse than the VAE
decoder's catastrophic 1.16. Three rounds of static iteration, no device time:

| build | Transpose nodes | transpose bytes | ratio |
|---|---|---|---|
| rank-4 attention, everything pinned | 337 | 497.45 MB | **1.35** |
| rank-3 attention, everything pinned | 191 | 407.15 MB | **1.10** |
| rank-3 attention, **`--input_layout NONTRIVIAL`** | **85** | **184.08 MB**† | **0.25** |

† the last row is measured on a float (layout-only) build, so its bytes are fp32; at A16
it is ~92 MB. The first two rows are quantised builds. The ratio is comparable across all
three because both sides of the ratio scale together.

### Diagnosis

`analyze_net.py` says *how much*; `work/device/trace_transposes.py` (written for this) says
*why*. It groups every Transpose by `(perm, in-shape, out-shape, producer type, consumer
types)` and ranks by total bytes:

```
     287.6 MB total   x18   15.98 MB each   (35% of all transposes)
    MatMul -> Transpose -> Eltwise_Binary        [24, 408, 408]  ->  [24, 408, 408]

     287.6 MB total   x18   15.98 MB each   (35% of all transposes)
    Eltwise_Binary -> Transpose -> Softmax       [24, 408, 408]  ->  [24, 408, 408]
```

**70% of all layout traffic was two permutations of the attention score matrix per block.**
The shapes are identical on both sides — it is a pure axis-format change, invisible in the
dims. The converter's own tensor names say it outright: `_MatMul_output_0_nfc` followed by
`_Add_5_output_0_ncf`. **MatMul and Softmax run NCF; the mask-add `Eltwise_Binary` runs
NFC.** Every block paid a round trip on its largest tensor.

Dropping from rank 4 to rank 3 did *not* escape this — it only renamed it. QNN has a
layout rule for rank-3 tensors (NCF/NFC) just as it does for rank-4 (NCHW/NHWC). Rank is
not an escape hatch; **absence of layout semantics is.**

### The fix

`--input_layout <name> NONTRIVIAL`, for every input that is not image-like:

```
--preserve_io layout latent_0 latent_1 latent_2 noise_pred
--input_layout attn_mask NONTRIVIAL --input_layout rope_cos NONTRIVIAL ...
```

NONTRIVIAL means "this tensor has no layout semantics", so the converter stops trying to
place it in a canonical format and stops dragging its consumers along. Only the latents keep
image semantics: they are 5-D and feed `pos_embed.proj`, a real Conv2d, where trap #7 applies.

**NONTRIVIAL is also the safe choice.** Unlike NHWC — which is what the Phase 4 decoder
needed, and which required `permute_states_nhwc.py` on the host — NONTRIVIAL keeps the bytes
as authored. Verified in the net.json: every graph input and the output came back with
`permute_order_to_src = [0, 1, 2, ...]`, i.e. identity. There is no host-side permutation to
get wrong, so no trap #8 exposure.

What survives is 4 groups × 18 blocks of `[408,24,64] ↔ [24,408,64]` — the token-major /
head-major swaps for q, k, v and the attention output. Those are genuine multi-head
attention permutations, not layout artefacts, and 0.25 is close to the floor for this
formulation (DistilT5 ships at 0.28).

### Two tools this produced

- **`work/device/trace_transposes.py`** — the "why" to `analyze_net.py`'s "how much".
  Reach for it the moment a ratio looks wrong.
- **`convert_mmdit_w8a16.sh <stage> --layout [pinned|nontrivial]`** — conversion with
  **no `--input_list` at all.** Axis tracking happens in IrOptimizer passes that run before
  the quantiser, so a float build makes the same layout decisions. **Validated as a faithful
  proxy**: the layout-only `pinned` build reproduced the quantised build's structure exactly
  — same ratio 1.10, same 191 nodes, same groups, only the bytes doubled (fp32 vs A16).
  It also removes the calibration set as a dependency while iterating.

### Turning transpose bytes into milliseconds

The ratio heuristic was calibrated on a conv net, so "1.10 is nearly as bad as 1.16" is
pattern-matching, not arithmetic. The Phase 4 decoder A/B gives a real conversion factor:
`vaedecs` 63.00 MB → 114.04 ms and `vaedecsn` 7.95 MB → 100.18 ms, i.e. 55 MB bought
13.86 ms — **~0.252 ms per MB of layout traffic on this chip.**

At A16 that puts the three builds at roughly **63 ms → 51 ms → 23 ms** of pure layout
traffic per stage-0 inference, against a paper target of 104.7 ms for the whole call. Worth
fixing, and now fixed — but it was never the 53× that the raw ratio comparison implied.

### Ruled out along the way

QNN has a native **MaskedSoftmax** op (`--apply_masked_softmax compressed|uncompressed`)
that would fuse MatMul → mask-add → Softmax and remove the pair by construction. It is **not
supported on HTP** — `htp.json` lists only `Softmax` and `LogSoftmax`. Cheap to check, worth
knowing.

## Recommended build order

1. `work/audit/mmdit_shapes.py` — **done**: 18 shapes, 3 padded envelopes.
2. `work/audit/mmdit_trace_probe.py` — **done**: it runs and it traces; the mask
   construction is what pollutes the graph.
3. Hoist `merge_input` to the host: `attention_mask` and RoPE become graph inputs, RoPE at
   rank 3 from the start (trap #1 for the rank reduction, trap #23 for why not rank 4).
   Verify against the stock module at each of the 18 shapes
   before exporting anything.
4. Export the **stage 0 max shape (280 image tokens) first** — smallest graph, identical
   structure, cheapest debug loop. Export into its own directory (external data). Run
   `analyze_net.py` before any device time.
5. Then stage 1, then stage 2. Calibrate per stage (trap #4): paper uses 300 samples each.
6. Check context-binary residency for three ~1.5 GB graphs before quoting an E2E figure.

## Status

- 5a shape enumeration — **done**: 18 shapes, 3 envelopes. Validated against the real
  pipeline: a captured reference run produced exactly the 18 predicted shapes, in order.
- 5b trace probe on the stock module — **done**: it runs and it traces
- 5c host-hoisted mask/RoPE rewrite + rank-3 RoPE — **done**; bit-exact at all 18 shapes
  in its rank-4 form, **124–128 dB** in the shipping rank-3 form (the fp32 floor)
- 5d transpose fight — **done, statically**: ratio 1.35 → 1.10 → **0.25**
- 5c-pad padding to the 3 envelopes — **done, 123–128 dB (the fp32 floor)**
- 5d calibration capture — **tooling done**, `capture_mmdit_calib.py`; 18 DiT calls per
  video, so 50 videos gives the 300 samples Table 9 wants for every stage at once
- 5e stage-0 convert — **done**: 300 samples, 35:04 wall, peak RSS **10.74 GB** against
  the 11 GB WSL cap. Host RAM was never the blocker it looked like. Context binary
  **1.548 GB**, `analyze_net.py` ratio **0.25**, zero float fallback.
- 5f stage-0 on device — **done, and both targets missed**:
  **19.47 dB** (mean of 16 held-out cases, min 12.05) against a 29 dB target, and
  **248.3 ms** against the paper's 104.7 ms. No NaN, outputs distinct.
- 5g accuracy diagnosis — **root cause found**, see below. Fix not yet tried.
- 5h latency diagnosis — *not started*.
- 5i stages 1 and 2 — *not started*. Same scripts, `1` / `2`. Expect ~2× and ~4× the
  conversion time, and stage 2's calibration set is ~4 GB (its `attn_mask` is 11.94 MB per
  sample) — build one stage at a time and delete before the next.

## 5g — why stage 0 is 9.5 dB short

**K and V are 8-bit.** QNN treats a MatMul's *second operand* as a "weight", and 16-bit
dynamic weights are off by default, so in every one of the 18 attention blocks:

```
_MatMul     q[UFIXED_16] x kT[UFIXED_8]  -> scores
_MatMul_1   scores[UFIXED_16] x v[UFIXED_8]
```

The converter says so during conversion — `mixedPrecisionForWeights: ... would not be
honoured. Since 16 bit dynamic weights are not supported by default. Kindly use the flag
--use_dynamic_16_bit weights`. That warning also appeared in the Phase 4b encoder build
(`mid_attentions.0`), where it did not matter because the encoder has one attention layer
and still cleared its target. Here it is eighteen.

**The fix to try**: `--use_dynamic_16_bit_weights` (a real flag, hidden behind
`argparse.SUPPRESS` at `qnn_quantizer.py:156`) together with
`--restrict_quantization_steps "-0x8000 0x7F7F"`, whose help says it "is required for
16-bit Matmul operations" and which is accepted because we pass
`--use_per_channel_quantization`. Check `htp.json` for a MatMul datatype constraint first.

### Ruled out — do not spend a conversion re-testing these

- **Padding.** Per-case SNR does correlate with padding fraction (r = −0.66) and the three
  worst cases are all unit 1, which is 71% padded. But `work/audit/mmdit_residual_scan.py`
  shows padded tokens peak **lower** than real ones — padded/real ratio **0.59×**. Padding
  is a second-order effect at most.
- **Outlier-dominated activation ranges.** `analyze_encodings.py` found late residual adds
  with range 55,130 and step 0.84 against a graph median of 24.47, which looks exactly like
  trap #3. It is not: the residual scan puts **max/p99.99 at only 1.1–1.5×** on the image
  stream. The distribution is genuinely wide, not outlier-driven, so
  `--act_quantizer_calibration percentile` (or `mse`) would reclaim almost nothing.
- The residual **does** grow 197 → 27,922 (140×) over 18 blocks on the text stream. Real,
  and worth remembering — but per-tensor SQNR computed from (range, std) is **60–80 dB**
  everywhere, which cannot produce a 19 dB output. That is what pointed at operand
  precision instead.

## 5h — latency, uninvestigated

248.3 ms against 104.7 ms, min-of-3-rounds in one thermal session (trap #10). Transposes
are already clean (0.25 ratio, 92.04 MB ≈ 23 ms at 0.252 ms/MB) and there is **zero float
fallback**, so neither explains it. Next step is `--profiling_level detailed` plus
`work/device/analyze_prof.py`. Worth a look in the graph: `Reshape` ×778 and
`StridedSlice` ×324. And note the converter's `reEvalMacsParams` reports **2.32 GMAC** for
this graph against a hand estimate of ~617 GMAC (1512 M params × 408 tokens) — roughly
266× too low. **Do not trust that counter on this model.**

## Practical notes for 5e

- **Consolidate the external data.** `torch.onnx.export` writes one file per tensor — 659
  of them — and the converter reads over the 9p `/mnt/c` mount where per-file overhead
  dominates. The export now re-saves with `all_tensors_to_one_file=True`, and the convert
  script stages the model onto WSL's ext4 before running.
- **`attn_mask` is the biggest input**, `[1,1,S,S]` fp32: 0.67 MB at stage 0 but **11.94 MB
  at stage 2**, ~4 GB for 300 calibration samples. Build one stage at a time and delete
  before the next. If it shows up in `analyze_net.py` or on device, the fix is to pass the
  1-D order/validity vectors and rebuild the mask on device — SDPA materialises the S×S
  tensor anyway, so that trades ~6 MB of per-inference input traffic for a few elementwise
  ops. Measure first.
- **`timestep_ratio` is one of three values** (1000 / 744 / 386, constant per stage across
  every unit), not a continuous ratio. Worth knowing when reading calibration ranges.
