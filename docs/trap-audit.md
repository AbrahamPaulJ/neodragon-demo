# Static trap audit — results

Run 2026-08-14 against `qualcomm-ai-research/neodragon` @ depth-1 clone in `src/neodragon`
and the real `text_encoder_3` weights from `Qualcomm-AI-Research/Neodragon`.

Scripts live in `work/audit/`. Every number below is reproducible by re-running them;
nothing here is inferred from reading alone.

**Headline: all three statically-inspectable traps are real, located to exact lines, and
each has a fix verified numerically before any compile was launched.** One of the three is
mis-described in the research brief and needs a different fix than the brief proposes.

---

## Trap #1 — RoPE 6D tiling — CONFIRMED

**Where:** `neodragon/pyramid_mmdit/modeling_embedding.py:11-24` (`rope`),
applied at `neodragon/pyramid_mmdit/modeling_block.py:66-73` (`apply_rope`).

The source documents the problem itself, at `pyramid_mmdit/__init__.py:276`:

```python
image_rotary_emb = [self.temp_rope_embed(input_ids) for input_ids in input_ids_list]
# [bs, seq_len, 1, head_dim // 2, 2, 2]
```

`rope()` materialises the full 2×2 rotation matrix per position, and `apply_rope` reshapes
q/k to `[bs, seq, heads, head_dim/2, 1, 2]`. Both operands are rank 6; the broadcast
multiply is rank 5.

Measured (`audit_rope.py`), config `num_heads=24, head_dim=64, text_len=128`:

| stage | seq | freqs shape | rank | fp32 |
|---|---|---|---|---|
| dit-low `[7x10x16]` | 1248 | `(1, 1248, 1, 32, 2, 2)` | 6 | 0.61 MB |
| dit-mid `[7x20x32]` | 4608 | `(1, 4608, 1, 32, 2, 2)` | 6 | 2.25 MB |
| dit-high `[7x40x64]` | 18048 | `(1, 18048, 1, 32, 2, 2)` | 6 | 8.81 MB |

Three separate findings:

1. **The 2×2 block is 2× redundant.** `freqs[...,0,0] == freqs[...,1,1]` (cos) and
   `freqs[...,0,1] == -freqs[...,1,0]` (sin) both verified `True`. Four slots, two distinct
   tensors.
2. **A rank-4 replacement is bit-exact.** Standard even/odd split with `cos`/`sin` held at
   `(B,N,1,d/2)`: `max|diff| = 0.000e+00` at both low and mid stages.
   The multiply *count* is unchanged (4 × B·S·H·d/2) — **the win is rank 6→4, not FLOPs.**
   That matters for how you describe the fix: it is a layout change for the HTP tiler, not
   an arithmetic optimisation.
3. **The whole tensor is constant-foldable.** `rope()` is a pure function of
   `(pos, dim, theta)`; verified deterministic across calls. Once a stage's shape is pinned
   the tensor never needs to appear in the graph at all — which is the stronger version of
   the paper's advice.

**Also found:** `rope()` computes in `float64` (`modeling_embedding.py:14`) — `omega`,
the einsum, and `cos`/`sin` are all f64 before a closing `.float()`. An f64 subgraph
reaching the converter is its own problem; constant-folding removes it along with the rest.

**Also found:** `modeling_block.py:109` does in-place index assignment into a stacked
tensor (`concat_qkv_tokens[:, :, 0], concat_qkv_tokens[:, :, 1] = ...`), which lowers to
ScatterND. The surrounding `torch.stack(...)` / `.unbind(2)` is a pure round-trip. Unbind
q/k/v first and the scatter disappears.

---

## Trap #2 — causal mask constant — CONFIRMED, and worse than described

**The brief says** the default mask value "is too large for the fixed-point runtime".
**Measured:** it is not merely large, it is `-inf`, and it is *invisible in the source*.

Neodragon's masks are **boolean** — built with `==`, `>=`, `&` at
`pyramid_mmdit/__init__.py:316-324` — and handed straight to
`F.scaled_dot_product_attention`. No fill value is written anywhere in the Python. The
constant is injected by the lowering.

Measured (`audit_mask.py`), exporting exactly that pattern:

| opset | emitted | dtype | where |
|---|---|---|---|
| 17 | `-inf` | float32 | `Constant` feeding a `Where` |
| 20 | `-inf` | float32 | same |

The T5 side has its own instance, from `transformers/models/t5/modeling_t5.py:1040`:

```python
causal_mask = (1.0 - causal_mask) * torch.finfo(inputs_embeds.dtype).min   # -3.4028e+38
```

**Fix, verified.** An explicit additive mask with a finite fill. Measured leakage and error
against the SDPA reference:

| fill | max leaked attn weight | max\|out−ref\| |
|---|---|---|
| −10000 | 0.000e+00 | 5.4e-07 |
| −1000 | 0.000e+00 | 5.4e-07 |
| **−100** | **2.8e-42** | **5.4e-07** |
| −60 | 6.7e-25 | 5.4e-07 |

`-100` is ample. For context, DistilT5's measured pre-softmax logit range is
`[-32.89, 32.84]`, so `-100` sits 67 below the smallest real score.

Because both instances land as graph constants, one graph pass covers the whole pipeline —
implemented as `retarget_extreme_constants()` in `work/export/graph_fixes.py`, reusable for
the MMDiT graphs in Phase 5.

---

## Trap #3 — T5 residual / FFN adds — CONFIRMED, but the brief prescribes the wrong fix

**The brief says:** "T5's residual-add and feed-forward-add operations exceed FP16 range —
apply a scaling factor to residual connections."

The first half is true. The second half does not address the actual failure.

### The range is real

Measured (`audit_t5.py`, 8 showcase prompts, seq 128, fp32):

| block | after self-attn add | after FFN add |
|---|---|---|
| 0 | 154 | 810 |
| 1 | 808 | **134,761** |
| 8 | 164,602 | **214,408** |
| 11 | 218,912 | **329,510** |

Peak **329,510 = 5.03× fp16 max**. Overflow starts at block 1 of 12.

### PyTorch fp16 looks fine — and this is a trap in itself

| path | inf | nan | rel_err vs fp32 | cosine |
|---|---|---|---|---|
| `from_pretrained(fp16)` | 0 | 0 | **0.0026** | 0.999997 |

T5LayerNorm computes its variance in fp32 and renormalises, so in PyTorch the large residual
never reaches the output.

> **CORRECTION — this measurement is misleading, and the device proved it.**
> HTP executes float graphs in **fp16 with no fp32 upcast in the layer norm**. On an
> S25 Ultra the unscaled float build returned **74% NaN** with finite values saturating at
> **±65408** (an fp16 value near the 65504 max). PyTorch's fp32 variance computation is a
> rescue the device does not provide.
>
> An earlier draft of this document concluded "residual scaling buys fp16 headroom the model
> does not need." **That was wrong.** Residual scaling is exactly the required fix for the
> float path. See `device-results.md` for the measured A/B.

### The actual failure is A16 resolution

The deployment target is W8A16 **integer**, where one outlier fixes the per-tensor scale:

| block | peak | median \|·\| | peak/median | a16 LSB | **effective bits** |
|---|---|---|---|---|---|
| 0 | 810 | 3.83 | 211 | 0.025 | 7.3 |
| 1 | 134,761 | 6.24 | 21,598 | 4.11 | **0.6** |
| 2 | 162,458 | 7.50 | 21,657 | 4.96 | **0.6** |
| 8 | 214,408 | 22.78 | 9,412 | 6.54 | 1.8 |
| 11 | 329,510 | 38.58 | 8,541 | 10.06 | 1.9 |

A typical activation gets **0.6–2.3 bits** of a 16-bit quantiser.

**Residual scaling does not fix *this* part.** Scaling divides peak and median equally, so
peak/median — and therefore effective bits — is exactly invariant. Verified: with the
embedding *and* every branch divided by S (T5LayerNorm is RMS-style and scale-invariant, so
this is exact — `rel_err` 2.6e-06 at S=8), the peak drops to 41,189 and fits fp16, while the
ratio is unchanged.

So the two problems are separate and need separate fixes:

| problem | fix | status |
|---|---|---|
| fp16 **overflow** on the float/HTP path | residual scaling | **done, verified on device: NaN → 49 dB** |
| a16 **resolution** under W8A16 | per-channel / outlier migration | open |

Residual scaling is necessary and confirmed. It is just not *sufficient* — it does nothing
for the quantised path, which is still ahead of us.

### What does fix it

The outliers are **structural and channel-localised** (`audit_t5_outliers.py`, 32 prompts,
d_model 768):

| block | peak | top-1 channel | #ch > 1% of peak | per-tensor bits | per-channel bits | gain |
|---|---|---|---|---|---|---|
| 1 | 140,955 | **190** | 53 | 0.5 | 8.4 | **+7.9** |
| 5 | 164,527 | **190** | 52 | 1.3 | 9.3 | **+8.0** |
| 11 | 331,610 | **190** | 33 | 1.9 | 10.7 | **+8.8** |

Channel **190** is the top channel in every block from 1 to 11. The top-5 set
`{190, 589, 399, 127, 16}` is *identical* across blocks 1–10 — input-independent, so a
**static** scale handles it. Dropping the top-16 channels of the last block cuts the peak
17× (331,610 → 20,073).

> **RESOLVED — this does not apply to DistilT5.** Paper Table 9 deploys DistilT5 (and both
> CLIPs) in **FP16**, never W8A16. The effective-bits arithmetic above is correct but
> describes a configuration the reference design never enters. See `paper-notes.md`.
>
> It stays relevant as a **warning for the MMDiT**, which *is* W8A16 (300 calibration samples
> per stage, 22–29 dB targets) and also carries a residual stream. If the MMDiT shows the
> same structural outlier-channel pattern, expect per-tensor a16 to cost bits there too — and
> note that HTP offers per-channel *weight* quantisation
> (`--use_per_channel_quantization`) but per-tensor *activation* quantisation, so the
> fallback would be a SmoothQuant-style migration into the consuming weights.

---

## Trap #5 — AIMET gap — partially answered

The SDK at `qairt/2.49.0.260730` is complete for this work: `qnn-onnx-converter`,
`qairt-quantizer`, `qnn-model-lib-generator`, `qnn-context-binary-generator`,
`qairt-accuracy-debugger`, and **`lib/hexagon-v79/`** (our target) plus `hexagon-v81`.

The existing `convert_inpaint_unet.sh` already runs W8A16 with per-channel weights through
`qnn-onnx-converter` alone — no AIMET involved. Whether that path is sufficient for the
MMDiT stages, or whether AIMET's outlier handling is needed for the channel problem above,
is still open.

---

## Not yet audited

- **Trap #4 (per-stage calibration)** — not statically inspectable; it is a data question.
- **The MMDiT export itself.** `PyramidMMDiT.forward` takes `List[List[Tensor]]`, loops over
  stages in Python, and does boolean scatter-assignment
  (`__init__.py:299`, `text_ids[encoder_attention_mask == 0] = 0`). It is not exportable
  as-is; a single-stage forward has to be written. This is the real Phase 5 cost and none of
  today's work reduced it.
- **`PatchEmbed3D.pos_embed`** is a persistent buffer of `192×192×1536` fp32 = **226 MB**,
  cropped per stage at `modeling_embedding.py:285-333`. It must be pre-cropped per stage or
  it bloats every one of the three graphs.


---

# Traps found 2026-08-23 (session 6)

## 33. `--use_per_channel_quantization` is CONVOLUTION-ONLY. Fully-connected weights need `--use_per_row_quantization`.

The converter's own help says it outright:

```
--use_per_channel_quantization
    ... enable per-channel quantization for convolution-based op weights
--use_per_row_quantization
    ... enable rowwise quantization of Matmul and FullyConnected ops
--enable_per_row_quantized_bias
    ... rowwise quantization of bias for FullyConnected, when weights are per-row quantized
```

**Every convert script in this project passed only the first flag**, so:

| module | per-channel weights | per-tensor weights |
|---|---:|---:|
| VAE decoder (conv) | 36 | 0 |
| VAE encoder (conv) | 31 | 6 |
| **MMDiT** | **0** | **428** |
| **ContextAdapter** | **0** | **5** |

The conv modules were fine by accident; every fully-connected weight in the MMDiT and the
ContextAdapter was per-tensor 8-bit. On weights with outliers this is severe -- measured
on the ContextAdapter's five Linears, per-tensor vs per-row 8-bit:

| linear | max\|w\| | per-tensor | per-row |
|---|---:|---:|---:|
| layers.0.1 | 2.469 | 22.81 dB | 36.19 |
| layers.3.1 | **11.688** | **18.43** | 33.49 |
| layers.4.0 | 5.406 | **14.83** | 22.98 |

Check it in `net.json`: a per-row weight carries `axis_scale_offset` with one entry per
output row and dtype `0x0308` (signed 8-bit); a per-tensor one carries a single
`scale_offset`. **Note the dtype changes**, so a checker looking only for unsigned
`0x0408` will report zero of both.

The MMDiT convert script now passes it. Its host ablation predicted 43.14 dB assuming
per-channel weights it never actually got, so the real W8 ceiling on device was lower than
that all along -- this is a free candidate for the remaining stage-0 and stage-2 gaps.

## 34. Clamping a saturating activation's input is free accuracy

The ContextAdapter's GELUs each lost 13-33 dB. Comparing the device's GELU against an
exact GELU of *the device's own input* isolates the op from what it inherited:

    layer 0: 19.59 dB        layer 3: 47.98 dB

Layer 3's op is fine; layer 0's is not. The error breakdown says why -- **89% of layer 0's
elements (469,489 of 524,288) sit below -8, where GELU is exactly 0, and the device
returns ~3.6e-3 there**. GELU's tail is computed as a product with the input, so the
absolute error scales with |x| and the large negative outliers dominate. Layer 3 has only
11% of its mass in that region.

`GELU(-6)` underflows to exactly 0 in fp32, so clamping the input is **bit-exact**:

    clamp at  -4 :  29.94 dB   (breaks)
    clamp at  -5 :  69.38 dB
    clamp at  -6 :  inf dB, max|diff| 0.000e+00
    clamp at  -8 :  inf dB

and it shrank layer 0's GELU input range from [-31.95, 4.88] to [-6, 4.88]. Apply to any
saturating activation (GELU, SiLU, sigmoid, tanh) whose input has a long dead tail.

## 35. Some modules should not be quantised at all

The ContextAdapter took five conversions to reach 9.44 dB and one to reach **39.36 dB**:

| build | SNR |
|---|---:|
| W8A16, per-tensor weights | 2.62 dB |
| + per-row | 4.61 |
| + W16 weights | 5.13 |
| + clamped GELU | 6.80 |
| W16 + clamp + per-row | 9.44 |
| **FP16, no quantisation** | **39.36** |

The signal that should have stopped the quantisation attempts sooner was available before
any of them: a host simulation with per-row W8 weights and nothing else quantised reached
only **16.84 dB**, below the useful bar, so W8 was never viable for this module. It is a
5-layer MLP with weight outliers and **no normalisation layers**, so weight error compounds
multiplicatively with depth.

Paper Table 9 has **no ContextAdapter row at all** and deploys the whole text path -- CLIP
L, CLIP G, DistilT5 -- in FP16. FP16 here matches the reference deployment, costs 520 MB
against ~1.5 GB of headroom, and the module runs **once per generation**, not per AR unit.

> **Rule: before spending conversions on a module, simulate its quantisation on the host
> and check the ceiling.** If a faithful simulation cannot clear the target, no converter
> flag will. FP16 is a legitimate answer for small modules outside the inner loop.

## 36. PowerShell writes BOMs and CRLF; QNN list files are parsed byte-exactly

Twice in one session:

- a `--set_output_tensors` list written with `>` got a UTF-8 BOM on the first name and a
  `\r` on the last, so `qnn-net-run` silently dumped N-2 tensors -- losing the single most
  useful tensor in a stage-1 dump
- a rewritten `calib_list_host.txt` written with `Set-Content` got CRLF, so the last path
  on every line did not exist and the converter aborted with "300 calibration files missing"

Write any list file consumed by the QNN toolchain from Bash, or with an explicit
ASCII/no-newline writer. `Set-Content -Encoding ascii` is **not** enough -- it still writes
CRLF.


## 37. A float model library above 2 GB of weights will not link

`qnn-model-lib-generator` embeds the weight blob in the model `.so`, and the x86-64 build
uses PC32 relocations, which cannot address past 2 GB. CLIP G is 694.7 M params -- 2.78 GB
at fp32 -- and the link died with a wall of:

```
relocation truncated to fit: R_X86_64_PC32 against `.tm_clone_table'
clang++: error: linker command failed with exit code 1
make: *** [Makefile.linux-x86_64:109: libs/x86_64-linux-clang/libclipg.so] Error 1
```

It is NOT a disk problem (56 GB free at the time) and NOT trap #30. The fix is
`--float_bitwidth 16` on the converter: the blob halves to 1.39 GB, links, and fp16 is
what HTP executes for a float graph anyway (trap #3), so nothing is lost.

Applies to any FP16-deployed module above ~500 M params. DistilT5 (130 M) and CLIP L
(123 M) are far below it; CLIP G is the first to cross.

> **Also: the convert scripts' `| tail -3` on lib-gen hides this.** The error above only
> appeared after re-running lib-gen by hand with `tail -40`. Keep the raw log
> (`tee "$OUT/converter_raw.log"`, added this session) or a link failure looks like a
> bare `RuntimeError: Failed to compile model library`.
