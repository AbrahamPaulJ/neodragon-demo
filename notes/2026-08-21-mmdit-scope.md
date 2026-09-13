# Session 3, part 2 — 2026-08-21 — Phase 5 scoped and rewritten

Same session as `2026-08-21-vae-encoder.md`, after Phase 4b closed. The remaining structural
risk was Phase 5, and CLAUDE.md's entry for it was three claims deep in assumption. All three
turned out to be wrong, and none of them needed a device or a GPU to disprove.

## The three corrections

**1. `num_stages` is always 1.** The `List[List[Tensor]]` signature and the Python stage loop
looked like the problem. But `_generate_one_unit` loops the pyramid *outside* the model —
`dit(sample=[latent_model_input])`, an outer list of length one, every single call. So the
stage loop, the `[i_p::num_stages]` strides, the `stack`+`rearrange` round trip and the
`__init__.py:299` boolean scatter all degenerate to identities.

**2. It exports as written.** `work/audit/mmdit_trace_probe.py` built the real 1512.2 M-param
DiT, ran it, and called `torch.onnx.export` on the unmodified module. It succeeded. What the
module note should have said is not "it cannot be exported" but "look at *what* it exports":
`ConstantOfShape`×201, `Where`×196, `Shape`×186, `Gather`×184, `Equal`×179, `Expand`×259 —
`merge_input`'s mask construction and `AdaLayerNorm.forward_with_pad`, traced into the graph
because they descend from the `encoder_attention_mask` **input** and so survive constant
folding.

**3. "3 static graphs" is 18 shapes.** `_prepare_past_condition_latents` grows the clean
history each unit, so every `(unit, stage)` pair has its own sequence length.
`work/audit/mmdit_shapes.py` enumerates them from pure arithmetic — 6 per stage, 18 total.
They collapse into 3 graphs padded to the per-stage maximum, which is what the paper's
`[7×…]` rows are.

The enumeration was later **validated against reality**: a captured reference run produced
exactly the 18 predicted shapes, in the predicted order.

## The rewrite

`work/export/export_mmdit_stage.py`. Principle: **patch the plumbing, keep the math.** Every
weight-bearing module is the stock object. Four replacements, each verified:

- `merge_input` → host-side `stage_conditioning()`; `attn_mask`, `rope_cos`, `rope_sin`
  become ordinary graph inputs
- `PatchEmbed3D.forward` → precomputed `pos_embed` crops as buffers; the 226 MB table never
  ships
- `AdaLayerNorm*.forward` → the plain broadcast path
- `apply_rope` → **rank 4** (trap #1), `cos`/`sin` at `[1, S, 1, 32]`

Result: **bit-exact at all 18 shapes.** `max|diff| = 0.000e+00`, `SNR = inf`. Not "close" —
identical. And `ConstantOfShape` 201→0, `Where` 196→0, `Equal` 179→0, `Expand` 259→0,
`Shape` 186→55.

**This is the state before the transpose fight below, which changed two of those choices.**
The shipping form drops the attention to **rank 3** and ships `cos`/`sin` at `[S, 1, 32]`.
Bit-exactness goes with it: reordering the attention changes SDPA's accumulation blocking,
so the shipping rewrite scores **124–128 dB** rather than `inf`. That is the fp32 floor
(~1e-5 on O(1) tensors) and sits ~95 dB below the 29 dB quantisation target, but it is worth
stating plainly rather than leaving the `inf` above to imply more than it now does.

## The padding insight

This is the part worth remembering. Front-padding a short unit up to the envelope looks
trivial until you notice that prepending frames shifts every image token's position id —
while the text tokens sit at id 0 and are *not* shifted. Every text-to-image relative
position would change, and with it every score. Naive padding is silently wrong.

It works here **because RoPE was hoisted to the host**. The host now chooses the position id
of every slot, so real tokens keep the ids they would have had at their true shape and only
padded slots get filler. The mask hides them, and a fully-masked query row is finite rather
than NaN precisely because `MASK_FILL` is −100 rather than −inf (trap #2, from Phase 1).

Measured across all 18 shapes: 123–128 dB, i.e. the fp32 accumulation floor. `e^-100 ≈ 4e-44`
is zero in fp32, so the padded tokens contribute literally nothing.

Two structural wins from one decision. Hoisting the preamble was done to clean the op
histogram; it turned out to be the thing that makes the 3-graph design possible at all.

## The transpose fight

The rewrite exported cleanly, so I converted it. `analyze_net.py`: **ratio 1.35** — worse
than the 1.16 that cost Phase 4 a factor of 53. Three static iterations followed, no device
time and no calibration data:

| build | Transpose nodes | ratio |
|---|---|---|
| rank-4 attention, all I/O pinned | 337 | 1.35 |
| rank-3 attention, all I/O pinned | 191 | 1.10 |
| rank-3 + `--input_layout NONTRIVIAL` | **85** | **0.25** |

`analyze_net.py` told me how much but not where, so I wrote `trace_transposes.py`: group
every Transpose by `(perm, shapes, producer type, consumer types)`, rank by bytes. It said
immediately that **70% of all layout traffic was two permutations of the [24,408,408]
attention score matrix per block** — and that the in and out shapes were *identical*, so it
was a pure axis-format change that no shape inspection would ever reveal.

The converter names its own tensors after the decision: `_MatMul_output_0_nfc` then
`_Add_5_output_0_ncf`. MatMul and Softmax run NCF, the mask-add `Eltwise_Binary` runs NFC,
and every block paid the round trip.

**My first fix was wrong in an instructive way.** I dropped the attention from rank 4 to
rank 3, reasoning that rank-4 tensors get read as images. That helped (1.35 → 1.10) but did
not solve it, because QNN has a canonical layout for rank 3 too. Rank is not an escape
hatch. What worked was telling the converter the tensors have **no layout semantics at
all** — `--input_layout <name> NONTRIVIAL` for everything that is not image-like, with
`--preserve_io layout` narrowed to just the 5-D latents that really do feed a Conv2d.

NONTRIVIAL is also the safe option, which matters after Phase 4. The decoder needed NHWC
states and therefore a host-side `permute_states_nhwc.py`, with all the trap-#8 exposure
that implies. NONTRIVIAL keeps bytes as authored: every input and the output came back with
`permute_order_to_src = [0,1,2,...]`, identity, nothing for the host to get wrong.

### Two tools, and a number

- **`work/device/trace_transposes.py`** — the "why" to `analyze_net.py`'s "how much".
- **`convert_mmdit_w8a16.sh <stage> --layout`** — conversion with **no `--input_list`**.
  Axis tracking is an IrOptimizer pass that runs before the quantiser, so a float build
  makes identical layout decisions in a fraction of the memory. I did not take that on
  faith: the layout-only run with the *old* pinned flags reproduced the quantised build
  exactly — same ratio 1.10, same 191 nodes, same groups, bytes merely doubled (fp32 vs
  A16). Control first, then experiment.
- **~0.252 ms per MB of layout traffic on this chip**, from the Phase 4 decoder A/B
  (63.00 MB → 114.04 ms vs 7.95 MB → 100.18 ms). Worth having, because it converts the
  ratio smell-test into arithmetic: the three builds are ~63 / ~51 / ~23 ms of layout
  traffic against a 104.7 ms target. Real, worth fixing — but never the 53× that
  "1.10 is basically 1.16" implied. The ratio heuristic was calibrated on a conv net and
  should not be pattern-matched onto a transformer.

Ruled out cheaply: QNN's `--apply_masked_softmax` would fuse MatMul → mask-add → Softmax
and delete the pair by construction, but `htp.json` lists only `Softmax` and `LogSoftmax`.
Not supported on this backend.

## Things that bit

- **Git Bash mangles `wsl` paths too.** `wsl -d Ubuntu -e bash /mnt/c/...` run through the
  Bash tool became `C:/Program Files/Git/mnt/c/...` and **exited 0** having done nothing.
  CLAUDE.md already said to run adb from PowerShell; it now says the same for `wsl`.
- **`torch.onnx.export` scattered 659 external-data files** into `work/onnx/` — 4.1 GB of
  loose sidecars next to the real artefacts. Export into a dedicated directory.
- **The DiT takes 1536-dim context, not 4096.** `config.joint_attention_dim = 4096` is
  residue; the pipeline runs `prompt_embeds = context_adapter(prompt_embeds)` first, and the
  adapter maps 4096 → 1536. The first probe crashed in `norm1_context`'s LayerNorm on this.
- **1512.2 M params, not 3.1 GB of them.** CLAUDE.md's "3.1 GB" is the bf16 checkpoint. In
  fp32 that is 6.05 GB, and it is the reason the next step is a memory question.

## Where it stands

The rewrite is done and proven. The conversion is not, and the obstacle is **host RAM**:
5.64 GiB fp32 ONNX against 15.6 GB of machine with WSL capped at 11 GB. `.wslconfig` was
already tuned for exactly this during the LocalDream port, so there is no easy headroom.
`work/qnn/convert_mmdit_w8a16.sh 0 --probe` runs a single-sample conversion under
`/usr/bin/time -v` to measure peak RSS before anyone commits to a 300-sample run.
