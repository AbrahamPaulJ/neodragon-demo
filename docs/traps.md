# Traps — all 41, with the measurement that established each

Split out of `CLAUDE.md` 2026-08-23. CLAUDE.md is loaded into context every session and
was 584 lines, most of it this list; the traps are needed when you are about to do the
specific thing they warn about, not on every turn.

**Read this file when:** converting a model, touching layout or quantisation, measuring
latency on device, or debugging something that "runs but is wrong". CLAUDE.md carries a
short index of the load-bearing ones.

## Traps — status

Traps 1–3 are **audited, fixed and verified on device**. Details and numbers in
`docs/trap-audit.md`; the paper's own wording is in `docs/paper-notes.md`.

1. **RoPE 6D tiling** — CONFIRMED at `modeling_embedding.py:11` + `modeling_block.py:69`.
   Rank-4 replacement verified bit-exact. Also fully constant-foldable. Not yet applied
   (Phase 5 hasn't started). The win is **rank 6→4, not FLOPs** — multiply count is identical
2. **Causal mask constant** — CONFIRMED. It is literal `-inf` from the SDPA lowering and
   `-3.4e38` from `transformers/models/t5/modeling_t5.py:1040`; **invisible in the source**.
   Fixed with `-100` (leakage 2.8e-42) via `work/export/graph_fixes.py`, reusable for the MMDiT
3. **T5 residual/FFN adds** — CONFIRMED, 5.03× fp16 max. **HTP runs float graphs in fp16 with
   no fp32 upcast in the layer norm**, so the unscaled build returned 74% NaN on device.
   Residual scaling fixes it (NaN → 49 dB). Fold the scale into weights, not runtime multiplies,
   and **scale `variance_epsilon` by 1/S²** or scale-invariance silently breaks
4. **Per-stage calibration sets** — CONFIRMED twice. Calibration must use **real pipeline
   latents** (synthetic `N(0,1)` was ~3× too narrow), and for a graph with **explicit state
   inputs the state must be real too** — zeros would have set 9 of 10 input ranges to 0.
   `work/device/make_stream_calib.py` replays each prompt in temporal order to capture it
5. **AIMET gap** — mostly moot. `qnn-onnx-converter` does W8A16 PTQ unaided and now reaches
   **34.27 dB against a 35 dB target**. `--quantization_overrides` (AIMET encodings) and
   `--algorithms` remain the route to the last 0.73 dB if it is ever worth it
6. **No official deployment code exists** — unchanged, and confirmed by reading the paper

### New traps, learned the hard way

7. **`--preserve_io layout` is mandatory for any conv model.** Without it the converter
   silently makes 5-D inputs channel-last while leaving outputs NCDHW. Measured **−0.75 dB**,
   with outputs still varying per input so nothing looks broken. Phase 1 escaped only because
   DistilT5's inputs are rank-2
8. **`qnn-net-run` parses raw files as float32 by default.** Integer-input *float* graphs need
   `--use_native_input_files`; **quantized** graphs (`UFIXED_POINT_16` I/O) must **omit** it.
   Getting it wrong yields identical, plausible-looking outputs from every input — always
   assert two different inputs give different outputs
9. **Don't hand-convert ONNX to fp16.** Measured 3.7× *slower* and 10 dB worse. The converter
   normalizes precision itself; let it own that
10. **The DSP throttles.** Clock swung 776 → 596 MHz between runs, inflating cycle counts on
    untouched ops enough to reverse an A/B verdict. Always interleave A/B in one thermal
    session and compare **min-of-N**, never averages
11. **Never synthesise a time or batch axis by reshaping the channel axis.** ← *the big one*.
    `TGrow` did `x.reshape(-1, C, H, W)` to fold time into batch. Free in NCHW (C is adjacent
    to N), a full scatter in NHWC (C is innermost) — and HTP runs convolution channel-last, so
    the converter wraps every such reshape in a transpose pair on your largest tensors. Cost
    measured: **779.80 MB of layout traffic per inference and a 53.5× latency penalty.**
    Carry state explicitly instead. Same lesson as trap #1's RoPE rank reduction
12. **`analyze_net.py` before any device time.** `work/device/analyze_net.py` reads the
    converter's `*_net.json` and reports transpose bytes ÷ compute bytes. **1.16 was
    catastrophic, 0.28 (DistilT5) is livable, 0.09 is clean.** Free, static, no device needed.
    Also note the QNN dtype encoding: `UFIXED_16` is `0x0416` and the low byte spells the
    width in *decimal* — reading it as hex silently doubles every 16-bit tensor.
    **Convert bytes to milliseconds before panicking:** the decoder A/B (63.00 MB →
    114.04 ms vs 7.95 MB → 100.18 ms) puts layout traffic at **~0.252 ms per MB** on this
    chip. The ratio is a smell test calibrated on a conv net; ms/MB is the arithmetic.
    And when the ratio *is* bad, `work/device/trace_transposes.py` says **why** — it groups
    every Transpose by producer/consumer op type and ranks by bytes.
    **The ratio also rises whenever you DELETE compute, which is usually the goal.** The
    session-7 stage-2 fix took the ratio from 0.06 to 0.11 while absolute Transpose bytes
    stayed *identical* (392.53 MB both) — the denominator had shrunk 42% because a MatMul
    per block was removed. Read the absolute Transpose bytes, not the ratio, any time the
    graph's arithmetic has changed. And when comparing two builds, check the activation
    width first: a `--layout` build is fp32 and a shipping build is A16, so their byte
    totals differ 2× for reasons that have nothing to do with the change
13. **Spill volume is not a proxy for anything.** spill_bytes nearly tripled (215 → 482 MB)
    while latency fell 53×. `vtcm_mb: 8` never needed tuning; what mattered was *what* was
    spilling
14. **adb: the phone can attach twice** (USB + wireless on tcp:5555), which makes bare adb
    fail "more than one device/emulator" — prefer the USB serial. And **adb writes progress
    to stderr on success**, so PowerShell `$ErrorActionPreference='Stop'` turns a completed
    push into a fatal error. Check `$LASTEXITCODE`. Both handled in `work/device/push_stream.ps1`

15. **Check what the deployment path actually calls before believing a module is hard.**
    The VAE encoder's `deque` feature cache was recorded as "not exportable as written" and
    blocked the phase for a week. `vae.encode()` takes `temporal_chunk=False` by default and
    the AR t2v flow hands it a single frame, so that branch is **never executed**. One grep
    of the call site would have said so
16. **Constant padding against a size-1 axis is free work you can delete.** At T=1 a causal
    time-3 conv pads two planes of *zeros*, so `out = w[2]·x` exactly — the other two terms
    are multiplications by zero. Collapsing it took the encoder from 1071.2 to **363.33
    GMAC** and removed every 5-D tensor, for **7.2× under the paper's latency**. Same family
    as traps #1 and #11: the wins keep coming from removing structure, not tuning the backend
17. **The converter fuses GroupNorm; PyTorch never emits the native op.** `torch.onnx.export`
    lowers `nn.GroupNorm` to Reshape → InstanceNormalization → Reshape → Mul → Add at opsets
    17, 18 *and* 20 — and that first reshape folds channels into a flattened axis, which is
    trap #11's shape. It is safe anyway: `OptimizeGroupNormTranslation` matches the pattern
    and emits a native `GroupNorm`, which `htp.json` supports. Verified — all 22 arrived
    fused, ratio 0.05. **Phase 3's SSD1B UNet can rely on this**
18. **Pick the SNR metric before running the comparison.** The encoder's output concatenates
    `mean` (std 1.85) with `logvar` (≈ −30, near-constant) on the channel axis. Whole-tensor
    SNR reads **46.24 dB** where the number that governs quality is **41.60 dB**, because the
    trivially-reproduced half dominates the norm. Any concatenated output with mixed dynamic
    ranges will flatter itself by default

19. **Never ask a tensor for a dimension you already know.** `t.view(t.shape[0], ...)`
    inside a traced graph emits `Shape`+`Gather` that constant folding **cannot** remove,
    because it descends from an input. Passing the known integers instead took the MMDiT
    from `Shape`×186 to ×55. The same rule killed `ConstantOfShape`×201 and `Where`×196:
    they were `AdaLayerNorm.forward_with_pad` materialising `zeros_like(x).repeat(1,1,6)`
    and scattering into it once per block, which at `num_stages == 1` is a plain broadcast
20. **Hoisting positional embeddings to the host is what makes padding exact.** Once the
    host chooses every slot's position id, a short sequence can be front-padded while its
    real tokens keep the ids they would have had — so every relative position, including
    text-to-image (text sits at id 0 and is *not* shifted), is unchanged. Padding the
    latents naively instead would have shifted every image token and silently altered every
    score. Measured 123–128 dB, i.e. the fp32 floor, across all 18 MMDiT shapes
21. **`torch.onnx.export` writes one external-data file per tensor** once the model passes
    2 GB — 659 of them for the MMDiT, dumped next to the `.onnx`, and the converter then
    reads them over the 9p `/mnt/c` mount where per-file overhead dominates. Export into a
    dedicated directory, re-save with `all_tensors_to_one_file=True`, and stage the model on
    WSL's ext4 before converting
22. **Pick the deploy-SNR target per stage for the MMDiT.** It is the only module where
    paper Table 9 gives three different numbers: 29 / 22 / 24 dB for stages 0 / 1 / 2 — the
    worst in the pipeline, and the place trap #3's residual-stream argument actually applies
    (DistilT5 escaped it by shipping FP16; the MMDiT is W8A16 with 18 residual blocks)

23. **Rank is not an escape from layout rules — `NONTRIVIAL` is.** QNN has a canonical
    layout for rank 3 (NCF/NFC) exactly as it does for rank 4 (NCHW/NHWC). In the MMDiT,
    MatMul and Softmax ran NCF while the mask-add `Eltwise_Binary` ran NFC, so **every
    block permuted its [24,408,408] score matrix twice** — 70% of all layout traffic, and
    invisible in the dims because the shape is unchanged. The converter names it for you:
    `_MatMul_output_0_nfc` then `_Add_5_output_0_ncf`. Dropping rank 4 → 3 only renamed the
    problem (1.35 → 1.10). The fix is **`--input_layout <name> NONTRIVIAL`** for every
    input without image semantics: 1.10 → **0.25**, 191 Transpose nodes → 85.
    NONTRIVIAL is also the *safe* layout — unlike NHWC it keeps bytes as authored, so
    `permute_order_to_src` stays identity and there is nothing for the host to get wrong.
    Pin `--preserve_io layout` only on tensors that genuinely feed convolution
24. **Iterate layout with a conversion that has no `--input_list` at all.** Axis tracking
    runs in IrOptimizer passes *before* the quantiser, so a float build makes identical
    layout decisions. `convert_mmdit_w8a16.sh <stage> --layout` does this, and it was
    **validated as a faithful proxy**: the layout-only build reproduced the quantised
    build's structure exactly — same ratio, same 191 nodes, same groups, only the bytes
    doubled (fp32 vs A16). It also drops the calibration set as a dependency
25. **QNN's MaskedSoftmax is not supported on HTP.** `--apply_masked_softmax` would fuse
    MatMul → mask-add → Softmax and remove trap #23's pair by construction, but `htp.json`
    lists only `Softmax` and `LogSoftmax`. Check the backend XML/JSON before designing
    around any converter optimisation flag
26. **`torch.stack(..., dim=-1)` is an element interleave, and it was 28.7% of the
    whole MMDiT.** 36 Concat ops, 108.6M cycles, 74.9 ms — while 54 concats of *twice
    the volume* on axis 0 cost **exactly zero**, because QNN folds an outermost-axis
    concat into its producers' allocation. The cost is **granularity, not bytes**, so
    `analyze_net.py` cannot see it and Phase 4's 0.252 ms/MB constant does not apply.
    Source: interleaved RoPE reassembling its rotated pairs. It never needs to —
    `<q',k'> = <q'_even,k'_even> + <q'_odd,k'_odd>`, so contract the two halves
    separately and add. Identical FLOPs. Counting the stride-2 slices and the rank
    bookkeeping around it, **RoPE was 43.3% of the graph, 6.6× all 255 FullyConnected
    layers combined.** Same family as #1/#11/#16/#23 — the win is removing structure
27. **QNN quantises a MatMul's *second operand* as a weight**, so attention **K and V
    land at 8 bits** in a W8A16 graph — pure activations at half the intended precision,
    in all 18 blocks. Only `work/device/matmul_bits.py` can see it (operand widths, with
    static tensors starred). `htp.json` lists `u16 × u16 → u16`, so it is a converter
    default, not a hardware limit: the hidden `--use_dynamic_16_bit_weights`
    (`help=argparse.SUPPRESS`, `qnn_quantizer.py:156`) plus
    `--restrict_quantization_steps "-0x8000 0x7F7F"` removes it — the pair must arrive
    as ONE argv entry (`validation_utils.two_hex:112` splits it), so build converter
    flags as a **bash array**. **BUT on the MMDiT this bought +0.01 dB and cost +18.6%
    latency** (`DYN16=1` to re-enable). Know the mechanism; do not assume it is your
    accuracy bug. A confirmed mechanism is not a confirmed cause
28b. **`--debug` does not scale on HTP.** All 2374 intermediates client-readable makes
    finalize fail outright (`error = 1002`) after 1:55. 314 tensors finalize in 86 s, 35
    in 71 s. Slice the dump with `--set_output_tensors`; `block_tensors.py` picks it
29. **The converter's tensor names are the ONNX names with every non-identifier
    character replaced by `_`** — `/norm1/norm.12/Constant_1_output_0` →
    `_norm1_norm_12_Constant_1_output_0`. So a device dump joins to an ONNX reference
    **mechanically, for every tensor**, which is what `layer_snr.py` exploits. It also
    makes **onnxruntime a far better reference than torch** here: proto load 0.002 s
    with `load_external_data=False`, session 11 s, **inference 1.5 s**. Append the
    wanted tensors to `graph.output` and re-save **into the same directory** so the
    external-data path still resolves
30. **`qnn-model-lib-generator` builds in a temp dir under the CWD**, exploding the
    weight blob into ~900 `.raw` files and `objcopy`ing each. On `/mnt/c` every one
    crosses the 9p mount; peak scratch **4.2 GB**. `cd` to ext4 first
31. **A tensor feeding a `chunk`/`split` gets ONE encoding, sized by its largest part.**
    ← *this is the Phase 5 accuracy bug*. The six AdaLN modulation chunks span 1.68 →
    18.41, so the tensor reads a healthy 29.35 dB while its small slices land at
    **11.97–16.93 dB** — and those are the `shift`/`scale` that multiply straight into a
    55 dB image stream, taking it to 16.51 dB in block 0. Invisible in the parent
    tensor's own encoding quality. **Look for it wherever one Linear produces several
    semantically different outputs**; the fix is separate Linear ops, identical FLOPs
32. **Conv outputs read as garbage in a name-matched layerwise diff, and it is an
    artefact.** `_proj_Conv_output_0` scores −2.99 dB against ONNX while the next
    `Reshape` scores 57 dB — the device writes conv output channel-last, ONNX is
    channel-first. Read the first layout-neutral tensor after a conv, not the conv

28. **A plausible mechanism is not a cause — and on a quantised graph you can usually
    settle it for free.** Four MMDiT accuracy hypotheses died in one session; three had
    convincing static evidence behind them. The two cheap ones were killed by *reading
    the converter's own encodings out of `net.json`* (the −100 mask inflates the score
    range only 1.68×, i.e. 0.75 bits) and the expensive one by a 42-minute conversion
    that moved SNR by 0.01 dB. **Read the encodings before building anything**, and when
    a mechanism survives that, still expect it to be innocent

### Traps from shipping it as an app (session 6, 2026-08-23)

33. **An app cannot exec a helper binary onto the DSP.** The desktop harness drives every
    graph by exec'ing `qnn-net-run`, so the first APK shipped that binary inside itself.
    It runs and then dies in `deviceCreate` with `loadRemoteSymbols failed with err 4000`.
    Bisected on device, the **byte-identical** binary (same md5), same `LD_LIBRARY_PATH`,
    same skel dir, same CWD: works from `/data/local/tmp` (`shell_data_file`), fails from
    `/data/app` (`apk_data_file`). No environment change fixes it — `dlopen` the backend
    **into the app process** instead (`work/android/app/src/main/cpp/ndqnn.cpp`).
    Generalisation: bisect an exec-location failure by *moving the binary*, not by editing
    the environment
34. **Declare `libcdsprpc.so` with `<uses-native-library>`.** It is a VENDOR library listed
    in `/vendor/etc/public.libraries.txt`, but from targetSdk 31 an app must name it in the
    manifest or the linker refuses it and `libQnnHtpV79Stub.so` fails to load. The real
    message is only visible if you **forward QNN's log callback to logcat** — by default it
    is discarded and all that escapes is `Failed to load skel, error: 4000`
35. **`mmap` context binaries; never read them into the heap.** A heap copy doubles peak
    footprint for the duration of the load (1.4 GB for clipg and ssd1bunet). On an 11 GB
    phone that drew lmkd: `MemFree` 110 MB, eight background apps culled in one sweep, the
    Activity killed mid-run. `mmap` + `MADV_SEQUENTIAL`, plus releasing each graph when the
    pipeline is done with it, took peak RSS **1.57 GB → 0.40 GB** at identical latency
36. **Precompute deterministic bookkeeping on the host; port only arithmetic.** The AR
    loop's envelope, past-condition slices, temporal RoPE ids and token layout depend only
    on `(unit_index, stage)` — never on prompt, seed or latent. All 18 were computed once
    against the reference (`export_video_structure.py`, 4.2 MB asset) rather than
    reimplementing four bookkeeping functions in Kotlin, each a chance at a silent
    off-by-one whose output looks plausible either way. The masks are NOT shipped (214 MB);
    they are rebuilt on device from `order` + `image_valid` + the text mask, which is the
    only prompt-dependent part
37. **The video path uses DIFFERENT text encoders from the first-frame path.**
    `TextEncoderBundle` loads `text_encoder`/`text_encoder_2` (both
    `CLIPTextModelWithProjection`) and keeps only the **pooled** `text_embeds`. The
    shipping `clipl` graph came from `ssd_1b_text_encoder`, a plain `CLIPTextModel` with no
    projection head, so it cannot supply that vector — hence `cliplp`. The checkpoints are
    the same weights at different precision (bf16 vs fp16, max abs diff 0.027 over all 517
    shared tensors), so CLIP G's existing graph serves both paths
38. **The streaming decoder packs frames FRAME-major.** `video` is `[1,24,320,512]` and
    frame *f* owns channels `[3f, 3f+3)`. Channel-major reading yields three plausible
    greyscale videos. `frames_to_trim = 7` applies to the WHOLE video: 7 latent frames × 8
    = 56 emitted, minus 7, is 49
39. **Feed the decoder's state outputs straight back as its state inputs.** The 9 MemBlock
    states are channel-last in the shipping build, which is why the desktop path needed
    `permute_states_nhwc.py`. Wiring `nstate_i → state_i` without ever interpreting the
    contents makes the layout question vanish, and the sizes come from the binary's own
    metadata instead of being hardcoded

40. **A fix that trades a LINEAR cost for a QUADRATIC one silently inverts at longer
    sequences.** ← *this is the Phase 5 stage-2 latency bug.* Session 2's RoPE rewrite
    swapped one full-width attention MatMul for two half-width ones plus an add, to dodge
    the `torch.stack(dim=-1)` interleave of trap #26. Measured **−30.6% on stage 0** (408
    tokens) — and the same code costs **29% of stage 2** (1728 tokens), because the
    interleave it removes is linear in sequence length while the extra score traffic it
    adds is quadratic. At 1728 tokens the score matrix is 143.3 MB and the graph writes it
    **five times per block — 12.9 GB, 74% of the whole graph.** Fix: concatenate the two
    rotated halves (a 32-wide contiguous copy, not an element interleave) and contract
    ONCE, exact by `<cat(a,b),cat(c,d)> = <a,c>+<b,d>`. **Re-price every shape-dependent
    optimisation at the largest shape you ship, not the one you profiled.**
    `docs/phase5-stage2-bandwidth.md`
41. **The MMDiT is bandwidth-bound, and op count is a dead lever.** The three stages have
    the *same* node count (1719/1719/1730) and latencies of 169.2/375.8/1851 ms, so op
    count cannot explain an 11× spread; op **output bytes** (1.00/1.96/9.71×) track
    latency (1.00/2.22/10.94×). The control was already run: the AdaLN fix deleted **655
    ops (27.6%)** from stage 0 and bought **1.7%**. Validated per-op against the stage-0
    detailed profile at **0.14 cycles/byte** overall, with MatMul 0.94×, Eltwise_Binary
    0.82× and Softmax 1.30× of that bulk rate — the three types that are 82.3% of stage 2.
    Tools: `work/device/stage_census.py`, `work/device/bytes_model.py`


### Traps from session 7 (2026-08-23)

42. **Per-op cycles/byte vary 20x, so an aggregate ms/MB cannot price an op-MIX change.**
    The stage-2 fused-score rewrite deleted 28% of the graph's output bytes and only 4.4%
    of its cycles, because the ops it removed were the CHEAPEST bytes in the graph. On the
    same `[24,1728,1728]` tensor: Eltwise_Binary **0.0103** cycles/byte, MatMul **0.0187**,
    Softmax **0.221**. The bytes model held ACROSS stages only because the op mix was
    identical there. `bytes_model.py` had already shown a 4x spread between op types and
    that was ignored in favour of the aggregate. **Use `prof_by_shape.py` before projecting
    any latency from a byte count.**
43. **Instrument LOAD separately from EXECUTE before comparing against published
    latency.** The first-frame path was recorded as 7.2 s against the paper's 1.53 s and
    called "the worst ratio in the pipeline" for a whole session. Both halves were wrong:
    the 7.2 s was a COLD run (binaries are `mmap`'d; run one page-faults ~2.9 GB in from
    storage, later runs hit the page cache), and it was wall clock compared against a
    table that measures inference only. **Compute alone was 1.18x the paper — essentially
    at parity.** Keeping CLIP G resident (1897 ms to load, 40 ms to run) and preloading in
    the background took it to 2.1-2.5 s warm at *lower* RSS.
44. **A deploy-SNR shortfall is not "not worth shipping" until you price the
    alternative.** QuickSRNet lands at **28.06 dB** against the paper's 48 dB — alarming
    until you measure what it replaces: bilinear 2x is **23.22 dB** from the same fp32
    reference. The quantisation error is smaller than the artefact the module exists to
    remove. Ship it.
45. **Verify background work by its OUTPUT, not by a process query.** `pgrep -f ab_stage`
    matched its own command line and reported a dead run as healthy for twenty minutes;
    the real evidence was a 0-byte log and a scratch dir untouched for three hours.
    `setsid` inside `adb shell` does not reliably survive the connection either — run
    device work in the foreground over USB and confirm log files appear.
46. **Compose `Modifier.size()` is clamped by the parent's constraints.** A zoomed image
    inside a fixed-aspect Box got squashed back to the window, silently destroying the
    aspect ratio. Two attempts failed this way (`graphicsLayer` scaling of a
    `ContentScale.Fit` image, then `size()`). The fix is a `Canvas` `drawImage` with an
    explicit destination rect evaluating the SAME expressions the inverse-mapping code
    uses, so preview and result cannot diverge.
47. **MediaCodec with a Surface input timestamps frames by WHEN THEY WERE SUBMITTED.**
    `KEY_FRAME_RATE` only hints at bitrate allocation. Drawing 49 bitmaps as fast as the
    loop runs produced a 62 fps file from a 24 fps request. Override
    `BufferInfo.presentationTimeUs` at the muxer — the container's timestamps are what a
    player obeys.

48. **Never run a multi-minute job in Compose's `rememberCoroutineScope()`.** That scope
    is tied to the COMPOSITION, so Android cancels it when the Activity is backgrounded or
    recreated. The 8 GB model download died mid-file every time the user looked away, and
    only appeared to "work" on restart because HTTP Range resumed the partial. Anything
    that must outlive the screen needs a **foreground service** (or WorkManager); Android
    also throttles network for non-foreground processes, so surviving cancellation alone
    is not enough. Publish progress through a process-wide object, not by binding -- the
    UI may not exist while the work runs and must be able to attach to it later.
