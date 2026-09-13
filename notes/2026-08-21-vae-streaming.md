# 2026-08-21 — the 24× VAE gap: diagnosed, fixed, measured

Session 2. Started from a cold re-read a week after session 1 (2026-08-14). Ended with
Phase 4 complete and the decoder **60.9× faster at higher accuracy**.

## Why the project looked stuck (it wasn't)

Every file in the project was written on 2026-08-14 between 16:39 and 22:42 — one ~6-hour
sitting. The last two actions were writing `docs/phase4-vae-decoder.md` (22:41, ending with a
newly-discovered open question) and marking `CLAUDE.md` "on hold" (22:42). The project
stopped at the end of a long day, not at a wall.

What made it *read* as stuck: `CLAUDE.md` contradicted itself. A section headed **"The one
open blocker: compute"** said the 6 GB VRAM gap "is the reason this is paused", while 40
lines below, Open Decisions said *"Resolved: compute path — the RTX 3050 runs the reference
pipeline fine."* The compute problem was solved during that same session and the header was
never rewritten. **Fixed.** Lesson: when a session resolves a blocker, rewrite the section
that names it, not just the changelog.

## The diagnosis (static, no device time)

Two suspects eliminated first:

- **`soc_model: 69` is correct.** `QNN_SOC_MODEL_SM8750 = 69` in
  `../LocalDream/qairt/2.49.0.260730/include/QNN/QnnTypes.h:1872`. The "unverified guess"
  note from session 1 is resolved. (`converters/common/backend_aware_configs/htp_v2.json`
  also maps `SM8750 -> v79`.)
- **No float fallback.** All 223 tensors in the quantised graph are UFIXED_16 / SFIXED_8 /
  SFIXED_32.

Then the actual finding, from the converter's own `vaedec1q_net.json` — an artefact that had
been sitting in `~/neodragon-build/` untouched since session 1:

**The ONNX had 2 Transposes. The converted graph had 47**, moving **779.80 MB** per
inference against only **671.17 MB** of convolution output. DistilT5 was the control and a
very good one: near-identical Transpose *count* (48) but **41× fewer bytes** (18.87 MB), and
it is only 3.7× off its paper figure.

Mechanism, visible in five lines of the graph:

```
Conv2d      [4, 320, 512, 128]   NHWC — HTP native
Transpose   [4, 128, 320, 512]   167.77 MB → forced to NCHW
Reshape     [8,  64, 320, 512]   TGrow: 128ch → 2 frames × 64ch
Transpose   [8, 320, 512,  64]   167.77 MB → back to NHWC
Conv2d      [8, 320, 512,  64]
```

`TGrow` folds time into the **batch** axis. Batch is outermost in both layouts, so that is a
free view in NCHW (C adjacent to N) and a genuine scatter in NHWC (C innermost) — and HTP
runs convolution channel-last. No converter flag fixes it.

## The fix

`work/export/export_vae_decoder_stream.py`. Follow upstream's `parallel=False` streaming
order instead of `parallel=True`, so time never enters batch:

1. **`TGrow` → conv + channel chunk.** Upstream does `conv → reshape(-1,C,H,W)` and the
   consumer does `.view(N, stride*C, H, W).chunk(stride, 1)`. The two reshapes cancel; it is
   really a 1×1 conv then a slice on the innermost axis, which is free in NHWC.
2. **The 9 MemBlock states become graph inputs and outputs.** Within one invocation state is
   threaded internally (blocks after the first `TGrow` run 2×, 4×, 8×); only the last value
   crosses the boundary.
3. **`post_quant_conv` folds to `Conv2d`** — it is a 1×1×1 `CausalConv3d`, `time_pad = 0`,
   pointwise. Removes the graph's only 5-D op.

The frame trim moved **out** of the graph: `frames_to_trim = 7` applies to the whole video,
so the caller drops the first 7 frames of invocation 1 only. 7 × 8 − 7 = **49 frames**,
matching Fig 13.

Correctness vs the stock `parallel=True` decoder at T=3: `max|diff| = 4.113e-06`,
**115.6 dB** — fp32 rounding.

## Results

| | T=1 parallel | streaming (4f) | **vaedecsn (4g, shipping)** | paper |
|---|---|---|---|---|
| Transpose nodes / bytes | 47 / 779.80 MB | 20 / 63.00 MB | **2 / 7.95 MB** | — |
| transpose ÷ compute | 1.16 | 0.09 | **0.01** | — |
| `Reshape` / `Pad` | 23 / 9 | 0 / 0 | **0 / 0** | — |
| MACs | 205.07 G | 205.07 G | 205.07 G | — |
| deploy SNR (held out) | 31.55 dB | 34.27 dB | **34.27 dB** | 35 dB |
| **latency / invocation** | **6.10 s** | 114.04 ms | **100.18 ms** | 248.9 ms |
| usable frames / invocation | 1 | 8 | **8** | — |
| **49-frame decode** | **298.9 s** | 0.80 s | **0.70 s** | — |

**60.9× per invocation, 487× per usable frame, +2.72 dB.** Identical arithmetic in every
build — the whole win was layout and discarded work.

After 4f the 20 surviving Transposes were all I/O boundary (`state_k_nhwc` ×9,
`nstate_k_nchw` ×9, `video_nchw`, `latent_nhwc`); 4g removed 18 of them.

Accuracy improving alongside latency was not predicted. Two plausible causes: the deleted
reshape/transpose chains were themselves quantised tensors, each an extra requantisation;
and the 7 discarded frames were still consuming calibration range.

## Things that cost time, recorded so they don't again

- **Calibration state cannot be zeros.** With 9 explicit state inputs, calibrating on zeros
  sets 9 of 10 input ranges to 0. `make_stream_calib.py` replays each prompt's 7 frames in
  temporal order; frame 0 legitimately has zero state, so it appears at its true 1-in-7
  rate. Measured `|state|max`: **0.000** at frame 0, **6.119** at frames 1–6.
- **Two different input lists.** The converter reads paths on the **host** (WSL); `qnn-net-run`
  reads them on the **device**. With one input this was invisible. The generator now emits
  `calib_list_host.txt` and `input_list.txt`, and the conversion script asserts every file
  exists before spending 50 calibration passes.
- **adb: the phone attaches twice** (USB `<DEVICE_SERIAL>` + wireless `<device-ip>:5555`) →
  "more than one device/emulator". Prefer USB. And **adb writes progress to stderr on
  success**, so PowerShell `$ErrorActionPreference='Stop'` makes a completed push look
  fatal. Check `$LASTEXITCODE`.
- **Git Bash mangles `/mnt/...`** into `C:/Program Files/Git/mnt/...`. Invoke WSL from
  PowerShell. Also `cd` in a Bash call persists — use absolute paths.
- **Backslashes get eaten in Bash heredocs here.** `"...\\\n"` arrived as backslash + `n`,
  which silently wrote BEL/VT control characters into a doc. Use `chr(92)` / `chr(10)`.
- **QNN dtype encoding**: `UFIXED_16` is `0x0416` and the low byte spells the width in
  *decimal*. Reading it as hex doubles every 16-bit tensor — it produced a 2× error mid-analysis.

## Reproduce

```powershell
# 1. export + verify the streaming graph (~1 min)
py -3.10 work\export\export_vae_decoder_stream.py

# 2. calibration with real states (~5 min, writes ~3.09 GB)
py -3.10 work\device\make_stream_calib.py

# 3. W8A16 conversion in WSL (~12 min)
py -3.10 work\device\permute_states_nhwc.py          # states -> NHWC for the shipping build
wsl -d Ubuntu -e bash /mnt/c/path/to/Neodragon/work/qnn/convert_vae_stream_nhwc.sh vaedecsn

# 4. THE CHEAP CHECK — before any device time
py -3.10 work\device\analyze_net.py work\device\vaedecs_net.json

# 5. device
py -3.10 work\device\make_stream_testset.py
.\work\device\push_stream.ps1 -Name vaedecsn
adb -s <DEVICE_SERIAL> shell "/data/local/tmp/nd/run_stream.sh vaedecsn N"
adb -s <DEVICE_SERIAL> pull /data/local/tmp/nd/sout_N work\device\sout_N
py -3.10 work\device\compare_stream.py sout_N
# latency: qnn-profile-viewer.exe is in the SDK's bin_64-windows-msvc -- no WSL needed
```

## Phase 4g — NHWC states, and a self-inflicted bug worth remembering

The 63 MB left after 4f was all I/O boundary. `--preserve_io layout` takes an explicit tensor
list, so naming only `latent video` frees the 18 state tensors to go channel-last:
**63.00 → 7.95 MB**, and **−10%** latency on a proper interleaved A/B.

The first A/B was thrown out: measured in separate thermal sessions it read −5% on min and
−19% on average. Two metrics disagreeing that badly *is* trap #10. Interleaved in one session
(`ab_stream.sh`) it was −11.1% / −10.0% in rounds 1–2, with round 3's NCHW run hitting a
181.9 ms thermal spike. Reported the min-of-all: **111.37 → 100.18 ms**.

**The bug worth remembering:** `push_stream.ps1` reported `pushed context binary:
vaedecsn_v79.bin` while actually pushing `vaedecs_v79.bin`. Cause was my own edit — a
`str.replace` whose search string contained backslashes silently failed to match (heredoc
backslash mangling, the third time that bit this session), so the push line kept the old
hardcoded name while the adjacent `Write-Output` — which had no backslashes — updated fine.
`qnn-net-run`'s only symptom was `Received path to an empty file`.

Two lessons, both now in the script:
- **Verify a push, never assume it.** `push_stream.ps1` now stats the device file and
  compares byte length, throwing on mismatch.
- **Assert on every scripted edit.** Every `str.replace` in this session's tooling is now
  preceded by `assert old in s`. The ones that had it caught their failures instantly; the
  one that didn't cost a wrong measurement and 20 minutes.

## Late finding: the VAE encoder is a missing phase

Checked at the very end of the session while assessing overall feasibility. The phase plan
(brief 7, and CLAUDE.md ever since) lists only the VAE **decoder**. But the AR t2v flow is:

```
prompt -> SSD1B generates a first FRAME (pixels)
       -> vae.encode(image)            <-- generation_utils.py:531
       -> MMDiT continues autoregressively in latent space
       -> vae.decode(latents)          <-- Phase 4, done
```

So the encoder is required for E2E and was never scoped. It is the hard half of the
"asymmetric" VAE: `modeling_causal_ops.py`'s `CausalConv3d` carries a stateful `deque`
feature cache and is not exportable as written. Paper Table 7 puts it at **1206.5 ms** on
8 Elite Gen4 — the most expensive single component in the whole pipeline, 5x the decoder's
original figure.

The good news is that this is *exactly* the problem Phase 4 just solved. A `deque` feature
cache is an implicit state buffer; `export_vae_decoder_stream.py` is a worked example of
converting one into explicit graph inputs and outputs. The pattern transfers directly.

**Do not quote an E2E timeline without scoping this first.**

## State at end of session

- Phase 4 (decoder) **done**. Phase 1 done. Phases 2, 3, 5, 6 untouched, plus a newly
  identified **VAE encoder** phase that was never in the plan (see above).
- **Shipping build: `work/device/vaedecsn_v79.bin`** — 100.18 ms, 34.27 dB, 8 frames per
  invocation. Expects **NHWC** state tensors; `latent`/`video` stay NCHW. `vaedecs_v79.bin`
  is the NCHW-state variant, 10% slower, kept for A/B only.
- **The device currently holds NHWC state raws.** Re-running `vaedecs` against them would
  produce quiet garbage — `permute_states_nhwc.py --revert` and re-push first.
- `work/calib/vae_dec_stream/` (~3.09 GB) is **deletable** — regenerate in ~5 min. Its state
  raws are currently **NHWC** (sentinel `.layout_nhwc`); `permute_states_nhwc.py --revert`
  puts them back.
- Deleted `~/neodragon-qnn` (1.1 GB redundant py3.12 venv). Note `work/qnn/setup_env.sh` and
  `work/qnn/install_onnx.sh` reference it and are now dead bootstrap scripts.
- Disk: **~33 GB free on C:**, down from 88 GB in session 1. WSL `~/npuconvert` is 35 GB and
  `~/neodragon-build` ~5 GB.
