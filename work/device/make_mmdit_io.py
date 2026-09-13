"""Turn captured DiT calls into per-stage QNN calibration / device inputs.

`work/pipeline/capture_mmdit_calib.py` stores the RAW call (latent list, context-adapter
output, prompt mask, pooled projection, timestep). This derives the actual graph inputs:
pads each call up to its stage's envelope and rebuilds `attn_mask` / `rope_cos` / `rope_sin`
with `export_mmdit_stage.stage_conditioning_padded`, so the padding scheme can change
without re-running the GPU.

Sizes are worth knowing before you run it. Per sample the dominant tensor is `attn_mask`,
[1,1,S,S] fp32:

    stage 0  S =  408      0.67 MB     ~1.6 MB/sample     ~0.5 GB for 300
    stage 1  S =  648      1.68 MB     ~2.6 MB/sample     ~0.8 GB for 300
    stage 2  S = 1728     11.94 MB    ~13.2 MB/sample     ~4.0 GB for 300

so build one stage at a time, convert, and delete before moving on.

  usage: py -3.10 work/device/make_mmdit_io.py --stage 0 --split calib
"""

import re
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "work" / "export"))
sys.path.insert(0, str(ROOT / "work" / "audit"))
CALLS = ROOT / "work" / "calib" / "mmdit"

DEVICE_ROOT = "/data/local/tmp/nd"

TEXT_TOKENS = 128
CAPTION_DIM = 1536
POOLED_DIM = 2048
LAT_C = 16
HEAD_DIM = 64
EMB_DIM = 1536          # time_text_embed output == transformer inner_dim


def _seq(env):
    from mmdit_shapes import tokens
    return TEXT_TOKENS + sum(tokens(x) for x in env)


# every graph input's shape, derived from the envelope alone
SHAPES = {
    "encoder_hidden_states": lambda env: (1, TEXT_TOKENS, CAPTION_DIM),
    # the shipping conditioning input: silu(time_text_embed(t, pooled)), computed on
    # the host. See export_mmdit_stage.host_temb -- session 5 measured time_proj's Sin
    # at 46.41 dB because `timestep_ratio` is encoded over [0, 1000] at 16 bits.
    "temb_act": lambda env: (1, EMB_DIM),
    # kept for the --cond raw ablation build only
    "pooled_projections": lambda env: (1, POOLED_DIM),
    "timestep_ratio": lambda env: (1,),
    # rank 3 on purpose: a rank-4 tensor entering the attention chain gets read as
    # image layout by the converter and permuted (measured ratio 1.35 -> see
    # docs/phase5-mmdit-scope.md).
    "attn_mask": lambda env: (1, _seq(env), _seq(env)),
    "rope_cos": lambda env: (_seq(env), 1, HEAD_DIM // 2),
    "rope_sin": lambda env: (_seq(env), 1, HEAD_DIM // 2),
}
for _i in range(8):
    SHAPES["latent_{}".format(_i)] = (
        lambda env, i=_i: (1, LAT_C, env[i][0], env[i][1], env[i][2]))


def load_call(path):
    d = np.load(path)
    n = int(d["n_latents"])
    lats = [torch.from_numpy(d["latent_{}".format(i)]).float() for i in range(n)]
    shapes = [(int(t.shape[2]), int(t.shape[3]), int(t.shape[4])) for t in lats]
    return d, lats, shapes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=[0, 1, 2])
    ap.add_argument("--split", default="calib", choices=["calib", "test"])
    ap.add_argument("--videos", default="", help="e.g. 0-49 (default: all found)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cond", default="temb", choices=["temb", "raw"],
                    help="temb: one hoisted silu(time_text_embed) input (shipping). "
                         "raw: pooled_projections + timestep_ratio, the ablation build")
    ap.add_argument("--out-name", default="",
                    help="override the output directory name")
    args = ap.parse_args()
    host_cond = args.cond == "temb"

    from export_mmdit_stage import (StageMMDiT, envelope_for, host_temb,  # noqa: E402
                                    pad_latents, stage_conditioning_padded)
    from mmdit_shapes import pyramid_hw, tokens                          # noqa: E402
    from neodragon.pyramid_mmdit import PyramidMMDiT                     # noqa: E402

    env = envelope_for(args.stage)
    n_img = sum(tokens(s) for s in env)
    stage_hw = pyramid_hw(args.stage)

    out = ROOT / "work" / ("calib" if args.split == "calib" else "device")
    out = out / (args.out_name or ("mmdit_s{}".format(args.stage) if args.split == "calib"
                                   else "mio_s{}".format(args.stage)))
    out.mkdir(parents=True, exist_ok=True)
    dev_dir = "{}/{}".format(DEVICE_ROOT, out.name)

    print("=" * 76)
    print("MMDiT stage {} {} inputs   envelope {}  ({} image tokens)".format(
        args.stage, args.split, " ".join("{}x{}x{}".format(*x) for x in env), n_img))
    print("=" * 76)

    # the DiT is needed only for _prepare_temporal_rope_ids and temp_rope_embed,
    # which are tiny -- but from_pretrained is the only clean way to get them.
    print("")
    print("[load] DiT (for the RoPE tables only) ...")
    dit = PyramidMMDiT.from_pretrained(
        str(ROOT / "work" / "models" / "neodragon" / "diffusion_transformer_320p"),
        torch_dtype=torch.float32).eval()

    # In test mode we also need the fp32 reference the device output is scored against,
    # produced by the ENVELOPE graph on the PADDED inputs -- i.e. exactly the graph that
    # ships. Scoring against the stock model at the unit's true shape would fold the
    # padding error (~125 dB, negligible) into the quantisation number and muddy it.
    net = (StageMMDiT(dit, env, temb_host=host_cond).eval()
           if args.split == "test" else None)

    lo, hi = (-1, 10 ** 9)
    if args.videos:
        lo, hi = (int(x) for x in args.videos.split("-"))

    files = sorted(CALLS.glob("call_*.npz"))
    assert files, "no captured calls -- run capture_mmdit_calib.py first"

    lines, n = [], 0
    for f in files:
        vid = int(f.stem.split("_")[1])
        if args.videos and not (lo <= vid <= hi):
            continue
        d, lats, shapes = load_call(f)
        if (shapes[-1][1], shapes[-1][2]) != stage_hw:
            continue                       # a different pyramid stage

        eam = torch.from_numpy(d["encoder_attention_mask"]).float()
        mask, cos, sin = stage_conditioning_padded(dit, shapes, env, eam)
        padded = pad_latents(lats, shapes, env)

        tag = "{:04d}".format(n)
        parts = []

        def emit(name, arr):
            p = out / "{}_{}.raw".format(name, tag)
            np.ascontiguousarray(arr, dtype=np.float32).tofile(p)
            parts.append("{}:={}/{}".format(name, dev_dir, p.name))

        pooled = torch.from_numpy(d["pooled_projections"]).float()
        tstep = torch.from_numpy(d["timestep_ratio"]).float()
        temb_act = host_temb(dit, tstep, pooled)

        emit("encoder_hidden_states", d["encoder_hidden_states"])
        if host_cond:
            emit("temb_act", temb_act.numpy())
        else:
            emit("pooled_projections", d["pooled_projections"])
            emit("timestep_ratio", d["timestep_ratio"])
        emit("attn_mask", mask.numpy())
        emit("rope_cos", cos.numpy())
        emit("rope_sin", sin.numpy())
        for i, la in enumerate(padded):
            emit("latent_{}".format(i), la.numpy())

        if net is not None:
            cond = (temb_act,) if host_cond else (pooled, tstep)
            with torch.no_grad():
                ref = net(torch.from_numpy(d["encoder_hidden_states"]).float(),
                          *cond, mask, cos, sin, *padded)
            np.ascontiguousarray(ref.numpy(), dtype=np.float32).tofile(
                out / "nref_{}.raw".format(tag))

        lines.append(" ".join(parts))
        n += 1
        if n % 25 == 0:
            print("  {:4d} samples".format(n))
        if args.limit and n >= args.limit:
            break

    name = "calib_list_host.txt" if args.split == "calib" else "input_list.txt"
    if args.split == "calib":
        # host paths for qnn-onnx-converter, NOT device paths (see convert scripts)
        host = [ln.replace(dev_dir, re.sub(r"^([A-Za-z]):", lambda m: "/mnt/" + m.group(1).lower(), str(out).replace("\\", "/")))
                for ln in lines]
        (out / name).write_text("\n".join(host) + "\n", newline="\n")
    else:
        (out / name).write_text("\n".join(lines) + "\n", newline="\n")

    # --input_dim manifest, so the convert script never hardcodes a stage's shapes
    if lines:
        dims = []
        for spec in lines[0].split():
            nm = spec.split(":=")[0]
            a = np.fromfile(out / "{}_0000.raw".format(nm), dtype=np.float32)
            shp = SHAPES[nm](env) if nm in SHAPES else None
            assert shp is not None, nm
            assert int(np.prod(shp)) == a.size, (nm, shp, a.size)
            dims.append("--input_dim {} {}".format(nm, ",".join(str(d) for d in shp)))
        (out / "dims.txt").write_text(" \\\n    ".join(dims) + "\n", newline="\n")
        print("  wrote dims.txt ({} inputs)".format(len(dims)))

    total = sum(f.stat().st_size for f in out.glob("*.raw"))
    if args.split == "test":
        print("  {} fp32 references written as nref_*.raw (NOT pushed to the device)"
              .format(n))
    print("")
    print("wrote {} samples ({} tensors each) to {}".format(
        n, len(lines[0].split()) if lines else 0, out))
    print("  {:.2f} GB on disk, list -> {}".format(total / 1024 ** 3, name))
    if args.split == "calib":
        print("  paper Table 9 uses 300 calibration samples for each MMDiT row")


if __name__ == "__main__":
    main()
