"""Capture real SSD1B UNet calls for Phase 3 calibration and testing.

Table 9 calibrates the SSD1B UNet with 500 samples. The first-frame pipeline runs 4
denoising steps per prompt (timesteps 999/749/499/249, guidance_scale 0.0 so no CFG and
batch 1), so 125 prompts give exactly 500 real calls.

What is stored is what the EXPORTED graph takes, not what the stock module takes: the two
sinusoidal projections are hoisted to the host (see export_ssd1b_unet.py), so each sample
is (sample, t_emb, aug_emb, encoder_hidden_states). Storing the hoisted form means the
calibration set matches the graph exactly and does not have to be regenerated if the
padding or conditioning scheme changes.

Per sample: sample 160 KB + encoder_hidden_states 630 KB + t_emb 1.2 KB + aug_emb 5 KB
~= 0.8 MB, so ~400 MB for 500.

  usage: py -3.10 work/pipeline/capture_unet_calib.py --prompts 125 --split calib
         py -3.10 work/pipeline/capture_unet_calib.py --prompts 4 --split test
"""

import re
import argparse
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT / "src" / "neodragon"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
_pkg = types.ModuleType("neodragon")
_pkg.__path__ = [str(REPO / "neodragon")]
sys.modules["neodragon"] = _pkg

DEVICE_ROOT = "/data/local/tmp/nd"
TAG = "ssd1bunet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=int, default=125)
    ap.add_argument("--split", default="calib", choices=["calib", "test"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()

    from run_reference import build_ssd1b
    from neodragon.utils.generation_utils import DEFAULT_PROMPT_MODIFIER

    out = (ROOT / "work" / "calib" / TAG if a.split == "calib"
           else ROOT / "work" / "device" / ("io_" + TAG))
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.raw"):
        f.unlink()

    prompts = [ln.strip() for ln in
               (REPO / "prompts" / "vbench_prompts.txt").read_text(
                   encoding="utf-8").splitlines() if ln.strip()]
    print("=" * 74)
    print("SSD1B UNET CAPTURE  split={}  {} prompts x 4 steps".format(
        a.split, a.prompts))
    print("=" * 74)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ffg = build_ssd1b(torch.float16)
    ffg.enable_model_cpu_offload(device=dev)
    unet = ffg.unet

    grabbed = []
    shapes = {}

    # A forward PRE-HOOK, not a monkeypatched forward. `enable_model_cpu_offload()`
    # installs accelerate hooks that re-bind `module.forward`, so an instance-attribute
    # patch survives only the FIRST generation -- the smoke test caught exactly that
    # (prompt 1: 4 calls, prompt 2: 0 calls). Hooks are re-entrant and survive.
    def spy(_module, args, kwargs):
        sample = args[0] if args else kwargs["sample"]
        timestep = args[1] if len(args) > 1 else kwargs["timestep"]
        ehs = kwargs.get("encoder_hidden_states")
        ack = kwargs.get("added_cond_kwargs")
        if ehs is None or ack is None:
            return None
        with torch.no_grad():
            w = next(unet.add_embedding.parameters())
            t = timestep if torch.is_tensor(timestep) else torch.tensor([timestep])
            t = t.reshape(-1)[:1].to(w.device, torch.float32)
            t_emb = unet.time_proj(t).to(torch.float32)
            te = ack["text_embeds"].to(w.device, torch.float32)
            ti = ack["time_ids"].to(w.device)
            ti_emb = unet.add_time_proj(ti.flatten()).reshape(te.shape[0], -1)
            add_in = torch.cat([te, ti_emb.to(torch.float32)], dim=-1)
            aug = unet.add_embedding(add_in.to(w.dtype))
        rec = {
            "sample": sample[:1].detach().float().cpu().numpy(),
            "t_emb": t_emb[:1].detach().float().cpu().numpy(),
            "aug_emb": aug[:1].detach().float().cpu().numpy(),
            "encoder_hidden_states": ehs[:1].detach().float().cpu().numpy(),
        }
        if not shapes:
            shapes.update({k: v.shape for k, v in rec.items()})
        grabbed.append(rec)
        return None

    handle = unet.register_forward_pre_hook(spy, with_kwargs=True)

    n = 0
    lines = []
    host = re.sub(r"^([A-Za-z]):", lambda m: "/mnt/" + m.group(1).lower(), str(out).replace("\\", "/"))
    for i in range(a.start, a.start + a.prompts):
        prompt = prompts[i % len(prompts)]
        gen = torch.Generator(device="cpu").manual_seed(a.seed + i)
        grabbed.clear()
        with torch.no_grad():
            ffg(prompt=prompt + DEFAULT_PROMPT_MODIFIER,
                num_images_per_prompt=1, generator=gen)
        for rec in grabbed:
            parts = []
            for k, v in rec.items():
                p = out / "{}_{:04d}.raw".format(k, n)
                np.ascontiguousarray(v, dtype=np.float32).tofile(p)
                parts.append("{}:={}/{}".format(
                    k, host if a.split == "calib" else
                    "{}/io_{}".format(DEVICE_ROOT, TAG), p.name))
            lines.append(" ".join(parts))
            n += 1
        print("[{:4d}/{}] {:2d} calls  {:.50s}".format(
            i + 1 - a.start, a.prompts, len(grabbed), prompt))

    handle.remove()

    name = "calib_list_host.txt" if a.split == "calib" else "input_list.txt"
    (out / name).write_text("\n".join(lines) + "\n", newline="\n")
    assert shapes, "the pre-hook never fired -- nothing was captured"
    if a.split == "calib":
        dims = " ".join("--input_dim {} {}".format(
            k, ",".join(str(x) for x in s)) for k, s in shapes.items())
        (out / "dims.txt").write_text(dims + "\n", newline="\n")
    tot = sum(f.stat().st_size for f in out.glob("*.raw"))
    print("")
    print("wrote {} samples ({:.2f} GB) -> {}".format(n, tot / 1024 ** 3, out))
    for k, s in shapes.items():
        print("   {:<24} {}".format(k, s))


if __name__ == "__main__":
    main()
