"""SSD1B VAE decoder -> ONNX + calibration + test set (Table 9 row "SSD1B Dec", 31 dB).

The last module on the first-frame path: the UNet's final latent [1,4,80,128] is decoded
to a 640x1024 image, which is then LANCZOS-downscaled to 512x320 and re-encoded by the
causal video VAE (Phase 4b) to seed the AR loop.

Calibration data has to be REAL final latents -- trap #4 -- so they are captured with a
pre-hook on `vae.decode` during actual SSD1B runs. There is exactly one decode per prompt,
so N prompts give N samples (unlike the UNet, where 4 denoising steps per prompt gave 4).

This is a conv decoder, so the traps that matter are #11 (reshape-into-batch layout, which
cost the video VAE decoder 53.5x) and #17 (GroupNorm arrives fused). `analyze_net.py`
screens both for free -- run it before device time.

  usage: py -3.10 work/export/export_ssd1b_vaedec.py --prompts 128 --export
"""

import argparse
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT / "src" / "neodragon"
MODEL = ROOT / "work" / "models" / "neodragon"
OUT = ROOT / "work" / "onnx"
CAL = ROOT / "work" / "calib"
DEV = ROOT / "work" / "device"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "work" / "pipeline"))
sys.path.insert(0, str(REPO))
_pkg = types.ModuleType("neodragon")
_pkg.__path__ = [str(REPO / "neodragon")]
sys.modules["neodragon"] = _pkg

DEVICE_ROOT = "/data/local/tmp/nd"
TAG = "ssd1bvaedec"
LAT = (1, 4, 80, 128)


class DecoderOnly(nn.Module):
    """`post_quant_conv` + `decoder`, i.e. exactly what AutoencoderKL.decode() runs.

    The scaling divide is left on the host: it is one multiply on a 40 KB tensor and
    keeping it out means the graph's input encoding is calibrated on the real latent
    distribution rather than a rescaled one.
    """

    def __init__(self, vae):
        super().__init__()
        self.post_quant_conv = vae.post_quant_conv
        self.decoder = vae.decoder

    def forward(self, latent):
        return self.decoder(self.post_quant_conv(latent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=int, default=128)
    ap.add_argument("--test", type=int, default=8)
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args()

    from diffusers import AutoencoderKL

    print("[load] ssd_1b_vae ...")
    # ships only as the fp16 twin -- name the variant explicitly (run_reference.pick())
    d = MODEL / 'ssd_1b_vae'
    kw = {'variant': 'fp16'} if (d / 'diffusion_pytorch_model.fp16.safetensors').exists() else {}
    vae = AutoencoderKL.from_pretrained(str(d), torch_dtype=torch.float32, **kw).eval()
    net = DecoderOnly(vae).eval()
    print("       decoder {:.1f} M params  scaling_factor {}".format(
        sum(p.numel() for p in net.parameters()) / 1e6, vae.config.scaling_factor))

    cache = CAL / TAG / "_latents.npy"
    total = a.prompts + a.test
    if a.reuse and cache.exists():
        arr = np.load(cache)
        assert arr.shape[0] >= total, (arr.shape, total)
        lats = [torch.from_numpy(arr[i:i + 1]) for i in range(total)]
        print("[reuse] {} cached latents".format(total))
    else:
        from run_reference import build_ssd1b
        from neodragon.utils.generation_utils import DEFAULT_PROMPT_MODIFIER
        prompts = [ln.strip() for ln in
                   (REPO / "prompts" / "vbench_prompts.txt").read_text(
                       encoding="utf-8").splitlines() if ln.strip()]
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ffg = build_ssd1b(torch.float16)
        ffg.enable_model_cpu_offload(device=dev)
        grabbed = []

        def spy(_m, args, kwargs):
            z = args[0] if args else kwargs.get("input")
            if z is not None and tuple(z.shape[1:]) == LAT[1:]:
                grabbed.append(z[:1].detach().float().cpu().numpy())
            return None

        # Hook post_quant_conv, NOT the vae module: the pipeline calls vae.decode(z),
        # which never goes through vae.__call__, so a hook on the vae itself never
        # fires (measured: 32 prompts, 0 captures). decode() -> _decode() ->
        # post_quant_conv(z) is the first nn.Module __call__ on the path, and its
        # input is the already scaling_factor-divided latent -- exactly what
        # DecoderOnly takes. run_reference.py line 15 uses this same capture point.
        h = ffg.vae.post_quant_conv.register_forward_pre_hook(spy, with_kwargs=True)
        print("[capture] {} prompts, 1 decode each ...".format(total))
        for i in range(total):
            with torch.no_grad():
                ffg(prompt=prompts[i % len(prompts)] + DEFAULT_PROMPT_MODIFIER,
                    num_images_per_prompt=1,
                    generator=torch.Generator(device="cpu").manual_seed(i))
            if (i + 1) % 16 == 0:
                print("  {}/{}  ({} captured)".format(i + 1, total, len(grabbed)))
        h.remove()
        assert len(grabbed) >= total, "hook fired {} times for {} prompts".format(
            len(grabbed), total)
        lats = [torch.from_numpy(x) for x in grabbed[:total]]
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, np.concatenate([x.numpy() for x in lats], 0))
        print("[cache] wrote {}".format(cache))

    x0 = lats[0]
    print("[shape] latent {}  range [{:.3f}, {:.3f}]".format(
        tuple(x0.shape), float(x0.min()), float(x0.max())))
    with torch.no_grad():
        ref0 = net(x0)
    print("[shape] image  {}  range [{:.3f}, {:.3f}]".format(
        tuple(ref0.shape), float(ref0.min()), float(ref0.max())))

    if not a.export:
        print("")
        print("pass --export to write the ONNX, calibration set and test set")
        return

    dst = OUT / TAG
    dst.mkdir(parents=True, exist_ok=True)
    raw = dst / (TAG + "_raw.onnx")
    torch.onnx.export(net, (x0,), str(raw), input_names=["latent"],
                      output_names=["image"], opset_version=17,
                      do_constant_folding=True, dynamo=False)
    import onnx
    from graph_fixes import op_histogram, scan_extreme
    m = onnx.load(str(raw), load_external_data=True)
    print("[ops]    " + ", ".join("{}x{}".format(k, v) for k, v in
                                  sorted(op_histogram(m).items(), key=lambda kv: -kv[1])[:12]))
    print("[safe]   unsafe constants: {}".format(len(scan_extreme(m))))
    print("[onnx]   {:.0f} MB".format(raw.stat().st_size / 1e6))

    cal = CAL / TAG
    cal.mkdir(parents=True, exist_ok=True)
    for f in cal.glob("*.raw"):
        f.unlink()
    host = str(cal).replace("C:", "/mnt/c").replace("\\", "/")
    rows = []
    for i in range(a.prompts):
        p = cal / "latent_{:04d}.raw".format(i)
        np.ascontiguousarray(lats[i].numpy(), dtype=np.float32).tofile(p)
        rows.append("latent:={}/{}".format(host, p.name))
    (cal / "calib_list_host.txt").write_text("\n".join(rows) + "\n", newline="\n")
    (cal / "dims.txt").write_text("--input_dim latent {}\n".format(
        ",".join(str(d) for d in LAT)), newline="\n")
    print("[calib]  {} samples".format(a.prompts))

    io = DEV / ("io_" + TAG)
    io.mkdir(parents=True, exist_ok=True)
    for f in io.glob("*.raw"):
        f.unlink()
    rows = []
    for j in range(a.test):
        x = lats[a.prompts + j]
        np.ascontiguousarray(x.numpy(), dtype=np.float32).tofile(
            io / "latent_{:04d}.raw".format(j))
        with torch.no_grad():
            r = net(x)
        np.ascontiguousarray(r.numpy(), dtype=np.float32).tofile(
            io / "vref_{:04d}.raw".format(j))
        rows.append("latent:={}/io_{}/latent_{:04d}.raw".format(DEVICE_ROOT, TAG, j))
    (io / "input_list.txt").write_text("\n".join(rows) + "\n", newline="\n")
    print("[test]   {} held-out samples + fp32 refs".format(a.test))


if __name__ == "__main__":
    main()
