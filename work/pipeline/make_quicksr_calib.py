"""Calibration frames for QuickSRNet: real decoded video frames, not synthetic images.

Trap #4, again. QuickSRNet's deployed input is whatever the streaming VAE decoder emits
at the very end of the pipeline -- frames that have already accumulated the quantisation
error of the MMDiT and the decoder, and that carry the specific texture statistics of a
320x512 Neodragon output. Calibrating on natural photographs, or on noise, would set the
encodings for a different distribution entirely.

So the frames come from decoding the kept video latents with the HOST fp32 VAE, exactly
as `decode_video_latents.py` does, including the detail that the first latent frame and
the rest use DIFFERENT scale/shift factors -- getting that wrong washes the whole video
out and gives no indication why.

Paper Table 9 uses 500 calibration samples for QuickSRNet. `work/calib/vae_dec` holds 56
latents, which decode to 8 frames each = 448 -- close enough, and the alternative is
several GPU-hours of fresh reference runs for the last 52.

Frames are written as [1,3,H,W] fp32 in **[0,1]**, which is the range the QuickSRNet
checkpoint was trained on. The VAE emits [-1,1]; that conversion happens here so the
calibration data and the deployed input agree.

  usage: py -3.10 work/pipeline/make_quicksr_calib.py [--limit 0]
"""

import argparse
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "src" / "neodragon"
MODEL = ROOT / "work" / "models" / "neodragon"
SRC = ROOT / "work" / "calib" / "vae_dec"
OUT = ROOT / "work" / "calib" / "quicksr"

sys.path.insert(0, str(REPO))
_pkg = types.ModuleType("neodragon")
_pkg.__path__ = [str(REPO / "neodragon")]
sys.modules["neodragon"] = _pkg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all latents")
    a = ap.parse_args()

    from neodragon.asymmetric_causal_video_vae import AsymmetricCausalVideoVAE

    lat_files = sorted(SRC.glob("latent_*.raw"))
    if a.limit:
        lat_files = lat_files[:a.limit]
    assert lat_files, f"no latents in {SRC}"
    print(f"[src]  {len(lat_files)} latents from {SRC}")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[load] causal_video_vae on {dev} ...")
    vae = AsymmetricCausalVideoVAE.from_pretrained(
        str(MODEL / "causal_video_vae"), torch_dtype=torch.float32).to(dev).eval()

    # NOT re-applying scale/shift. run_reference.py hooks vae.post_quant_conv and notes
    # that what it saves is "exactly the tensor the graph's `latent` input -- already
    # scale/shift-corrected by _decode_latent". Applying the factors again here would
    # decode a wrong-scaled latent and produce washed-out frames with nothing to say why.

    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.raw"):
        f.unlink()

    # The latents are per-FRAME slices of whole videos: run_reference wrote 7 latent
    # frames per prompt. Decoding them one at a time gives ONE frame each, because the
    # causal decoder needs the temporal context -- 56 latents produced 56 frames instead
    # of the expected 392. They have to be re-stacked into [1,16,7,40,64] per video.
    T = 7
    assert len(lat_files) % T == 0, f"{len(lat_files)} latents is not a multiple of {T}"
    n_vid = len(lat_files) // T
    print(f"[group] {len(lat_files)} latent frames -> {n_vid} videos of {T}")

    n = 0
    for v_i in range(n_vid):
        stack = [np.fromfile(lat_files[v_i * T + t], dtype=np.float32).reshape(16, 1, 40, 64)
                 for t in range(T)]
        lat = torch.from_numpy(np.concatenate(stack, axis=1)[None]).to(dev)  # [1,16,7,40,64]
        with torch.no_grad():
            vid = vae.decode(lat).sample            # [1, 3, T', H, W] in [-1, 1]
        fr = vid[0].permute(1, 0, 2, 3).contiguous()   # [T', 3, H, W]
        # [-1,1] -> [0,1]: the range the QuickSRNet checkpoint expects.
        fr = ((fr + 1.0) * 0.5).clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
        for t in range(fr.shape[0]):
            fr[t][None].tofile(OUT / f"frame_{n:04d}.raw")
            n += 1
        print(f"  video {v_i+1}/{n_vid}: {fr.shape[0]} frames  (total {n})")

    # WSL paths, not Windows ones. The converter runs under WSL (x86_64-linux-clang)
    # and cannot see "C:/..." -- it reports every entry as a missing calibration file
    # and refuses to start, which is exactly how the first attempt failed.
    def wsl(path):
        t = path.resolve().as_posix()
        return "/mnt/" + t[0].lower() + t[2:] if len(t) > 1 and t[1] == ":" else t

    lst = OUT / "calib_list_host.txt"
    with open(lst, "w", newline="\n") as fh:
        for k in range(n):
            fh.write(f"frame:={wsl(OUT / f'frame_{k:04d}.raw')}\n")

    dims = OUT / "dims.txt"
    dims.write_text("--input_dim frame 1,3,320,512\n", newline="\n")

    mb = sum(f.stat().st_size for f in OUT.glob('*.raw')) / 1e6
    print(f"\nwrote {n} frames ({mb:.0f} MB) to {OUT}")
    print(f"  list -> {lst.name}, dims -> {dims.name}")
    print("  paper Table 9 uses 500 calibration samples for QuickSRNet")


if __name__ == "__main__":
    main()
