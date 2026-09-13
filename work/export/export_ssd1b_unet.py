"""Phase 3 -- the SSD1B UNet as a static graph, with the sinusoidal embeddings hoisted.

The first-frame path: `first_frame_gen.py` runs SSD-1B at **640x1024** (latent 80x128),
4 steps, timesteps [999, 749, 499, 249], **guidance_scale 0.0 -> no CFG, batch 1**. The
resulting 1024x640 frame is LANCZOS-downscaled to 512x320 and encoded by the causal video
VAE (Phase 4b) to seed the AR loop.

Config: block_out_channels [320, 640, 1280], DownBlock2D then two CrossAttnDownBlock2D,
transformer_layers_per_block [[1],[2,2],[4,4]], cross_attention_dim 2048. So attention
runs at 40x64 = **2560 tokens** (larger than MMDiT stage 2's 1728) and at 20x32 = 640.

Two lessons from this session are applied BEFORE the first conversion rather than after:

  * **The sinusoidal embeddings are hoisted to the host.** In the MMDiT, `time_proj`'s
    `Sin` was quantised at its ARGUMENT -- `timestep_ratio` encoded over [0, 1000] at 16
    bits, a 0.0153 rad step, a ~47 dB ceiling -- and everything downstream inherited it;
    hoisting it was worth **+8.41 dB**. The same shape is here twice, in `get_time_embed`
    (timestep) and `get_aug_embed` (the `add_time_proj` sinusoids over time_ids). Both are
    replaced by ordinary inputs. It costs the host nothing: with 4 timesteps and one set of
    added conditions, there are exactly 4 distinct values per generation.

  * **Watch the conv layout.** This is the most conv-heavy module in the pipeline and
    therefore the one most likely to inherit trap #11, the reshape-into-batch that cost the
    VAE decoder a 53.5x latency penalty. `analyze_net.py` reads the ratio statically for
    free -- run it before any device time. Trap #17 says GroupNorm arrives fused natively,
    verified on the VAE decoder (all 22 fused, ratio 0.05), so the norms should be fine.

  usage: py -3.10 work/export/export_ssd1b_unet.py            # shapes + fp32 check
         py -3.10 work/export/export_ssd1b_unet.py --export
"""

import argparse
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT / "src" / "neodragon"
MODEL = ROOT / "work" / "models" / "neodragon"
OUT = ROOT / "work" / "onnx"
sys.path.insert(0, str(REPO))
_pkg = types.ModuleType("neodragon")
_pkg.__path__ = [str(REPO / "neodragon")]
sys.modules["neodragon"] = _pkg

TAG = "ssd1bunet"
HEIGHT, WIDTH = 640, 1024          # first_frame_gen.DEFAULT_INFERENCE_PARAMS
LAT_H, LAT_W = HEIGHT // 8, WIDTH // 8
TEXT_TOKENS = 77                   # CLIP
CROSS_DIM = 2048
TIMESTEPS = [999, 749, 499, 249]


class HoistedUNet(nn.Module):
    """UNet2DConditionModel taking `t_emb` and `aug_emb` instead of timestep/added conds.

    diffusers 0.35.1 routes both through overridable methods -- `get_time_embed()` and
    `get_aug_embed()` -- so the hoist needs no copy of the 150-line forward: both are
    replaced by closures returning the supplied tensors, and every weight-bearing module
    (including `time_embedding`) stays in the graph exactly as the stock model has it.
    """

    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        self._t_emb = None
        self._aug = None
        unet.get_time_embed = lambda sample, timestep: self._t_emb
        unet.get_aug_embed = lambda emb, encoder_hidden_states, added_cond_kwargs: self._aug

    def forward(self, sample, t_emb, aug_emb, encoder_hidden_states):
        self._t_emb, self._aug = t_emb, aug_emb
        return self.unet(sample, 0, encoder_hidden_states=encoder_hidden_states,
                         added_cond_kwargs={}, return_dict=False)[0]


def host_embeds(unet, timestep, text_embeds, time_ids, dtype=torch.float32):
    """What the host must compute: the two sinusoidal projections, on real values."""
    with torch.no_grad():
        t = torch.tensor([timestep], dtype=dtype)
        t_emb = unet.time_proj(t).to(dtype)                       # [1, 320]
        time_ids_emb = unet.add_time_proj(time_ids.flatten()).reshape(1, -1)
        add_in = torch.cat([text_embeds, time_ids_emb.to(dtype)], dim=-1)
        aug = unet.add_embedding(add_in)                          # [1, 1280]
    return t_emb, aug


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    a = ap.parse_args()

    from diffusers import UNet2DConditionModel

    print("[load] ssd_1b_unet (fp32, CPU) ...")
    unet = UNet2DConditionModel.from_pretrained(
        str(MODEL / "ssd_1b_unet"), torch_dtype=torch.float32).eval()
    print("       {:.1f} M params".format(
        sum(p.numel() for p in unet.parameters()) / 1e6))
    print("[shape] latent {}x{}  text {}x{}  timesteps {}".format(
        LAT_H, LAT_W, TEXT_TOKENS, CROSS_DIM, TIMESTEPS))

    g = torch.Generator().manual_seed(0)
    sample = torch.randn(1, 4, LAT_H, LAT_W, generator=g)
    ehs = torch.randn(1, TEXT_TOKENS, CROSS_DIM, generator=g)
    text_embeds = torch.randn(1, 1280, generator=g)
    time_ids = torch.tensor([[HEIGHT, WIDTH, 0.0, 0.0, HEIGHT, WIDTH]])

    t_emb, aug = host_embeds(unet, TIMESTEPS[0], text_embeds, time_ids)
    print("[host]  t_emb {}  aug_emb {}".format(tuple(t_emb.shape), tuple(aug.shape)))

    # stock reference, through the untouched model
    with torch.no_grad():
        ref = unet(sample, torch.tensor(TIMESTEPS[0], dtype=torch.float32),
                   encoder_hidden_states=ehs,
                   added_cond_kwargs={"text_embeds": text_embeds,
                                      "time_ids": time_ids}, return_dict=False)[0].clone()

    net = HoistedUNet(unet).eval()
    with torch.no_grad():
        got = net(sample, t_emb, aug, ehs)
    md = float((ref - got).abs().max())
    snr = 20 * torch.log10(ref.norm() / (got - ref).norm()).item()
    print("[check] hoisted vs stock: max|diff| = {:.3e}   SNR = {:.1f} dB".format(md, snr))
    assert md < 1e-4, "the hoist must reproduce the stock model"
    print("[shape] noise_pred {}".format(tuple(got.shape)))

    if not a.export:
        print("")
        print("pass --export to write the ONNX")
        return

    dst = OUT / TAG
    dst.mkdir(parents=True, exist_ok=True)
    raw = dst / (TAG + "_raw.onnx")
    names = ["sample", "t_emb", "aug_emb", "encoder_hidden_states"]
    print("")
    print("[export] {} -> {}".format(names, raw))
    torch.onnx.export(net, (sample, t_emb, aug, ehs), str(raw),
                      input_names=names, output_names=["noise_pred"],
                      opset_version=17, do_constant_folding=True, dynamo=False)

    import onnx
    sys.path.insert(0, str(HERE))
    from graph_fixes import op_histogram, scan_extreme
    m = onnx.load(str(raw), load_external_data=True)
    hist = op_histogram(m)
    print("[ops]    " + ", ".join("{}x{}".format(k, v) for k, v in
                                  sorted(hist.items(), key=lambda kv: -kv[1])[:16]))
    print("[safe]   unsafe constants: {}".format(len(scan_extreme(m))))
    for f in dst.iterdir():
        if f.name != raw.name:
            f.unlink()
    onnx.save_model(m, str(raw), save_as_external_data=True,
                    all_tensors_to_one_file=True, location=TAG + "_weights.bin",
                    size_threshold=1024)
    total = sum(f.stat().st_size for f in dst.iterdir())
    print("[size]   {:.2f} GB".format(total / 1024 ** 3))


if __name__ == "__main__":
    main()
