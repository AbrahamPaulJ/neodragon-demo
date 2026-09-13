"""Precompute every deterministic structure the video path needs, for the Android app.

The AR video loop's hard parts are not arithmetic -- they are bookkeeping: the pyramid
envelope each stage pads to, which history slices form the past conditions, the temporal
RoPE ids, and the attention mask layout. All of it depends ONLY on (unit_index, stage),
never on the prompt, the seed or any latent value. So it is computed once here, against
the real reference implementation, and shipped as an asset.

That is deliberate risk management. Porting `_prepare_past_condition_latents`,
`align_to_envelope`, `_prepare_temporal_rope_ids` and `temp_rope_embed` to Kotlin would
be four chances to introduce a silent off-by-one in code whose output looks plausible
either way. Precomputing them leaves Kotlin doing only tensor arithmetic it can be
checked on.

The ONE prompt-dependent part of the mask is which of the 128 text positions are valid,
so the mask is rebuilt on-device from `order` + `image_valid` + the text mask -- see
VideoStructure.kt. Shipping the masks themselves would be 214 MB.

Outputs, into app/src/main/assets/:
    mmdit_temb.ndw        time_text_embed weights (host-side, see host_temb)
    video_structure.ndw   per (unit,stage): rope_cos, rope_sin, order, image_valid
    video_structure.json  envelopes, pad plans, history gather recipes, constants

  usage: py -3.10 work/export/export_video_structure.py
"""

import json
import math
import struct
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "src" / "neodragon"
MODEL = ROOT / "work" / "models" / "neodragon"
ASSETS = ROOT / "work" / "android" / "app" / "src" / "main" / "assets"

sys.path.insert(0, str(ROOT / "work" / "export"))
sys.path.insert(0, str(ROOT / "work" / "audit"))
sys.path.insert(0, str(REPO))
_pkg = types.ModuleType("neodragon")
_pkg.__path__ = [str(REPO / "neodragon")]
sys.modules["neodragon"] = _pkg

from export_app_assets import write_ndw                              # noqa: E402
from export_mmdit_stage import align_to_envelope, envelope_for       # noqa: E402
from mmdit_shapes import (FRAMES_PER_UNIT, NUM_LATENT_UNITS, NUM_STAGES,  # noqa: E402
                          PATCH, START_UNIT, TEXT_TOKENS, past_condition_shapes,
                          pyramid_hw)


def history_recipe(unit_index, stage):
    """`_prepare_past_condition_latents` expressed as history-pyramid slices.

    Returns [(level, t_begin, t_end), ...] in the order the pipeline concatenates them,
    against a history tensor of `unit_index` latent frames. Mirrors
    generation_utils.py:178-227 exactly, with the negative indices resolved.
    """
    if unit_index == 0:
        return []
    T = unit_index * FRAMES_PER_UNIT
    out = [(stage, T - FRAMES_PER_UNIT, T)]          # last_cond_latent

    cur_unit_ptx = 1
    cur_stage = stage
    while cur_unit_ptx < unit_index:
        cur_stage = max(cur_stage - 1, 0)
        if cur_stage == 0:
            break
        cur_unit_ptx += 1
        out.append((cur_stage,
                    T - cur_unit_ptx * FRAMES_PER_UNIT,
                    T - (cur_unit_ptx - 1) * FRAMES_PER_UNIT))
    if cur_stage == 0 and cur_unit_ptx < unit_index:
        out.append((0, 0, T - cur_unit_ptx * FRAMES_PER_UNIT))

    return list(reversed(out))


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)

    print("[load] PyramidMMDiT on CPU (needed only for temp_rope_embed and "
          "time_text_embed) ...")
    from neodragon.pyramid_mmdit import PyramidMMDiT
    dit = PyramidMMDiT.from_pretrained(
        str(MODEL / "diffusion_transformer_320p"), torch_dtype=torch.float32).eval()

    # ---- host-side conditioning MLPs ------------------------------------------
    print("[1/3] time_text_embed weights ...")
    tte = dit.time_text_embed
    ten = {}
    for tag, mod in (("timestep_embedder", tte.timestep_embedder),
                     ("text_embedder", tte.text_embedder)):
        for i in (1, 2):
            lin = getattr(mod, "linear_{}".format(i))
            ten["{}.linear_{}.weight".format(tag, i)] = lin.weight.detach().numpy()
            ten["{}.linear_{}.bias".format(tag, i)] = lin.bias.detach().numpy()
    for k, v in ten.items():
        print("      {:38s} {}".format(k, tuple(v.shape)))
    write_ndw(ASSETS / "mmdit_temb.ndw", ten)

    # ---- per (unit, stage) structure ------------------------------------------
    print("[2/3] RoPE tables + token layout for all "
          "{} calls ...".format((NUM_LATENT_UNITS - START_UNIT) * NUM_STAGES))
    struct_t = {}
    meta = {}
    for unit in range(START_UNIT, NUM_LATENT_UNITS):
        for stage in range(NUM_STAGES):
            h, w = pyramid_hw(stage)
            real = past_condition_shapes(unit, stage) + [(FRAMES_PER_UNIT, h, w)]
            env = envelope_for(stage)
            plan = align_to_envelope(real, env)

            # temporal ids for the REAL shapes, in real order (start_time_stamp runs
            # over the real sequence, which is what keeps front-padding exact)
            real_ids, start = [], 0
            for (t, rh, rw) in real:
                real_ids.append(dit._prepare_temporal_rope_ids(
                    1, t, rh // PATCH, rw // PATCH, torch.device("cpu"),
                    start_time_stamp=start))
                start += t

            ids, valid = [], []
            for slot, (t, eh, ew) in zip(plan, env):
                ri, n_pad = slot
                n_spatial = (eh // PATCH) * (ew // PATCH)
                if ri is None:
                    ids.append(torch.zeros(1, t * n_spatial, 1))
                    valid.append(torch.zeros(t * n_spatial, dtype=torch.bool))
                    continue
                if n_pad:
                    ids.append(torch.zeros(1, n_pad * n_spatial, 1))
                    valid.append(torch.zeros(n_pad * n_spatial, dtype=torch.bool))
                ids.append(real_ids[ri])
                valid.append(torch.ones(real_ids[ri].shape[1], dtype=torch.bool))

            image_ids = torch.cat(ids, dim=1)
            image_valid = torch.cat(valid, dim=0)
            text_ids = torch.zeros(1, TEXT_TOKENS, 1)
            input_ids = torch.cat([text_ids, image_ids], dim=1)

            freqs = dit.temp_rope_embed(input_ids)
            cos = freqs[0, ..., 0, 0].float().numpy()        # [S,1,D/2]
            sin = freqs[0, ..., 1, 0].float().numpy()

            key = "u{}s{}".format(unit, stage)
            struct_t[key + ".rope_cos"] = cos
            struct_t[key + ".rope_sin"] = sin
            struct_t[key + ".order"] = input_ids[0, :, 0].float().numpy()
            struct_t[key + ".image_valid"] = image_valid.float().numpy()

            meta[key] = {
                "stage": stage,
                "unit": unit,
                "real_shapes": [list(map(int, s)) for s in real],
                "env_shapes": [list(map(int, s)) for s in env],
                # (envelope slot) -> (index into the latent list, frames of front pad)
                "pad_plan": [[-1 if p[0] is None else int(p[0]), int(p[1])]
                             for p in plan],
                "history": [list(map(int, r)) for r in history_recipe(unit, stage)],
                "seq_len": int(input_ids.shape[1]),
                "n_image_tokens": int(image_ids.shape[1]),
            }
            print("      {}  S={:5d}  env={}  real={}".format(
                key, input_ids.shape[1], len(env), len(real)))

    write_ndw(ASSETS / "video_structure.ndw", struct_t)

    # ---- scalars --------------------------------------------------------------
    print("[3/3] scheduler + VAE constants ...")
    from neodragon.pyramid_scheduler import PyramidFlowMatchEulerDiscreteScheduler
    from neodragon.utils.generation_utils import (VAE_SCALE_FACTOR, VAE_SHIFT_FACTOR,
                                                  VAE_VIDEO_SCALE_FACTOR,
                                                  VAE_VIDEO_SHIFT_FACTOR,
                                                  DEFAULT_PROMPT_MODIFIER)
    sched = PyramidFlowMatchEulerDiscreteScheduler()

    # With num_inference_steps=[1,1,1] each stage is ONE Euler step from sigma 1 to 0,
    # so the only per-stage values that matter are the timestep_ratio fed to the DiT and
    # the original start sigma used by the upsample correction.
    stage_timestep = [float(sched.get_stage_timesteps(1, s)[0]) for s in range(NUM_STAGES)]
    orig_start = [float(sched.orig_start_sigmas[s]) for s in range(NUM_STAGES)]

    gamma = float(sched.config.gamma)
    cov = np.eye(4) * (1 + gamma) - np.ones((4, 4)) * gamma
    chol = np.linalg.cholesky(cov)          # block-noise sampler, as a constant

    cfg = {
        "num_stages": NUM_STAGES,
        "start_unit": START_UNIT,
        "num_latent_units": NUM_LATENT_UNITS,
        "frames_per_unit": FRAMES_PER_UNIT,
        "text_tokens": TEXT_TOKENS,
        "patch": PATCH,
        "lat_c": int(dit.config.in_channels),
        "video_height": 320,
        "video_width": 512,
        "num_frames": 49,
        "stage_timestep": stage_timestep,
        "orig_start_sigmas": orig_start,
        "gamma": gamma,
        "block_noise_chol": [[float(x) for x in row] for row in chol],
        "vae_scale_factor": VAE_SCALE_FACTOR,
        "vae_shift_factor": VAE_SHIFT_FACTOR,
        "vae_video_scale_factor": VAE_VIDEO_SCALE_FACTOR,
        "vae_video_shift_factor": VAE_VIDEO_SHIFT_FACTOR,
        "prompt_modifier": DEFAULT_PROMPT_MODIFIER,
        "mask_fill": -100.0,
        "pooled_projection_dim": int(dit.config.pooled_projection_dim),
        "inner_dim": int(tte.timestep_embedder.linear_1.out_features),
        "calls": meta,
    }
    (ASSETS / "video_structure.json").write_text(json.dumps(cfg, indent=1))
    print("  stage timesteps {}".format(["%.2f" % v for v in stage_timestep]))
    print("  orig start sigmas {}".format(["%.4f" % v for v in orig_start]))
    print("  wrote video_structure.json")

    total = sum(p.stat().st_size for p in ASSETS.iterdir() if p.is_file())
    print("")
    print("assets total {:.2f} MB".format(total / 1048576))


if __name__ == "__main__":
    sys.exit(main())
