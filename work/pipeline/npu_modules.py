"""NPU-backed stand-ins for the reference pipeline's heavy modules.

The E2E driver keeps `generation_utils.py`'s orchestration untouched -- the pyramid
scheduler, the history pyramid, the per-stage sigmas, `_upsample_pyramidal_latent` -- and
swaps only the modules. That way the first end-to-end run answers the one question the
port has not answered ("what does quantisation do across the whole AR loop") without also
risking a reimplementation bug in the scheduler.

Each stand-in exposes exactly the call signature the pipeline already uses, does the
host-side preamble the exported graph expects, and routes the heavy tensor work to a
context binary on the phone.
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "work" / "export"))
sys.path.insert(0, str(ROOT / "work" / "audit"))

from npu_graph import NpuGraph                                   # noqa: E402


class NpuContextAdapter:
    """`context_adapter(prompt_embeds)` -> [1,128,1536]. FP16 graph, 39.36 dB."""

    def __init__(self, adb=None):
        self.g = NpuGraph("ctxadaptfp16", ["prompt_embeds"], ["context"], adb=adb,
                          tag="io_ctxadapt")

    def __call__(self, prompt_embeds):
        out = self.g.run(prompt_embeds=prompt_embeds.detach().float().cpu().numpy())
        return torch.from_numpy(out["context"]).reshape(prompt_embeds.shape[0], -1, 1536)


class NpuPyramidMMDiT:
    """Drop-in for `PyramidMMDiT.__call__` across all three pyramid stages.

    The pipeline calls it as
        dit(sample=[latents], encoder_hidden_states=..., encoder_attention_mask=...,
            pooled_projections=..., timestep_ratio=...)
    and expects a tuple whose first element is the noise prediction.

    The stage is inferred from the CURRENT latent's spatial size, which is what
    `pyramid_hw()` defines; the latents are then padded to that stage's envelope and the
    mask / RoPE / temb are rebuilt exactly as `make_mmdit_io.py` does for calibration, so
    the graph sees precisely the distribution it was calibrated on.
    """

    STAGE_BIN = {0: "mmdit_s0g", 1: "mmdit_s1f", 2: "mmdit_s2f"}

    def __init__(self, dit, adb=None):
        from export_mmdit_stage import envelope_for
        from mmdit_shapes import pyramid_hw
        self.dit = dit                      # kept ONLY for the host-side preamble
        self.env = {s: envelope_for(s) for s in range(3)}
        self.hw = {pyramid_hw(s): s for s in range(3)}
        self.graphs = {}
        self.adb = adb
        self.calls = 0

    # `generate()` reads attributes off the dit besides calling it (config for the
    # latent-channel count and patch size, dtype/device for the noise it allocates).
    # Forward anything we do not define to the real module so the stand-in is a true
    # drop-in and the pipeline needs no edits.
    def __getattr__(self, name):
        return getattr(self.__dict__["dit"], name)

    def _graph(self, stage):
        if stage not in self.graphs:
            env = self.env[stage]
            names = (["encoder_hidden_states", "temb_act", "attn_mask",
                      "rope_cos", "rope_sin"]
                     + ["latent_{}".format(i) for i in range(len(env))])
            self.graphs[stage] = NpuGraph(self.STAGE_BIN[stage], names, ["noise_pred"],
                                          adb=self.adb,
                                          tag="io_mmdit_s{}".format(stage))
        return self.graphs[stage]

    def __call__(self, sample, encoder_hidden_states, encoder_attention_mask,
                 pooled_projections, timestep_ratio, **kw):
        from export_mmdit_stage import (host_temb, pad_latents,
                                        stage_conditioning_padded)
        lats = sample[0] if isinstance(sample[0], (list, tuple)) else sample
        shapes = [(int(t.shape[2]), int(t.shape[3]), int(t.shape[4])) for t in lats]
        stage = self.hw[(shapes[-1][1], shapes[-1][2])]
        env = self.env[stage]

        eam = encoder_attention_mask.detach().float().cpu()
        mask, cos, sin = stage_conditioning_padded(self.dit, shapes, env, eam)
        padded = pad_latents([t.detach().float().cpu() for t in lats], shapes, env)
        temb = host_temb(self.dit,
                         timestep_ratio.detach().float().cpu(),
                         pooled_projections.detach().float().cpu())

        feed = {
            "encoder_hidden_states": encoder_hidden_states.detach().float().cpu().numpy(),
            "temb_act": temb.numpy(),
            "attn_mask": mask.numpy(),
            "rope_cos": cos.numpy(),
            "rope_sin": sin.numpy(),
        }
        for i, la in enumerate(padded):
            feed["latent_{}".format(i)] = la.numpy()

        out = self._graph(stage).run(**feed)
        t, h, w = env[-1]
        pred = torch.from_numpy(out["noise_pred"]).reshape(1, 16, t, h, w)
        self.calls += 1
        return (pred.to(lats[-1].device, lats[-1].dtype),)


class NpuVaeDecoder:
    """Causal video VAE decoder, 8 frames per invocation, explicit MemBlock state.

    The shipping build carries its 9 MemBlock states as NHWC (`.layout_nhwc`); the
    streaming rewrite is what took this module from 6.10 s to 100.18 ms per invocation.
    Wiring it needs the state plumbing from export_vae_decoder_stream.py, so it is left
    for the driver to fill in once the MMDiT loop is proven end to end.
    """

    def __init__(self, *a, **kw):
        raise NotImplementedError(
            "wire the 9 MemBlock states (see work/export/export_vae_decoder_stream.py "
            "and work/device/permute_states_nhwc.py) before using this")
