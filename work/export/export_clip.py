"""CLIP L and CLIP G (SSD1B text encoders) -> ONNX, for FP16 deployment.

Table 9 deploys both in **FP16, no calibration** -- same as DistilT5, which ships at
49.04 dB. So there is no calibration set here and the convert script passes no
--input_list at all; the converter emits a float graph and HTP runs it in fp16 (trap #3).

The two differ in what the pipeline consumes:
  * CLIP L (`ssd_1b_text_encoder`, CLIPTextModel) -> hidden_states[-2], [1,77,768]
  * CLIP G (`ssd_1b_text_encoder_2`, CLIPTextModelWithProjection) -> hidden_states[-2],
    [1,77,1280], PLUS text_embeds [1,1280] which becomes the UNet's `text_embeds`
    added condition.

SDXL concatenates the two penultimate hidden states to [1,77,2048] -- exactly the
`encoder_hidden_states` the converted UNet takes. That concat is on the LAST axis of a
rank-3 tensor with 77 rows, far below the ~320-row threshold where QNN's multi-input
concat interleaves (session-6 trap), and it is done on the HOST anyway.

Trap #3 applies: HTP runs float graphs in fp16 with no fp32 upcast in the layer norm.
DistilT5 needed residual scaling to avoid NaN. CLIP's activations are far smaller, but
the export prints max|activation| so the risk is visible rather than assumed.

  usage: py -3.10 work/export/export_clip.py --export
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
DEV = ROOT / "work" / "device"
sys.path.insert(0, str(REPO))
_pkg = types.ModuleType("neodragon")
_pkg.__path__ = [str(REPO / "neodragon")]
sys.modules["neodragon"] = _pkg

DEVICE_ROOT = "/data/local/tmp/nd"
TOKENS = 77


class ClipL(nn.Module):
    """CLIPTextModel -> the PENULTIMATE hidden state, which is what SDXL uses."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids):
        out = self.m(input_ids, output_hidden_states=True)
        return out.hidden_states[-2]


class ClipLP(nn.Module):
    """CLIP L *WithProjection* -> penultimate hidden state AND the pooled embedding.

    The video path needs something the first-frame path never did. `TextEncoderBundle`
    loads `text_encoder` (not `ssd_1b_text_encoder`) as CLIPTextModelWithProjection and
    keeps only `[0]`, i.e. `text_embeds`, concatenating it with CLIP G's to form the
    MMDiT's 2048-d `pooled_projections`. The shipping `clipl` graph is a plain
    CLIPTextModel and has no projection head, so it cannot supply that vector.

    The two checkpoints hold the same weights at different precisions (bf16 here vs fp16
    for the SSD1B copy; max abs difference 0.027 across all 517 shared tensors), so this
    is an added output, not a different model.
    """

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids):
        out = self.m(input_ids, output_hidden_states=True)
        return out.hidden_states[-2], out.text_embeds


class ClipG(nn.Module):
    """CLIPTextModelWithProjection -> penultimate hidden state AND the pooled embedding."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids):
        out = self.m(input_ids, output_hidden_states=True)
        return out.hidden_states[-2], out.text_embeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    a = ap.parse_args()

    from transformers import (CLIPTextModel, CLIPTextModelWithProjection,
                              CLIPTokenizer)

    tok = CLIPTokenizer.from_pretrained(str(MODEL / "ssd_1b_tokenizer"))
    ids = tok("a cinematic photo of a mountain lake at sunrise",
              padding="max_length", max_length=TOKENS, truncation=True,
              return_tensors="pt").input_ids

    specs = [
        ("clipl", ClipL, CLIPTextModel, "ssd_1b_text_encoder", ["hidden"]),
        ("clipg", ClipG, CLIPTextModelWithProjection, "ssd_1b_text_encoder_2",
         ["hidden", "text_embeds"]),
        ("cliplp", ClipLP, CLIPTextModelWithProjection, "text_encoder",
         ["hidden", "text_embeds"]),
    ]
    only = os.environ.get("ONLY")
    if only:
        specs = [s for s in specs if s[0] == only]
    for tag, wrap, cls, sub, outs in specs:
        print("")
        print("[load] {} ...".format(sub))
        # These ship ONLY as model.fp16.safetensors, so the fp16 variant must be named
        # explicitly -- same convention as run_reference.py's pick(). Loaded into fp32
        # for export; the deployed graph is float and HTP runs it in fp16 anyway.
        d = MODEL / sub
        kw = {'variant': 'fp16'} if (d / 'model.fp16.safetensors').exists() else {}
        m = cls.from_pretrained(str(d), torch_dtype=torch.float32, **kw).eval()
        net = wrap(m).eval()
        with torch.no_grad():
            r = net(ids)
        rs = r if isinstance(r, tuple) else (r,)
        print("       {:.1f} M params  ->  {}".format(
            sum(p.numel() for p in m.parameters()) / 1e6,
            ", ".join(str(tuple(x.shape)) for x in rs)))
        peak = max(float(x.abs().max()) for x in rs)
        print("       max|activation| = {:.1f}   (fp16 max is 65504 -- trap #3)".format(peak))
        assert peak < 1e4, "activations too large for a safe fp16 graph"

        if not a.export:
            continue
        dst = OUT / tag
        dst.mkdir(parents=True, exist_ok=True)
        raw = dst / (tag + "_raw.onnx")
        torch.onnx.export(net, (ids,), str(raw), input_names=["input_ids"],
                          output_names=outs, opset_version=17,
                          do_constant_folding=True, dynamo=False)
        print("[onnx]   {:.0f} MB -> {}".format(raw.stat().st_size / 1e6, raw))

        io = DEV / ("io_" + tag)
        io.mkdir(parents=True, exist_ok=True)
        for f in io.glob("*.raw"):
            f.unlink()
        # input_ids are INTEGER -- a float graph with an integer input needs
        # --use_native_input_files at run time (trap #8).
        np.ascontiguousarray(ids.numpy(), dtype=np.int32).tofile(io / "input_ids_0000.raw")
        for nm, x in zip(outs, rs):
            np.ascontiguousarray(x.numpy(), dtype=np.float32).tofile(
                io / "{}ref_0000.raw".format(nm))
        (io / "input_list.txt").write_text(
            "input_ids:={}/io_{}/input_ids_0000.raw\n".format(DEVICE_ROOT, tag),
            newline="\n")
        (io / "dims.txt").write_text(
            "--input_dim input_ids 1,{}\n".format(TOKENS), newline="\n")
        print("[test]   1 sample + fp32 refs -> {}".format(io))

    if not a.export:
        print("")
        print("pass --export to write the ONNX files")


if __name__ == "__main__":
    main()
