"""Tokenised-prompt calibration for CLIP G, so it can be built W8A16 instead of FP16.

Why not just ship FP16 as Table 9 does: CLIP G is 694.7 M params, and at fp32 its weight
blob is 2.78 GB -- past the x86-64 PC32 relocation limit, so the model library will not
link (trap #37). Forcing `--float_bitwidth 16` makes it link but measured only **14.71 dB**
on the penultimate hidden state, against CLIP L's 60.75 dB. The error is spread evenly
across tokens (real prompt tokens ~25 dB, padding ~16 dB) rather than concentrated in
outliers, which is fp16 accumulating through 32 layers.

W8A16 sidesteps both problems: an 8-bit blob is ~0.69 GB so it links comfortably, and A16
activations carry more precision through the depth than fp16 does.

No GPU needed -- the only input is `input_ids`, so calibration is just tokenising prompts.

  usage: py -3.10 work/pipeline/make_clip_calib.py --prompts 300
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "src" / "neodragon"
MODEL = ROOT / "work" / "models" / "neodragon"
CAL = ROOT / "work" / "calib"
TOKENS = 77


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=int, default=300)
    ap.add_argument("--tag", default="clipg")
    ap.add_argument("--tokenizer", default="ssd_1b_tokenizer_2")
    a = ap.parse_args()

    from transformers import CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained(str(MODEL / a.tokenizer))
    prompts = [ln.strip() for ln in
               (REPO / "prompts" / "vbench_prompts.txt").read_text(
                   encoding="utf-8").splitlines() if ln.strip()]

    out = CAL / a.tag
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.raw"):
        f.unlink()
    host = str(out).replace("C:", "/mnt/c").replace("\\", "/")

    rows = []
    for i in range(a.prompts):
        ids = tok(prompts[i % len(prompts)], padding="max_length",
                  max_length=TOKENS, truncation=True,
                  return_tensors="np").input_ids.astype(np.int32)
        p = out / "input_ids_{:04d}.raw".format(i)
        np.ascontiguousarray(ids, dtype=np.int32).tofile(p)
        rows.append("input_ids:={}/{}".format(host, p.name))

    (out / "calib_list_host.txt").write_text("\n".join(rows) + "\n", newline="\n")
    (out / "dims.txt").write_text(
        "--input_dim input_ids 1,{}\n".format(TOKENS), newline="\n")
    print("wrote {} tokenised prompts -> {}".format(a.prompts, out))
    print("  NOTE: input_ids are INT32. A quantised graph reads raws natively, so"
          " qnn-net-run must OMIT --use_native_input_files for the W8A16 build"
          " (trap #8) -- the opposite of the FP16 build.")


if __name__ == "__main__":
    main()
