"""Generate device inputs for DistilT5 and the fp32 reference output.

Writes int32 .raw files matching the context binary's declared inputs
(input_ids [1,128] INT_32, attention_mask [1,128] INT_32) plus the
onnxruntime reference for prompt_embeds [1,128,4096] FLOAT_32.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "work" / "models" / "neodragon"
ONNX = ROOT / "work" / "onnx" / "distilt5_qnn.onnx"
OUT = ROOT / "work" / "device" / "io"
OUT.mkdir(parents=True, exist_ok=True)

SEQ = 128
PROMPTS = [
    "a red sports car driving along a coastal road at sunset",
    "a golden retriever puppy running through tall grass",
    "timelapse of clouds moving over a mountain range",
]


def main():
    from transformers import T5Tokenizer
    import onnxruntime as ort

    tok = T5Tokenizer.from_pretrained(MODEL / "tokenizer_3")
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(ONNX), so, providers=["CPUExecutionProvider"])

    lines = []
    for i, p in enumerate(PROMPTS):
        b = tok([p], padding="max_length", max_length=SEQ, truncation=True,
                add_special_tokens=True, return_tensors="np")
        ids = b["input_ids"].astype(np.int32)
        am = b["attention_mask"].astype(np.int32)

        ids.tofile(OUT / f"input_ids_{i}.raw")
        am.tofile(OUT / f"attention_mask_{i}.raw")

        ref = sess.run(None, {"input_ids": ids.astype(np.int64),
                              "attention_mask": am.astype(np.int64)})[0]
        ref.astype(np.float32).tofile(OUT / f"ref_{i}.raw")

        lines.append(
            f"input_ids:=/data/local/tmp/nd/io/input_ids_{i}.raw "
            f"attention_mask:=/data/local/tmp/nd/io/attention_mask_{i}.raw"
        )
        print(f"[{i}] tokens={int(am.sum())}/{SEQ}  ref max|.|={np.abs(ref).max():.4f}  "
              f"std={ref.std():.4f}  '{p[:40]}'")

    (OUT / "input_list.txt").write_text("\n".join(lines) + "\n", newline="\n")
    print(f"\nwrote {len(PROMPTS)} cases to {OUT}")
    print("input_list.txt:")
    print((OUT / "input_list.txt").read_text())


if __name__ == "__main__":
    main()
