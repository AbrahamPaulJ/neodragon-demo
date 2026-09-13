"""Device inputs + fp32 reference for the TAEHV decoder (T=1)."""

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
ONNX = ROOT / "work" / "onnx" / "vaedec_t1_qnn.onnx"
OUT = ROOT / "work" / "device" / "vio"
OUT.mkdir(parents=True, exist_ok=True)

T, N_CASES = 1, 3


def main():
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(ONNX), so, providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(0)
    lines = []
    for i in range(N_CASES):
        # decoder input is Clamp()ed to +-3, so N(0,1) latents exercise the
        # full useful range without saturating the tanh
        z = rng.standard_normal((1, 16, T, 40, 64)).astype(np.float32)
        z.tofile(OUT / f"latent_{i}.raw")
        ref = sess.run(None, {"latent": z})[0].astype(np.float32)
        ref.tofile(OUT / f"vref_{i}.raw")
        lines.append(f"latent:=/data/local/tmp/nd/vio/latent_{i}.raw")
        print(f"[{i}] z{z.shape} -> out{ref.shape}  "
              f"min={ref.min():.4f} max={ref.max():.4f} std={ref.std():.4f}")

    (OUT / "input_list.txt").write_text("\n".join(lines) + "\n", newline="\n")
    print(f"\nwrote {N_CASES} cases to {OUT}")


if __name__ == "__main__":
    main()
