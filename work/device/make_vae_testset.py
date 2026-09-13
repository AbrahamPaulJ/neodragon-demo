"""Build a held-out test set for the quantised VAE decoder.

Calibration used latents 0000-0049. Samples 0050+ were never seen by the
quantiser, so they are the honest set to measure deploy SNR on. References come
from the fp32 ONNX, matching how the paper defines deploy SNR ("between original
FP models vs. the deployed models").
"""

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
ONNX = ROOT / "work" / "onnx" / "vaedec_t1_qnn.onnx"
CALIB = ROOT / "work" / "calib" / "vae_dec"
OUT = ROOT / "work" / "device" / "vio_real"
OUT.mkdir(parents=True, exist_ok=True)

N_CALIB = 50


def main():
    held = sorted(CALIB.glob("latent_*.raw"))[N_CALIB:]
    print(f"held-out samples (not seen by the quantiser): {len(held)}")

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(ONNX), so, providers=["CPUExecutionProvider"])

    lines = []
    for i, p in enumerate(held):
        z = np.fromfile(p, dtype=np.float32).reshape(1, 16, 1, 40, 64)
        z.tofile(OUT / f"latent_{i}.raw")
        ref = sess.run(None, {"latent": z})[0].astype(np.float32)
        ref.tofile(OUT / f"vref_{i}.raw")
        lines.append(f"latent:=/data/local/tmp/nd/vio_real/latent_{i}.raw")
        print(f"  [{i}] {p.name}  z range [{z.min():.2f}, {z.max():.2f}] std {z.std():.2f}"
              f"  -> out [{ref.min():.3f}, {ref.max():.3f}]")

    (OUT / "input_list.txt").write_text("\n".join(lines) + "\n", newline="\n")
    print(f"\nwrote {len(held)} test cases to {OUT}")


if __name__ == "__main__":
    main()
