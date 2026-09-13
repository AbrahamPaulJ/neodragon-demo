"""Held-out device test set for the streaming TAEHV decoder.

Samples 50-55 are prompt 7 frames 1-6: real pipeline latents with real non-zero
MemBlock states, never seen by the quantiser (which takes the first 50). Same
held-out definition as the T=1 build, so the deploy-SNR numbers are comparable
with `docs/phase4-vae-decoder.md`'s 31.55 dB.

The inputs already exist -- work/device/make_stream_calib.py wrote all 56 samples.
This only adds the fp32 ONNX references and a device input_list restricted to the
held-out range.

  usage: py -3.10 work/device/make_stream_testset.py
"""

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
ONNX = ROOT / "work" / "onnx" / "vaedec_stream_qnn.onnx"
CALIB = ROOT / "work" / "calib" / "vae_dec_stream"
OUT = ROOT / "work" / "device" / "vio_stream"

FIRST, N_CASES, N_STATES = 50, 6, 9
DEVICE_DIR = "/data/local/tmp/nd/vio_stream"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(ONNX), so, providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]

    lines = []
    for c in range(N_CASES):
        idx = FIRST + c
        feed = {"latent": np.fromfile(CALIB / f"latent_{idx:04d}.raw",
                                      dtype=np.float32).reshape(1, 16, 40, 64)}
        parts = [f"latent:={DEVICE_DIR}/latent_{idx:04d}.raw"]
        for k, inp in enumerate(sess.get_inputs()[1:]):
            a = np.fromfile(CALIB / f"state_{k}_{idx:04d}.raw", dtype=np.float32)
            feed[inp.name] = a.reshape(tuple(inp.shape))
            parts.append(f"state_{k}:={DEVICE_DIR}/state_{k}_{idx:04d}.raw")
        lines.append(" ".join(parts))

        out = sess.run(None, {n: feed[n] for n in in_names})
        out[0].astype(np.float32).tofile(OUT / f"vref_{c}.raw")
        print(f"  [{c}] sample {idx}  video{out[0].shape}  "
              f"min={out[0].min():.4f} max={out[0].max():.4f} "
              f"|state|max={max(abs(a).max() for a in out[1:]):.3f}")

    (OUT / "input_list.txt").write_text("\n".join(lines) + "\n", newline="\n")
    print(f"\nwrote {N_CASES} references + input_list.txt to {OUT}")
    print(f"the .raw INPUTS live in {CALIB} -- push those too "
          f"(only samples {FIRST}-{FIRST+N_CASES-1} are needed)")


if __name__ == "__main__":
    main()
