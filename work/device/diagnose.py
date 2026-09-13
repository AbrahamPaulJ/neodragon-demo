"""Where does input-dependence get lost: ONNX export, or QNN conversion?

Runs both ONNX graphs on CPU with the same three device inputs and checks
(a) that outputs differ per prompt and (b) that they match the reference.
"""

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "work" / "device"
ONNX = ROOT / "work" / "onnx"
SHAPE = (1, 128, 4096)


def run_all(path):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    itype = {i.name: i.type for i in sess.get_inputs()}
    outs = []
    for i in range(3):
        ids = np.fromfile(W / "io" / f"input_ids_{i}.raw", dtype=np.int32).reshape(1, 128)
        am = np.fromfile(W / "io" / f"attention_mask_{i}.raw", dtype=np.int32).reshape(1, 128)
        if "int64" in itype["input_ids"]:
            ids = ids.astype(np.int64)
        if "int64" in itype["attention_mask"]:
            am = am.astype(np.int64)
        outs.append(sess.run(None, {"input_ids": ids, "attention_mask": am})[0])
    return outs, itype


def check(label, path):
    print(f"\n--- {label} ---")
    print(f"    {path.name}")
    outs, itype = run_all(path)
    print(f"    declared input types: {itype}")
    for i, o in enumerate(outs):
        ref = np.fromfile(W / "io" / f"ref_{i}.raw", dtype=np.float32).reshape(SHAPE)
        d = ref - o
        snr = 20 * np.log10(np.linalg.norm(ref) / np.linalg.norm(d)) if np.linalg.norm(d) else np.inf
        print(f"    case {i}: max|.|={np.abs(o).max():.4f}  vs ref SNR={snr:>7.2f} dB")
    d01 = np.abs(outs[0] - outs[1]).max()
    d02 = np.abs(outs[0] - outs[2]).max()
    print(f"    input-dependent? case0-case1 max|d|={d01:.6f}  case0-case2 max|d|={d02:.6f}"
          f"   -> {'YES' if d01 > 1e-5 else 'NO -- graph ignores input'}")


def main():
    print("=" * 74)
    print("Diagnostic: input-dependence and accuracy, CPU onnxruntime")
    print("=" * 74)
    check("unscaled (used to build the refs)", ONNX / "distilt5_qnn.onnx")
    check("residual-scaled S=16", ONNX / "distilt5_scaled_qnn.onnx")
    check("scaled, before mask fix + fold", ONNX / "distilt5_scaled_raw.onnx")


if __name__ == "__main__":
    main()
