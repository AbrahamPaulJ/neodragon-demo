"""Held-out device test set for the 2-D VAE encoder.

The 16 inputs already exist -- work/pipeline/capture_first_frames.py --split test
wrote real SSD1B first frames from showcase prompts. Calibration comes from a
DISJOINT set (vbench prompts, work/calib/vae_enc), so the deploy SNR is honest.

This adds the fp32 ONNX references and the device input_list, and can also emit
the NHWC copies the `vaeencn` build needs.

Measurement note that matters. `moments` is [mean | logvar] on the channel axis,
and logvar sits near -30 while mean has std ~1.7. Whole-tensor SNR is therefore
dominated by the logvar half -- which is nearly constant and trivially easy to
reproduce -- and would flatter the build. The number that decides whether the
encoder is good enough is SNR on the MEAN half, because
generation_utils.py:531-534 does

    image_latent = mean + exp(0.5*logvar) * noise      # exp(0.5*-30) ~ 3e-7
    image_latent = (image_latent - SHIFT) * SCALE

so what reaches the DiT is the mean, affinely rescaled. compare_enc.py reports
mean-half SNR as the headline and the other two for transparency.

  usage: py -3.10 work/device/make_enc_io.py [--nhwc]
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
ONNX = ROOT / "work" / "onnx" / "vaeenc2d_qnn.onnx"
EIO = ROOT / "work" / "device" / "eio"
CALIB = ROOT / "work" / "calib" / "vae_enc"

IMG_SHAPE = (1, 3, 320, 512)
DEVICE_DIR = "/data/local/tmp/nd/eio"


def to_nhwc(src, dst):
    """[1,C,H,W] float32 raw -> [1,H,W,C]. The image's native host layout."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    a = np.fromfile(src, dtype=np.float32).reshape(IMG_SHAPE)
    np.ascontiguousarray(a.transpose(0, 2, 3, 1)).tofile(dst)


def main():
    nhwc = "--nhwc" in sys.argv
    imgs = sorted(EIO.glob("image_*.raw"))
    assert imgs, "no test images -- run capture_first_frames.py --split test first"

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(ONNX), so, providers=["CPUExecutionProvider"])

    lines = []
    prev = None
    for c, p in enumerate(imgs):
        x = np.fromfile(p, dtype=np.float32).reshape(IMG_SHAPE)
        out = sess.run(None, {"image": x})[0]
        out.astype(np.float32).tofile(EIO / "mref_{}.raw".format(c))
        lines.append("image:={}/{}".format(DEVICE_DIR, p.name))
        mean, logvar = out[:, :16], out[:, 16:]
        # trap #8: two different inputs must give two different outputs.
        assert prev is None or np.abs(out - prev).max() > 1e-3, \
            "case {} reproduced the previous output -- inputs are not reaching the graph".format(c)
        prev = out
        print("  [{:2d}] {}  mean std={:.3f} range=[{:.2f},{:.2f}]  "
              "logvar range=[{:.1f},{:.1f}]".format(
                  c, p.name, mean.std(), mean.min(), mean.max(),
                  logvar.min(), logvar.max()))

    (EIO / "input_list.txt").write_text("\n".join(lines) + "\n", newline="\n")
    print("")
    print("wrote {} references + input_list.txt to {}".format(len(imgs), EIO))

    if nhwc:
        out_dir = CALIB / "nhwc"
        n = 0
        for p in sorted(CALIB.glob("image_*.raw")):
            to_nhwc(p, out_dir / p.name)
            n += 1
        for p in imgs:
            to_nhwc(p, EIO / "nhwc" / p.name)
        (EIO / "nhwc" / "input_list.txt").write_text(
            "\n".join(l.replace(DEVICE_DIR, DEVICE_DIR + "/nhwc") for l in lines) + "\n",
            newline="\n")
        (out_dir / ".layout_nhwc").write_text("permuted by make_enc_io.py --nhwc\n")
        print("wrote {} NHWC calibration copies to {} and {} test copies to {}".format(
            n, out_dir, len(imgs), EIO / "nhwc"))


if __name__ == "__main__":
    main()
