"""Calibration + held-out test data for the streaming TAEHV decoder.

The streaming graph takes 9 MemBlock states as explicit inputs, so calibration
needs realistic state ranges as well as realistic latents. Zeros would be the
state-input equivalent of the synthetic-N(0,1) mistake in trap #4: only correct
for invocation 1, and roughly an order of magnitude too narrow for every
invocation after it.

So replay each prompt's 7 latent frames through the graph in temporal order and
dump the actual (latent, state_0..state_8) tuple seen at every step. Frame 0 of
each prompt legitimately has zero state, so the zero case appears at its true 1-in-7
frequency rather than being assumed or excluded.

`work/calib/vae_dec/latent_XXXX.raw` is prompt-major, 7 frames per prompt, 8
prompts = 56 samples (see work/pipeline/run_reference.py). The converter takes the
first 50 to match paper Table 9; samples 50-55 (prompt 7, frames 1-6, all with
non-zero state) stay held out for deploy-SNR measurement.

NOTE ON DISK: the 9 states are 13.76 M floats = 55 MB per sample in fp32, so a
full 56-sample set is ~3.1 GB. Delete work/calib/vae_dec_stream once the context
binary is built -- the converter only reads it once.

  usage: py -3.10 work/device/make_stream_calib.py [--limit N]
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
ONNX = ROOT / "work" / "onnx" / "vaedec_stream_qnn.onnx"
SRC = ROOT / "work" / "calib" / "vae_dec"
OUT = ROOT / "work" / "calib" / "vae_dec_stream"

FRAMES_PER_PROMPT = 7
N_STATES = 9
# qnn-net-run reads these paths on the DEVICE; qnn-onnx-converter reads them on the
# HOST (under WSL). They are different filesystems, so emit both lists -- feeding
# device paths to the converter is a silent "file not found" per sample.
DEVICE_DIR = "/data/local/tmp/nd/vio_stream"
def _wsl(p):
    """C:/x/y -> /mnt/c/x/y, so the list is readable by the converter under WSL."""
    p = p.resolve()
    return f"/mnt/{p.drive[0].lower()}{p.as_posix()[2:]}" if p.drive else p.as_posix()


HOST_DIR = os.environ.get("ND_WSL_CALIB_DIR", _wsl(OUT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N samples (disk control)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(ONNX), so, providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]
    state_shapes = [tuple(i.shape) for i in sess.get_inputs()[1:]]
    assert len(state_shapes) == N_STATES, state_shapes

    latents = sorted(SRC.glob("latent_*.raw"))
    print(f"[src] {len(latents)} latents in {SRC}")
    assert len(latents) % FRAMES_PER_PROMPT == 0, len(latents)

    lines = []
    host_lines = []
    stats = []
    n = 0
    for start in range(0, len(latents), FRAMES_PER_PROMPT):
        # fresh video -> zero state, exactly as the device caller will start
        states = [np.zeros(s, dtype=np.float32) for s in state_shapes]
        for path in latents[start:start + FRAMES_PER_PROMPT]:
            if args.limit is not None and n >= args.limit:
                break
            z = np.fromfile(path, dtype=np.float32).reshape(1, 16, 40, 64)

            z.tofile(OUT / f"latent_{n:04d}.raw")
            names = [f"latent:=latent_{n:04d}.raw"]
            for k, s in enumerate(states):
                s.tofile(OUT / f"state_{k}_{n:04d}.raw")
                names.append(f"state_{k}:=state_{k}_{n:04d}.raw")
            lines.append(" ".join(p.replace(":=", f":={DEVICE_DIR}/") for p in names))
            host_lines.append(" ".join(p.replace(":=", f":={HOST_DIR}/") for p in names))

            stats.append((n, float(np.abs(z).max()),
                          max(float(np.abs(s).max()) for s in states)))
            out = sess.run(None, {n_: v for n_, v in
                                  zip(in_names, [z] + states)})
            states = [np.ascontiguousarray(a, dtype=np.float32) for a in out[1:]]
            n += 1
        if args.limit is not None and n >= args.limit:
            break

    (OUT / "input_list.txt").write_text("\n".join(lines) + "\n", newline="\n")
    (OUT / "calib_list_host.txt").write_text("\n".join(host_lines) + "\n", newline="\n")

    print(f"\n[out] {n} samples -> {OUT}")
    print(f"[out] input_list.txt      device paths, for qnn-net-run")
    print(f"[out] calib_list_host.txt host paths,   for qnn-onnx-converter")
    print(f"[out] {len(lines)} lines, {N_STATES+1} inputs each")

    # The point of the exercise: show that state range is NOT approximated by zero.
    print("\n[range] |latent|max and |state|max, first frame of each prompt vs rest")
    firsts = [s for s in stats if s[0] % FRAMES_PER_PROMPT == 0]
    rest = [s for s in stats if s[0] % FRAMES_PER_PROMPT != 0]
    for label, group in (("frame 0 (zero state)", firsts), ("frames 1-6", rest)):
        if not group:
            continue
        lat = max(g[1] for g in group)
        st = max(g[2] for g in group)
        print(f"  {label:22} n={len(group):3}  |latent|max={lat:7.3f}  |state|max={st:8.3f}")

    total = sum(f.stat().st_size for f in OUT.glob("*.raw"))
    print(f"\n[disk] {total/1e9:.2f} GB in {OUT}  -- delete after the context binary is built")


if __name__ == "__main__":
    main()
