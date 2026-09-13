"""Run one converted QNN context binary on the phone, from host Python.

The E2E driver keeps ALL of the reference pipeline's orchestration -- the pyramid
scheduler, the history pyramid, the per-stage sigmas -- and swaps only the heavy modules
for NPU-backed stand-ins. This is the transport for those stand-ins.

Deliberately simple and slow: one adb round trip per call (write raws -> push -> run ->
pull -> read). A 49-frame video is ~18 MMDiT calls + 4 UNet + a handful of others, so a
few minutes per video. That is fine for the first end-to-end run, whose job is to answer
"is the pipeline numerically correct", not "how fast is it". Per-module latency is already
measured properly on device with min-of-N in one thermal session (trap #10); this harness
must not be used for timing.

Two traps are handled here rather than left to the caller:

  * **trap #8** -- a QUANTISED graph (UFIXED_POINT_16 I/O) takes fp32 raws and must NOT
    get --use_native_input_files; a FLOAT graph with integer inputs MUST get it. Wrong
    either way gives identical, plausible-looking outputs for every input, so `native`
    is an explicit per-graph property and `run()` asserts the output actually changed.
  * **CRLF/BOM** -- every list file is written LF-only, ASCII, no trailing newline
    mangling. PowerShell-written lists silently broke both a tensor dump and a
    calibration list earlier in this project.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ADB = os.environ.get("ADB", "adb")  # platform-tools adb; override with $ADB
DEV = "/data/local/tmp/nd"


class Adb:
    def __init__(self, serial=None):
        self.serial = serial or self._pick()

    def _pick(self):
        out = subprocess.run([str(ADB), "devices"], capture_output=True, text=True).stdout
        devs = [ln.split()[0] for ln in out.splitlines()[1:]
                if ln.strip().endswith("device")]
        if not devs:
            raise RuntimeError("no adb device attached")
        return devs[0]

    def _run(self, args, **kw):
        return subprocess.run([str(ADB), "-s", self.serial] + args,
                              capture_output=True, text=True, **kw)

    def push(self, local, remote):
        r = self._run(["push", str(local), remote])
        if r.returncode != 0:
            raise RuntimeError("adb push failed: {}".format(r.stderr.strip()))

    def pull(self, remote, local):
        r = self._run(["pull", remote, str(local)])
        if r.returncode != 0:
            raise RuntimeError("adb pull failed: {}".format(r.stderr.strip()))

    def shell(self, cmd):
        return self._run(["shell", cmd]).stdout


class NpuGraph:
    """One context binary, invoked per call with named fp32 tensors."""

    def __init__(self, name, inputs, outputs, adb=None, native=False, tag=None):
        self.name = name
        self.inputs = list(inputs)      # ordered input names
        self.outputs = list(outputs)    # ordered output names
        self.adb = adb or Adb()
        self.native = native            # integer-input float graph -> True (trap #8)
        self.tag = tag or ("io_" + name)
        self.calls = 0
        self._last = None
        self.adb.shell("mkdir -p {}/{}".format(DEV, self.tag))

    def run(self, **tensors):
        missing = [k for k in self.inputs if k not in tensors]
        if missing:
            raise KeyError("{}: missing inputs {}".format(self.name, missing))
        tmp = Path(tempfile.mkdtemp(prefix="npu_"))
        parts = []
        for k in self.inputs:
            v = np.asarray(tensors[k])
            dt = np.int32 if (self.native and v.dtype.kind in "iu") else np.float32
            p = tmp / "{}_0000.raw".format(k)
            np.ascontiguousarray(v, dtype=dt).tofile(p)
            self.adb.push(p, "{}/{}/".format(DEV, self.tag))
            parts.append("{}:={}/{}/{}".format(k, DEV, self.tag, p.name))
        lst = tmp / "input_list.txt"
        lst.write_text(" ".join(parts) + "\n", newline="\n")
        self.adb.push(lst, "{}/{}/".format(DEV, self.tag))

        odir = "run_{}".format(self.tag)
        nat = " --use_native_input_files" if self.native else ""
        cmd = ("cd {d} && export LD_LIBRARY_PATH={d}/lib && export ADSP_LIBRARY_PATH={d}/dsp"
               " && rm -rf {o} && ./qnn-net-run --backend lib/libQnnHtp.so"
               " --retrieve_context ctx/{n}_v79.bin --input_list {t}/input_list.txt"
               " --output_dir {o} --perf_profile burst{nat} >/dev/null 2>&1;"
               " ls {o}/Result_0 | wc -l").format(
            d=DEV, o=odir, n=self.name, t=self.tag, nat=nat)
        n = self.adb.shell(cmd).strip()
        if not n or n == "0":
            raise RuntimeError("{}: qnn-net-run produced no outputs".format(self.name))

        self.adb.pull("{}/{}/Result_0".format(DEV, odir), tmp)
        got = {}
        for o in self.outputs:
            f = tmp / "Result_0" / "{}.raw".format(o)
            if not f.exists():
                raise RuntimeError("{}: no output {} (got {})".format(
                    self.name, o, [x.name for x in (tmp / "Result_0").iterdir()]))
            got[o] = np.fromfile(f, np.float32)

        # trap #8 detector: if the graph is ignoring its inputs, consecutive different
        # calls return identical bytes.
        sig = got[self.outputs[0]][:64].tobytes()
        if self.calls and sig == self._last:
            raise RuntimeError(
                "{}: identical output for a different input -- inputs are not reaching "
                "the graph (trap #8: --use_native_input_files is {})".format(
                    self.name, "set" if self.native else "unset"))
        self._last, self.calls = sig, self.calls + 1
        return got
