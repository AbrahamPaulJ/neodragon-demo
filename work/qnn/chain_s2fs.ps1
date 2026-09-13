# Unattended chain for the stage-2 fused-score rebuild (session 7, trap #40).
#
# Fire this only AFTER the layout census has confirmed the predicted byte reduction --
# it commits ~4 hours (30 min GPU capture + ~10 min I/O build + ~3 h conversion) and
# there is no point spending that on a graph that did not get smaller.
#
# Run it from PowerShell, never Git Bash: `wsl -e bash /mnt/c/...` gets its path mangled
# by Git Bash into "C:/Program Files/Git/mnt/c/..." and dies with exit 127. That already
# happened once this session.
#
#   powershell -NoProfile -File work\qnn\chain_s2fs.ps1
#
# Every step appends to work\qnn\chain_s2fs.log and is skipped if its output already
# exists, so the script is safe to re-run after an interruption.

$ErrorActionPreference = "Continue"
$nd  = (Resolve-Path "$PSScriptRoot\..\..").Path
$ndWsl = (wsl -d Ubuntu -e wslpath -a ($nd -replace '\\','/')).Trim()
$log = "$nd\work\qnn\chain_s2fs.log"
$NAME = "mmdit_s2fs"

function Say($m) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $m
    $line | Tee-Object -FilePath $log -Append
}

Say "=== chain start: $NAME ==="

# ---- 1. raw DiT call captures ------------------------------------------------
# Deleted in session 6. These are the RAW calls (latent list, context-adapter output,
# prompt mask, pooled projection, timestep), NOT the derived graph inputs, so they are
# independent of the graph rewrite and are reusable for any future requantisation.
# 50 videos x 6 stage-2 calls = 300 samples, which is what paper Table 9 uses.
$raw = "$nd\work\calib\mmdit"
$have = 0
if (Test-Path $raw) { $have = (Get-ChildItem "$raw\*.npz" -ErrorAction SilentlyContinue).Count }
if ($have -ge 900) {
    Say "step 1 SKIP: $have raw call records already present"
} else {
    Say "step 1: capture_mmdit_calib.py --num-prompts 50  (~30 min on the 3050)"
    & py -3.10 "$nd\work\pipeline\capture_mmdit_calib.py" --num-prompts 50 *>&1 |
        Out-File "$nd\work\pipeline\capture_s2fs.log" -Encoding utf8
    Say "step 1 exit $LASTEXITCODE"
    if ($LASTEXITCODE -ne 0) { Say "ABORT: capture failed"; exit 1 }
}

# ---- 2. derived per-stage calibration + test inputs ---------------------------
# ~4.0 GB for 300 stage-2 samples; attn_mask [1,1728,1728] fp32 is 11.94 MB of each.
if (Test-Path "$nd\work\calib\$NAME\calib_list_host.txt") {
    Say "step 2a SKIP: calibration list already built"
} else {
    Say "step 2a: make_mmdit_io.py --stage 2 --split calib --out-name $NAME"
    & py -3.10 "$nd\work\device\make_mmdit_io.py" --stage 2 --split calib --out-name $NAME *>&1 |
        Out-File "$nd\work\device\mio_s2fs_calib.log" -Encoding utf8
    Say "step 2a exit $LASTEXITCODE"
    if ($LASTEXITCODE -ne 0) { Say "ABORT: calib build failed"; exit 1 }
}

if (Test-Path "$nd\work\device\mio_s2fs\input_list.txt") {
    Say "step 2b SKIP: test set already built"
} else {
    Say "step 2b: make_mmdit_io.py --stage 2 --split test --out-name mio_s2fs"
    & py -3.10 "$nd\work\device\make_mmdit_io.py" --stage 2 --split test --out-name mio_s2fs *>&1 |
        Out-File "$nd\work\device\mio_s2fs_test.log" -Encoding utf8
    Say "step 2b exit $LASTEXITCODE"
}

# ---- 3. the W8A16 conversion --------------------------------------------------
# s2f's converter wall time was 2:52:07; budget ~3 h. Host RAM is the constraint, not
# time (WSL gets 10 GiB + 48 GiB swap against a 5.6 GiB fp32 ONNX).
Say "step 3: full W8A16 conversion (budget ~3 h)"
& wsl -d Ubuntu -e bash $ndWsl/work/qnn/run_stage.sh full 2 $NAME *>&1 |
    Out-File "$nd\work\qnn\s2fs_full.log" -Encoding utf8
Say "step 3 exit $LASTEXITCODE"

$bin = "$nd\work\device\${NAME}_v79.bin"
if (Test-Path $bin) {
    Say "DONE: $bin  $([math]::Round((Get-Item $bin).Length/1MB,1)) MB"
} else {
    Say "DONE but NO BINARY at $bin -- read work\qnn\s2fs_full.log"
}
Say "=== chain end ==="
