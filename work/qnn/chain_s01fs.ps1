# Stages 0 and 1 with the session-7 fused-score fix (trap #40), run back to back.
#
# Stage 2 is already built (mmdit_s2fs, -28.3% output bytes). These two are the rest of
# the win: 169.2 -> ~146 ms and 375.8 -> ~307 ms, about 0.55 s more off the video. They
# are much cheaper than stage 2 was -- s0/s1 converter wall times were 38:32 and 48:43,
# against stage 2's 2:28:21.
#
# Their EXPORTS are already proven: `--pad-check` built StageMMDiT for all three stages
# with fuse_score=True and every unit came back at the fp32 floor (121.6-127.1 dB). So
# there is no equivalence risk left here, only conversion.
#
# DISK IS THE CONSTRAINT, not time. C: hit 100% during the stage-2 run. Each ONNX is
# ~5.7 GB regardless of stage (it is the same 1.5B model), and the converter stages a
# second copy onto ext4. So this does ONE STAGE AT A TIME and deletes that stage's ONNX
# and calibration set before starting the next.
#
#   powershell -NoProfile -File work\qnn\chain_s01fs.ps1
#
# Every step skips itself if its output already exists; safe to re-run after a stop.

$ErrorActionPreference = "Continue"
$nd  = (Resolve-Path "$PSScriptRoot\..\..").Path
$ndWsl = (wsl -d Ubuntu -e wslpath -a ($nd -replace '\\','/')).Trim()
$log = "$nd\work\qnn\chain_s01fs.log"

function Say($m) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $m
    $line | Tee-Object -FilePath $log -Append
}

function FreeGB { [math]::Round((Get-PSDrive C).Free / 1GB, 1) }

Say "=== chain start: stages 0 and 1, fused score.  C: $(FreeGB) GB free ==="

foreach ($stage in 0, 1) {
    $NAME = "mmdit_s${stage}fs"
    $bin  = "$nd\work\device\${NAME}_v79.bin"
    if (Test-Path $bin) { Say "$NAME SKIP: binary already present"; continue }

    # --- guard: never start a conversion without room for it ------------------
    if ((FreeGB) -lt 14) {
        Say "ABORT before ${NAME}: only $(FreeGB) GB free, need ~14. Reclaim first."
        exit 1
    }

    # --- calibration, rebuilt from the kept raw captures (~30 s) --------------
    if (Test-Path "$nd\work\calib\$NAME\calib_list_host.txt") {
        Say "$NAME step 1 SKIP: calibration already built"
    } else {
        Say "$NAME step 1: make_mmdit_io --stage $stage --split calib"
        & py -3.10 "$nd\work\device\make_mmdit_io.py" --stage $stage --split calib --out-name $NAME *>&1 |
            Out-File "$nd\work\device\${NAME}_calib.log" -Encoding utf8
        Say "$NAME step 1 exit $LASTEXITCODE"
        if ($LASTEXITCODE -ne 0) { Say "ABORT: calib build failed"; exit 1 }
    }

    # --- export ---------------------------------------------------------------
    $raw = "$nd\work\onnx\${NAME}_env\${NAME}_env_raw.onnx"
    if (Test-Path $raw) {
        Say "$NAME step 2 SKIP: ONNX already present"
    } else {
        Say "$NAME step 2: export ONNX (~15 min, ~5.7 GB)"
        & py -3.10 "$nd\work\export\export_mmdit_stage.py" --stage $stage --envelope --export --tag "${NAME}_env" *>&1 |
            Out-File "$nd\work\export\${NAME}_export.log" -Encoding utf8
        Say "$NAME step 2 exit $LASTEXITCODE"
        if (-not (Test-Path $raw)) { Say "ABORT: no ONNX at $raw"; exit 1 }
    }

    # --- convert --------------------------------------------------------------
    Say "$NAME step 3: W8A16 conversion.  C: $(FreeGB) GB free"
    & wsl -d Ubuntu -e bash $ndWsl/work/qnn/run_stage.sh full $stage $NAME *>&1 |
        Out-File "$nd\work\qnn\${NAME}_full.log" -Encoding utf8
    Say "$NAME step 3 exit $LASTEXITCODE"

    if (Test-Path $bin) {
        Say "$NAME DONE: $([math]::Round((Get-Item $bin).Length/1MB,1)) MB"
    } else {
        Say "$NAME FAILED: no binary. Read work\qnn\${NAME}_full.log"
    }

    # --- reclaim before the next stage ---------------------------------------
    # Both are regenerable: the ONNX by re-export (~15 min), the calibration set from
    # work/calib/mmdit (the raw captures, deliberately kept) in ~30 s.
    Remove-Item -Recurse -Force "$nd\work\onnx\${NAME}_env" -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force "$nd\work\calib\$NAME" -ErrorAction SilentlyContinue
    & wsl -d Ubuntu -e bash -c "rm -rf ~/neodragon-build/onnx/${NAME}_env"
    Say "$NAME reclaimed ONNX + calib.  C: $(FreeGB) GB free"
}

Say "=== chain end.  C: $(FreeGB) GB free ==="
