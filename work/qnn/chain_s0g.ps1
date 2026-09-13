# Rebuild stage 0 on the unified code path, once stage 2's conversion is done.
#
# Why this is not optional: export_mmdit_stage.py now produces the rank-2 graph with
# _join(), so the shipping 27.89 dB stage-0 binary (mmdit_s0f) can no longer be
# reproduced from the source that is in the tree. It is also expected to help --
# rank 2 cut stage 1's graph 2246 -> 1715 nodes and its transpose ratio 0.18 -> 0.14.
#
# Built as mmdit_s0g so mmdit_s0f survives for an A/B. The graph's inputs and output are
# unchanged by the rewrite, so the calibration and test sets are copied, not rebuilt.
$nd = (Resolve-Path "$PSScriptRoot\..\..").Path
$ndWsl = (wsl -d Ubuntu -e wslpath -a ($nd -replace '\\','/')).Trim()
$log = "$nd\work\qnn\chain_s0g.log"
"waiting for stage 2 ($(Get-Date -Format HH:mm:ss))" | Out-File $log

while ($true) {
    Start-Sleep -Seconds 120
    $t = Get-Content "$nd\work\qnn\s2f_full.log" -Raw -ErrorAction SilentlyContinue
    if ($t -match "mmdit_s2f_v79.bin") { break }
    if ($t -match "ERROR|Traceback") { "stage 2 failed - stopping" | Out-File $log -Append; exit 1 }
}
"stage 2 done $(Get-Date -Format HH:mm:ss); rebuilding stage 0 as mmdit_s0g" | Out-File $log -Append

# reuse stage 0's calibration and test sets -- shapes are identical
if (-not (Test-Path "$nd\work\calib\mmdit_s0g\dims.txt")) {
    Copy-Item -Recurse -Force "$nd\work\calib\mmdit_s0f" "$nd\work\calib\mmdit_s0g"
    (Get-Content "$nd\work\calib\mmdit_s0g\calib_list_host.txt") -replace "mmdit_s0f", "mmdit_s0g" |
        Set-Content "$nd\work\calib\mmdit_s0g\calib_list_host.txt" -Encoding ascii
}
if (-not (Test-Path "$nd\work\device\mio_s0g\input_list.txt")) {
    Copy-Item -Recurse -Force "$nd\work\device\mio_s0f" "$nd\work\device\mio_s0g"
    (Get-Content "$nd\work\device\mio_s0g\input_list.txt") -replace "mio_s0f", "mio_s0g" |
        Set-Content "$nd\work\device\mio_s0g\input_list.txt" -Encoding ascii
}

& py -3.10 "$nd\work\export\export_mmdit_stage.py" --stage 0 --envelope --export --tag mmdit_s0g_env *>&1 |
    Out-File "$nd\work\export\s0g_export.log"
"export exit $LASTEXITCODE" | Out-File $log -Append

wsl -d Ubuntu -e bash $ndWsl/work/qnn/run_stage.sh full 0 mmdit_s0g *>&1 |
    Out-File "$nd\work\qnn\s0g_full.log"
"convert exit $LASTEXITCODE at $(Get-Date -Format HH:mm:ss)" | Out-File $log -Append
