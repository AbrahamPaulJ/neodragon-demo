# Fire the stage-2 rebuild the moment stage 1's context binary lands, so the ~55-minute
# conversions run back to back instead of leaving the host idle in between.
#
# Stage 2's ONNX has to be re-exported: the one built earlier predates both the rank-2
# rewrite and the pad+add `_join`, and its 4 latent slots (200/160/160/640 rows) are well
# past the ~320-row concat threshold, so it would hit the interleave exactly as stage 1
# did. Its calibration and test sets are shape-identical and are reused.
$nd = (Resolve-Path "$PSScriptRoot\..\..").Path
$ndWsl = (wsl -d Ubuntu -e wslpath -a ($nd -replace '\\','/')).Trim()
$bin = "$nd\work\device\mmdit_s1f_v79.bin"
$before = (Get-Item $bin).LastWriteTime

while ($true) {
    Start-Sleep -Seconds 60
    if ((Get-Item $bin).LastWriteTime -gt $before) { break }
    if (Select-String -Path "$nd\work\qnn\s1f_padadd_full.log" -Pattern "ERROR|Traceback" -Quiet) {
        "stage 1 failed - not starting stage 2" | Out-File "$nd\work\qnn\chain_s2.log"
        exit 1
    }
}
"stage 1 binary landed $(Get-Date -Format HH:mm:ss); starting stage 2" |
    Out-File "$nd\work\qnn\chain_s2.log"

Remove-Item -Recurse -Force "$nd\work\onnx\mmdit_s2f_env" -ErrorAction SilentlyContinue
wsl -d Ubuntu -e bash -c 'rm -rf ~/neodragon-build/onnx/mmdit_s2f_env'
& py -3.10 "$nd\work\export\export_mmdit_stage.py" --stage 2 --envelope --export --tag mmdit_s2f_env *>&1 |
    Out-File "$nd\work\export\s2f_padadd_export.log"
"export exit $LASTEXITCODE" | Out-File "$nd\work\qnn\chain_s2.log" -Append

wsl -d Ubuntu -e bash $ndWsl/work/qnn/run_stage.sh full 2 mmdit_s2f *>&1 |
    Out-File "$nd\work\qnn\s2f_full.log"
"convert exit $LASTEXITCODE at $(Get-Date -Format HH:mm:ss)" | Out-File "$nd\work\qnn\chain_s2.log" -Append
