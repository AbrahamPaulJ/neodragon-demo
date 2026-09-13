# Export one MMDiT stage's envelope graph and build its calibration + test inputs.
#
# Run through Start-Process -NoNewWindow so the job outlives whatever launched it:
# a plain background shell gets reaped, and these steps are 15-40 minutes each.
#
#   powershell -File work\export\build_stage.ps1 -Stage 1
#   powershell -File work\export\build_stage.ps1 -Stage 2 -Name mmdit_s2f
param([int]$Stage = 1, [string]$Name = "")
$ErrorActionPreference = "Continue"
$nd = (Resolve-Path "$PSScriptRoot\..\..").Path
$name = if ($Name) { $Name } else { "mmdit_s${Stage}f" }
$suf = $name -replace "^mmdit_", ""
$log = "$nd\work\export\session6_$suf.log"

function Step($title, $argv) {
    "=== $title ===" | Out-File -FilePath $log -Append -Encoding utf8
    (Get-Date -Format "HH:mm:ss") | Out-File -FilePath $log -Append -Encoding utf8
    & py -3.10 @argv 2>&1 |
        Where-Object { $_ -notmatch "^Multiple distributions|Interpolation of the Pos|^-{4,}" } |
        Out-File -FilePath $log -Append -Encoding utf8
    "exit $LASTEXITCODE" | Out-File -FilePath $log -Append -Encoding utf8
}

Remove-Item $log -ErrorAction SilentlyContinue
Step "export $name" @("$nd\work\export\export_mmdit_stage.py", "--stage", "$Stage",
                      "--envelope", "--export", "--tag", "${name}_env")
Step "calib $name" @("$nd\work\device\make_mmdit_io.py", "--stage", "$Stage",
                     "--split", "calib", "--out-name", "$name")
Step "test $name"  @("$nd\work\device\make_mmdit_io.py", "--stage", "$Stage",
                     "--split", "test", "--out-name", "mio_$suf",
                     "--videos", "50-99", "--limit", "16")
"=== done $name ===" | Out-File -FilePath $log -Append -Encoding utf8
