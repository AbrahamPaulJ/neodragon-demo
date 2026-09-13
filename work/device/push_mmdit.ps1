# Stage one MMDiT pyramid stage on the S25 Ultra. Run from PowerShell --
# Git Bash mangles /data/... paths, and so does the Bash tool for `wsl` invocations.
#
#   .\work\device\push_mmdit.ps1 -Stage 0
#   .\work\device\push_mmdit.ps1 -Stage 0 -BinaryOnly
#
# Note the scale: the context binary here is ~1.5 GB, not the 12-42 MB of the VAE
# builds. Over wireless adb that is minutes -- prefer USB for this one. Same two traps
# as push_stream.ps1 (trap #14): the phone can attach twice, and adb writes progress to
# stderr even on success, so check $LASTEXITCODE. The push is verified by byte length,
# never assumed.
param([int]$Stage = 0, [string]$Serial = "", [switch]$BinaryOnly, [string]$Name = "")
$ErrorActionPreference = "Continue"
$adb = if ($env:ADB) { $env:ADB } else { "adb" }
$nd  = (Resolve-Path "$PSScriptRoot\..\..").Path
$dev = "/data/local/tmp/nd"
$name = if ($Name) { $Name } else { "mmdit_s$Stage" }
$suf  = $name -replace "^mmdit_", ""

function Adb { & $adb @args 2>$null; if ($LASTEXITCODE -ne 0) { throw "adb $args -> exit $LASTEXITCODE" } }

if (-not $Serial) {
    $serials = (& $adb devices 2>$null | Select-String "\tdevice$") |
               ForEach-Object { ($_ -split "\s+")[0] }
    if (-not $serials) { throw "no device attached -- plug in the S25 Ultra" }
    $usb = $serials | Where-Object { $_ -notmatch ":" }
    $Serial = if ($usb) { @($usb)[0] } else { @($serials)[0] }
}
Write-Output "using device: $Serial"
if ($Serial -match ":") {
    Write-Output "  WARNING: wireless transport. This pushes ~1.5 GB; USB is much faster."
}
$a = @("-s", $Serial)

Adb @a shell "mkdir -p $dev/ctx $dev/mio_$suf"
$bin = "$nd\work\device\${name}_v79.bin"
if (-not (Test-Path $bin)) { throw "no such context binary: $bin" }

$free = (& $adb @a shell "df /data | tail -1" 2>$null) -split "\s+"
Write-Output "  device /data free: $($free[3])"

Adb @a push $bin "$dev/ctx/${name}_v79.bin"
$want = (Get-Item $bin).Length
$got = (& $adb @a shell "stat -c %s $dev/ctx/${name}_v79.bin" 2>$null) -replace "[^0-9]",""
if ("$got" -ne "$want") { throw "push verify FAILED: device=$got want=$want" }
Write-Output "  pushed + verified: ${name}_v79.bin ($([math]::Round($want/1MB,1)) MB)"
if ($BinaryOnly) { Write-Output "binary only -- inputs left as-is"; exit 0 }

$src = "$nd\work\device\mio_$suf"
if (-not (Test-Path "$src\input_list.txt")) {
    throw "no input_list.txt in $src -- run make_mmdit_io.py --stage $Stage --split test"
}
$n = 0
foreach ($f in Get-ChildItem "$src\*.raw" -Exclude "nref_*.raw") {
    Adb @a push $f.FullName "$dev/mio_$suf/"
    $n++
}
Adb @a push "$src\input_list.txt" "$dev/mio_$suf/"
Adb @a push "$nd\work\device\run_mmdit.sh" "$dev/"
Adb @a shell "chmod +x $dev/run_mmdit.sh"
Write-Output "  pushed $n input tensors"
Write-Output "staged. now: adb -s $Serial shell NAME=$name $dev/run_mmdit.sh $Stage"
