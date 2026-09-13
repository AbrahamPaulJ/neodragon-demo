# Stage the 2-D VAE encoder on the S25 Ultra. Run from PowerShell --
# Git Bash mangles /data/... paths (see CLAUDE.md).
#
#   .\work\device\push_enc.ps1                  # vaeenc, NCHW test images
#   .\work\device\push_enc.ps1 -Name vaeencn -Nhwc
#   .\work\device\push_enc.ps1 -BinaryOnly      # swap the graph, keep the inputs
#
# Same two Windows traps as push_stream.ps1 (trap #14): the phone can attach
# twice (USB + tcp:5555), and adb writes progress to stderr even on success, so
# check $LASTEXITCODE rather than the error stream. And VERIFY the push -- a
# silently stale binary cost a wrong measurement once already.
param([string]$Serial = "", [string]$Name = "vaeenc", [switch]$Nhwc, [switch]$BinaryOnly)
$ErrorActionPreference = "Continue"
$adb = if ($env:ADB) { $env:ADB } else { "adb" }
$nd  = (Resolve-Path "$PSScriptRoot\..\..").Path
$dev = "/data/local/tmp/nd"

function Adb { & $adb @args 2>$null; if ($LASTEXITCODE -ne 0) { throw "adb $args -> exit $LASTEXITCODE" } }

if (-not $Serial) {
    $serials = (& $adb devices 2>$null | Select-String "\tdevice$") |
               ForEach-Object { ($_ -split "\s+")[0] }
    if (-not $serials) { throw "no device attached -- plug in the S25 Ultra" }
    $usb = $serials | Where-Object { $_ -notmatch ":" }   # ':' means network transport
    $Serial = if ($usb) { @($usb)[0] } else { @($serials)[0] }
}
Write-Output "using device: $Serial"
$a = @("-s", $Serial)

Adb @a shell "mkdir -p $dev/ctx $dev/eio/nhwc"
$bin = "$nd\work\device\${Name}_v79.bin"
if (-not (Test-Path $bin)) { throw "no such context binary: $bin" }
Adb @a push $bin "$dev/ctx/${Name}_v79.bin"
$want = (Get-Item $bin).Length
$got = (& $adb @a shell "stat -c %s $dev/ctx/${Name}_v79.bin" 2>$null) -replace "[^0-9]",""
if ("$got" -ne "$want") { throw "push verify FAILED: device=$got want=$want" }
Write-Output "  pushed + verified: ${Name}_v79.bin ($want bytes)"
if ($BinaryOnly) { Write-Output "binary only -- inputs left as-is"; exit 0 }

$src = if ($Nhwc) { "$nd\work\device\eio\nhwc" } else { "$nd\work\device\eio" }
$dst = if ($Nhwc) { "$dev/eio/nhwc" } else { "$dev/eio" }
if (-not (Test-Path "$src\input_list.txt")) {
    throw "no input_list.txt in $src -- run make_enc_io.py$(if ($Nhwc) {' --nhwc'})"
}
$n = 0
foreach ($f in Get-ChildItem "$src\image_*.raw") {
    Adb @a push $f.FullName "$dst/"
    $n++
}
Adb @a push "$src\input_list.txt" "$dst/"
Adb @a push "$nd\work\device\run_enc.sh" "$dev/"
Adb @a shell "chmod +x $dev/run_enc.sh"
Write-Output "  pushed $n test images to $dst"
Write-Output "staged on $Serial. now: adb -s $Serial shell $dev/run_enc.sh $Name"
