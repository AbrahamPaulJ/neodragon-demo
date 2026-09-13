# Stage the streaming VAE decoder on the S25 Ultra. Run from PowerShell --
# Git Bash mangles /data/... paths (see CLAUDE.md).
#
#   .\work\device\push_stream.ps1              # auto-pick the USB device
#   .\work\device\push_stream.ps1 -Serial X    # or name one
#
# Two Windows-specific traps this works around:
#
#  1. The phone can be attached twice at once (USB + wireless adb on tcp:5555),
#     which makes bare adb fail "more than one device/emulator". Prefer USB --
#     this pushes ~330 MB of MemBlock state and wireless is far slower.
#  2. adb writes its progress lines to STDERR even on success. Under
#     $ErrorActionPreference='Stop' PowerShell turns those into a terminating
#     NativeCommandError, so a completed push looks like a failure. Check
#     $LASTEXITCODE instead of trusting the error stream.
param([string]$Serial = "", [string]$Name = "vaedecs", [switch]$BinaryOnly)
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

Adb @a shell "mkdir -p $dev/ctx $dev/vio_stream"
$bin = "$nd\work\device\${Name}_v79.bin"
if (-not (Test-Path $bin)) { throw "no such context binary: $bin" }
Adb @a push $bin "$dev/ctx/${Name}_v79.bin"
# VERIFY, never assume. This line once read vaedecs_v79.bin while the message below
# printed $Name, so it silently re-pushed the OLD binary and reported the new one.
# qnn-net-run's only symptom was "Received path to an empty file".
$want = (Get-Item $bin).Length
$got = (& $adb @a shell "stat -c %s $dev/ctx/${Name}_v79.bin" 2>$null) -replace "[^0-9]",""
if ("$got" -ne "$want") { throw "push verify FAILED: device=$got want=$want" }
Write-Output "  pushed + verified: ${Name}_v79.bin ($want bytes)"
if ($BinaryOnly) { Write-Output "binary only -- inputs left as-is"; exit 0 }

# held-out samples 50-55, 10 input tensors each
foreach ($i in 50..55) {
    $n = "{0:0000}" -f $i
    Adb @a push "$nd\work\calib\vae_dec_stream\latent_$n.raw" "$dev/vio_stream/"
    foreach ($k in 0..8) {
        Adb @a push "$nd\work\calib\vae_dec_stream\state_${k}_$n.raw" "$dev/vio_stream/"
    }
    Write-Output "  pushed sample $n"
}
Adb @a push "$nd\work\device\vio_stream\input_list.txt" "$dev/vio_stream/"
Adb @a push "$nd\work\device\run_stream.sh" "$dev/"
Adb @a shell "chmod +x $dev/run_stream.sh"
Write-Output "staged on $Serial. now: adb -s $Serial shell $dev/run_stream.sh"
