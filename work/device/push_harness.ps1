# Re-stage the desktop QNN harness at /data/local/tmp/nd on the S25 Ultra.
#
# Session 6 deleted this directory to reclaim 19 GB, on the correct reasoning that the
# APK never touches it -- the app dlopen's the backend in-process (trap #33). But every
# `push_*.ps1` / `run_*.sh` measurement script drives `qnn-net-run` from there, so any
# device A/B needs it back first. Nothing in the repo rebuilt it until this script.
#
#   .\work\device\push_harness.ps1
#   .\work\device\push_harness.ps1 -Serial <DEVICE_SERIAL>
#
# What goes where, and why:
#   nd/qnn-net-run          the runner itself (aarch64-android)
#   nd/lib/                 CPU-side backend: libQnnHtp.so + the V79 stub it dlopen's,
#                           plus Prepare/NetRunExtensions/System. LD_LIBRARY_PATH.
#   nd/dsp/                 libQnnHtpV79Skel.so -- the HEXAGON-side skel, which comes
#                           from lib/hexagon-v79/unsigned/, NOT from lib/aarch64-android.
#                           ADSP_LIBRARY_PATH.
#   nd/*.sh                 the measurement scripts
#
# Traps handled (#14): the phone can attach twice (USB + wireless) so a bare adb fails
# "more than one device"; and adb writes progress to stderr even on success, so
# $LASTEXITCODE is checked rather than trusting a non-empty stderr.

param([string]$Serial = "", [string]$HtpArch = "v79")   # -HtpArch v75 for an 8 Gen 3, etc.
$ErrorActionPreference = "Continue"

$adb   = if ($env:ADB) { $env:ADB } else { "adb" }
$nd    = (Resolve-Path "$PSScriptRoot\..\..").Path
$qairt = $env:QNN_SDK_ROOT
if (-not $qairt) { throw "set QNN_SDK_ROOT to your QAIRT SDK root" }
$dev   = "/data/local/tmp/nd"

if (-not $Serial) {
    $serials = (& $adb devices 2>$null | Select-String "\tdevice$") |
               ForEach-Object { ($_ -split "\s+")[0] }
    if (-not $serials) { throw "no device attached" }
    $usb = $serials | Where-Object { $_ -notmatch ":" }      # prefer USB over wireless
    $Serial = if ($usb) { @($usb)[0] } else { @($serials)[0] }
}
Write-Output "device: $Serial"
$a = @("-s", $Serial)

function Adb { & $adb @args; if ($LASTEXITCODE -ne 0) { throw "adb failed: $args" } }

Adb @a shell "mkdir -p $dev/lib $dev/dsp $dev/ctx"

# --- the runner -------------------------------------------------------------
$runner = "$qairt\bin\aarch64-android\qnn-net-run"
Adb @a push $runner "$dev/qnn-net-run"
Adb @a shell "chmod 755 $dev/qnn-net-run"

# --- CPU-side libs ----------------------------------------------------------
$A = $HtpArch.ToUpper()
$libs = @("libQnnHtp.so", "libQnnHtp${A}Stub.so", "libQnnHtpPrepare.so",
          "libQnnHtpNetRunExtensions.so", "libQnnSystem.so")
foreach ($l in $libs) {
    $p = "$qairt\lib\aarch64-android\$l"
    if (-not (Test-Path $p)) { Write-Output "  MISSING $l"; continue }
    Adb @a push $p "$dev/lib/"
}

# --- Hexagon skel -----------------------------------------------------------
# NOTE the different source directory. Pushing the aarch64 libQnnHtpV79.so here instead
# is the classic mistake and shows up as "Failed to load skel, error: 4000".
Adb @a push "$qairt\lib\hexagon-$HtpArch\unsigned\libQnnHtp${A}Skel.so" "$dev/dsp/"

# --- measurement scripts ----------------------------------------------------
foreach ($s in (Get-ChildItem "$nd\work\device\*.sh")) {
    Adb @a push $s.FullName "$dev/"
}

# --- verify -----------------------------------------------------------------
Write-Output ""
Write-Output "--- staged ---"
& $adb @a shell "ls -la $dev; echo '--- lib ---'; ls $dev/lib; echo '--- dsp ---'; ls $dev/dsp"
Write-Output ""
$probe = & $adb @a shell "cd $dev && LD_LIBRARY_PATH=$dev/lib ADSP_LIBRARY_PATH=$dev/dsp ./qnn-net-run --version 2>&1 | head -3"
Write-Output "qnn-net-run --version -> $probe"
