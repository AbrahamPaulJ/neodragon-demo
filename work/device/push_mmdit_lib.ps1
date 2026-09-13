# Push the aarch64-android MODEL LIBRARY (not the context binary) for one MMDiT stage,
# plus the block-name list and the run script. ~1.5 GB -- USB strongly preferred
# (measured 45 s on USB vs 100 s and two outright failures on wireless).
#
#   .\work\device\push_mmdit_lib.ps1 -Stage 0
param([int]$Stage = 0, [string]$Serial = "", [string]$Name = "")
$ErrorActionPreference = "Continue"
$adb = if ($env:ADB) { $env:ADB } else { "adb" }
$nd  = (Resolve-Path "$PSScriptRoot\..\..").Path
$dev = "/data/local/tmp/nd"
$name = if ($Name) { $Name } else { "mmdit_s$Stage" }
$suf  = $name -replace "^mmdit_", ""

function Adb { & $adb @args 2>$null; if ($LASTEXITCODE -ne 0) { throw "adb $args -> exit $LASTEXITCODE" } }

if (-not $Serial) {
    $serials = (& $adb devices 2>$null | Select-String "\tdevice$") | ForEach-Object { ($_ -split "\s+")[0] }
    if (-not $serials) { throw "no device attached" }
    $usb = $serials | Where-Object { $_ -notmatch ":" }
    $Serial = if ($usb) { @($usb)[0] } else { @($serials)[0] }
}
Write-Output "using device: $Serial"
if ($Serial -match ":") { Write-Output "  WARNING: wireless transport for a 1.5 GB push." }
$a = @("-s", $Serial)

$so = "$nd\work\device\lib$name.so"
if (-not (Test-Path $so)) { throw "no such model lib: $so  (pull it out of WSL first)" }
$want = (Get-Item $so).Length
Write-Output "  local lib: $([math]::Round($want/1MB,1)) MB"

$free = (& $adb @a shell "df /data | tail -1" 2>$null) -split "\s+"
Write-Output "  device /data free: $($free[3])"

Adb @a push $so "$dev/lib/lib$name.so"
$got = (& $adb @a shell "stat -c %s $dev/lib/lib$name.so" 2>$null) -replace "[^0-9]",""
if ("$got" -ne "$want") { throw "push verify FAILED: device=$got want=$want" }
Write-Output "  pushed + verified: lib$name.so"

if (Test-Path "$nd\work\device\blocknames_$suf.txt") {
    Adb @a push "$nd\work\device\blocknames_$suf.txt" "$dev/"
}
Adb @a push "$nd\work\device\run_mmdit_lib.sh" "$dev/"
Adb @a shell "chmod +x $dev/run_mmdit_lib.sh"
Write-Output "staged. now: adb -s $Serial shell NAME=$name $dev/run_mmdit_lib.sh $Stage acc"
