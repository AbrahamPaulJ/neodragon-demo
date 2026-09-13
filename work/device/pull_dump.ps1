# Pull a device tensor dump (from run_mmdit_lib.sh 'debug' or 'blocks') to the host.
# The --debug dump is ~4.3 GB / 2374 files for stage 0; USB does it in a couple of
# minutes, wireless does not (trap #14 / the 1.5 GB push failures).
#
#   .\work\device\pull_dump.ps1 -Name ddbg_s0
param([string]$Name = "ddbg_s0", [string]$Serial = "")
$adb = if ($env:ADB) { $env:ADB } else { "adb" }
$nd  = (Resolve-Path "$PSScriptRoot\..\..").Path
if (-not $Serial) {
    $serials = (& $adb devices 2>$null | Select-String "\tdevice$") | ForEach-Object { ($_ -split "\s+")[0] }
    $usb = $serials | Where-Object { $_ -notmatch ":" }
    $Serial = if ($usb) { @($usb)[0] } else { @($serials)[0] }
}
$dst = "$nd\work\device\$Name"
if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
Write-Output "pulling $Name from $Serial ..."
$t0 = Get-Date
& $adb -s $Serial pull "/data/local/tmp/nd/$Name" $dst 2>&1 | Select-Object -Last 1
$n = (Get-ChildItem -Recurse -File $dst | Measure-Object -Property Length -Sum)
Write-Output ("  {0} files, {1:N2} GB, {2:N0}s" -f $n.Count, ($n.Sum/1GB), ((Get-Date)-$t0).TotalSeconds)
