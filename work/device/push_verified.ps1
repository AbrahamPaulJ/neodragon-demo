# Push a file to the phone and VERIFY it arrived whole, retrying if not.
#
# Wireless adb silently truncates large pushes. It happened twice in one session on
# >1 GB files and both times the symptom was misleading:
#   * a 1.5 GB model .so arrived as 113 MB      -> "Failed to load ModelLib!"
#   * a 1.40 GB context binary arrived as 577 MB -> "Create From Binary failure"
# Neither adb nor qnn-net-run reported a transfer problem, and `adb push` exited 0.
#
# So: never trust a push of anything large. Compare byte length and retry.
#
#   powershell -File work\device\push_verified.ps1 -Local <file> -Remote <dir-or-path> [-Serial ...]
param(
    [Parameter(Mandatory=$true)][string]$Local,
    [Parameter(Mandatory=$true)][string]$Remote,
    [string]$Serial = "",
    [int]$Retries = 3
)
$ErrorActionPreference = "Continue"
$adb = if ($env:ADB) { $env:ADB } else { "adb" }
if (-not $Serial) {
    $devs = (& $adb devices | Select-String "\tdevice$") | ForEach-Object { ($_ -split "\s+")[0] }
    if (-not $devs) { throw "no adb device attached" }
    # prefer USB (no colon) -- wireless is where truncation happens
    $usb = $devs | Where-Object { $_ -notmatch ":" }
    $Serial = if ($usb) { @($usb)[0] } else { @($devs)[0] }
}
if (-not (Test-Path $Local)) { throw "no such file: $Local" }
$want = (Get-Item $Local).Length
$dest = if ($Remote.EndsWith("/")) { $Remote + (Split-Path $Local -Leaf) } else { $Remote }

for ($i = 1; $i -le $Retries; $i++) {
    $t = Measure-Command { & $adb -s $Serial push $Local $dest 2>&1 | Out-Null }
    $got = (& $adb -s $Serial shell "stat -c %s $dest" 2>$null) -replace "[^0-9]", ""
    if ("$got" -eq "$want") {
        "{0}  {1:N1} MB in {2:N0}s ({3:N1} MB/s) verified" -f (Split-Path $Local -Leaf),
            ($want/1MB), $t.TotalSeconds, ($want/1MB/[math]::Max($t.TotalSeconds,0.1))
        exit 0
    }
    Write-Output ("attempt {0}: TRUNCATED want={1} got={2} -- retrying" -f $i, $want, $got)
}
throw "push of $Local never verified after $Retries attempts (transport: $Serial)"
