# Min-of-N accelerator execute time per profiling log (trap #10: never averages).
# qnn-profile-viewer prints three blocks -- Average, Min, Max -- and the FIRST
# "Accelerator (execute) time" match belongs to Average, which is the wrong one.
#
#   .\work\device\perf_min.ps1 work\device\dperf_s0 D161 D162 D163 Q1 Q2 Q3
param([string]$Dir, [Parameter(ValueFromRemainingArguments=$true)][string[]]$Tags)
$pv = "$env:QNN_SDK_ROOT\bin\x86_64-windows-msvc\qnn-profile-viewer.exe"
"{0,-8} {1,12} {2,12}  {3}" -f "log", "min ms", "avg ms", "log created"
"-" * 62
foreach ($t in $Tags) {
    $f = Join-Path $Dir "$t.log"
    if (-not (Test-Path $f)) { "{0,-8} {1}" -f $t, "MISSING"; continue }
    $out = & $pv --input_log $f 2>$null
    $created = ($out | Select-String "^Log File Created: ") -replace "Log File Created: ", ""
    $minIdx = ($out | Select-String -Pattern "Execute Stats \(Min\)" | Select-Object -First 1).LineNumber
    $avgIdx = ($out | Select-String -Pattern "Execute Stats \(Average\)" | Select-Object -First 1).LineNumber
    function Val($from) {
        for ($i = $from; $i -lt [Math]::Min($from + 20, $out.Count); $i++) {
            if ($out[$i] -match "Backend \(Accelerator \(execute\) time\): (\d+) us") {
                return [double]$Matches[1] / 1000.0
            }
        }
        return $null
    }
    "{0,-8} {1,12:N1} {2,12:N1}  {3}" -f $t, (Val $minIdx), (Val $avgIdx), $created
}
