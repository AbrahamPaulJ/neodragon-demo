# Automated go/no-go for the stage-2 fused-score rebuild, then fire the 4-hour chain.
#
# Why this exists: the layout conversion, the GPU capture and the full W8A16 conversion
# add up to ~4 hours of unattended work, and the decision between them ("did the graph
# actually get smaller?") is a single number. Encoding that number as a gate means the
# night proceeds whether or not anyone is watching -- including if the driving session
# hits its usage limit mid-way.
#
# The gate: how many times does one block materialise the [24,1728,1728] score matrix?
#   shipping mmdit_s2f  -> 5.00 per block  (2 MatMul + 2 Eltwise_Binary + 1 Softmax)
#   expected mmdit_s2fs -> 3.00 per block  (1 MatMul + 1 Eltwise_Binary + 1 Softmax)
# Anything above 3.5 means the rewrite did not land the way it was designed to, and the
# 4 hours should NOT be spent. See docs/phase5-stage2-bandwidth.md and trap #40.
#
#   powershell -NoProfile -File work\qnn\gate_s2fs.ps1
#
# Safe to re-run: chain_s2fs.ps1 skips any step whose output already exists.

$ErrorActionPreference = "Continue"
$nd   = (Resolve-Path "$PSScriptRoot\..\..").Path
$log  = "$nd\work\qnn\gate_s2fs.log"
$net  = "$nd\work\device\mmdit_s2fs_net.json"
$base = "$nd\work\device\mmdit_s2f_net.json"

function Say($m) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $m
    $line | Tee-Object -FilePath $log -Append
}

Say "=== gate start; waiting for $net ==="

# The layout conversion of stage 2 is the long pole here; stage 0 took ~2 min but this
# graph is 5.6 GB. Wait up to 2 h, checking every 30 s.
$deadline = (Get-Date).AddHours(2)
while (-not (Test-Path $net)) {
    if ((Get-Date) -gt $deadline) {
        Say "TIMEOUT: no net.json after 2 h. Read work\qnn\s2fs_layout.log. NOT firing the chain."
        exit 1
    }
    Start-Sleep -Seconds 30
}
# The converter copies the file at the end of its run, but give the copy a moment to
# settle before reading it -- a partially written JSON would fail to parse and read as
# a gate failure rather than what it is.
Start-Sleep -Seconds 5
Say "net.json present ($([math]::Round((Get-Item $net).Length/1MB,1)) MB)"

$out = & py -3.10 "$nd\work\device\score_traffic.py" $base $net 2>&1
$out | Out-File "$nd\work\qnn\gate_s2fs_census.log" -Encoding utf8
$out | ForEach-Object { Say "  $_" }

# Last match is the new graph, since $net is passed second.
$m = [regex]::Matches(($out -join "`n"), 'materialisations:\s+(\d+)\s+\(([\d.]+) per block')
if ($m.Count -lt 2) {
    Say "GATE FAIL: could not parse score_traffic.py output. NOT firing the chain."
    exit 1
}
$oldPer = [double]$m[0].Groups[2].Value
$newPer = [double]$m[1].Groups[2].Value
Say "score materialisations per block: shipping $oldPer  ->  rewrite $newPer"

if ($newPer -gt 3.5) {
    Say "GATE FAIL: expected 3.00 per block, got $newPer. The rewrite did not land."
    Say "NOT firing the chain -- do not spend 4 h on this graph. Investigate first."
    exit 1
}
if ($newPer -ge $oldPer) {
    Say "GATE FAIL: no reduction ($oldPer -> $newPer). NOT firing the chain."
    exit 1
}

$saved = 1 - ($newPer / $oldPer)
Say ("GATE PASS: score traffic cut {0:P0} per block. Firing chain_s2fs.ps1." -f $saved)
& powershell -NoProfile -File "$nd\work\qnn\chain_s2fs.ps1"
Say "chain exit $LASTEXITCODE"
Say "=== gate end ==="
