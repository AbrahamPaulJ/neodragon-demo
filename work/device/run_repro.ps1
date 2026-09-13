# Convert + push + run + score one concat reproducer variant. About two minutes each.
#
#   powershell -File work\device\run_repro.ps1 -Variant padadd
param([string]$Variant = "cat3", [string]$Serial = "")
$ErrorActionPreference = "Continue"
$adb = if ($env:ADB) { $env:ADB } else { "adb" }
$nd = (Resolve-Path "$PSScriptRoot\..\..").Path
$ndWsl = (wsl -d Ubuntu -e wslpath -a ($nd -replace '\\','/')).Trim()
$dev = "/data/local/tmp/nd"
$name = "crepro_$Variant"

if (-not $Serial) {
    $Serial = ((& $adb devices | Select-String "\tdevice$") | ForEach-Object { ($_ -split "\s+")[0] })[0]
    if (-not $Serial) { throw "no device attached" }
}
Write-Output "device: $Serial   variant: $Variant"

& py -3.10 "$nd\work\audit\concat_repro.py" --variant $Variant --export 2>&1 |
    Select-String "^\[" | ForEach-Object { Write-Output "  $_" }

wsl -d Ubuntu -e bash $ndWsl/work/qnn/convert_repro.sh $Variant 2>&1 |
    Select-String "ERROR|-> work/device" | ForEach-Object { Write-Output "  $_" }
if (-not (Test-Path "$nd\work\device\${name}_v79.bin")) { throw "conversion produced no binary" }

& $adb -s $Serial shell "mkdir -p $dev/$name" 2>&1 | Out-Null
& $adb -s $Serial push "$nd\work\device\$name" "$dev/" 2>&1 | Out-Null
& $adb -s $Serial push "$nd\work\device\${name}_v79.bin" "$dev/ctx/" 2>&1 | Out-Null
& $adb -s $Serial shell "cd $dev && export LD_LIBRARY_PATH=$dev/lib && export ADSP_LIBRARY_PATH=$dev/dsp && rm -rf cout_$Variant && ./qnn-net-run --backend lib/libQnnHtp.so --retrieve_context ctx/${name}_v79.bin --input_list $name/input_list.txt --output_dir cout_$Variant --perf_profile burst > /dev/null 2>&1; ls cout_$Variant/Result_0" 2>&1 | Out-Null

Remove-Item -Recurse -Force "$nd\work\device\cout_$Variant" -ErrorAction SilentlyContinue
& $adb -s $Serial pull "$dev/cout_$Variant" "$nd\work\device\cout_$Variant" 2>&1 | Out-Null
& py -3.10 "$nd\work\audit\concat_repro.py" --variant $Variant --check "$nd\work\device\cout_$Variant" 2>&1 |
    Select-String "device vs|rows in|VERDICT|dev " | ForEach-Object { Write-Output "  $_" }
