# Fire the CLIP G W8A16 build once the SSD1B VAE decoder conversion finishes.
$nd = (Resolve-Path "$PSScriptRoot\..\..").Path
$ndWsl = (wsl -d Ubuntu -e wslpath -a ($nd -replace '\\','/')).Trim()
while ($true) {
    Start-Sleep -Seconds 60
    $t = Get-Content "$nd\work\qnn\ssd1bvaedec_full.log" -Raw -ErrorAction SilentlyContinue
    if ($t -match "ssd1bvaedec_v79.bin") { break }
    if ($t -match "Traceback|RuntimeError") { "vaedec failed" | Out-File "$nd\work\qnn\chain_clipgq.log"; exit 1 }
}
"vaedec done $(Get-Date -Format HH:mm:ss); starting clipg W8A16" | Out-File "$nd\work\qnn\chain_clipgq.log"
$env:QUANT = "1"
wsl -d Ubuntu -e bash -c "QUANT=1 bash $ndWsl/work/qnn/convert_clip_fp16.sh clipg" *>&1 |
    Out-File "$nd\work\qnn\clipgq_full.log"
"clipgq exit $LASTEXITCODE at $(Get-Date -Format HH:mm:ss)" | Out-File "$nd\work\qnn\chain_clipgq.log" -Append
