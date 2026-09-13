#!/usr/bin/env bash
# Verify every file referenced by the converter's calibration list exists in WSL.
D=${ND:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/work/calib/vae_dec_stream
N=${1:-50}
head -"$N" "$D/calib_list_host.txt" | tr ' ' '\n' | sed 's/^.*:=//' | sort -u > /tmp/calib_files.txt
echo "distinct files referenced by first $N lines: $(wc -l < /tmp/calib_files.txt)"
missing=0
while read -r f; do
  [ -f "$f" ] || { echo "MISSING $f"; missing=$((missing+1)); }
done < /tmp/calib_files.txt
echo "missing: $missing"
