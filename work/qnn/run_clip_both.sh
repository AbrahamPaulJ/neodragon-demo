#!/usr/bin/env bash
# Run CLIP L then CLIP G back to back so the FP16 text encoders land in one go.
set -uo pipefail
cd "$(dirname "$0")"
for m in clipl clipg; do
  echo "===== $m ====="
  ./convert_clip_fp16.sh "$m" || echo "FAILED: $m"
done