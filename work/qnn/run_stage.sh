#!/usr/bin/env bash
# WSL entry point for one MMDiT stage variant. Supersedes run_s0f.sh (kept as the
# stage-0 shorthand).
#
# Exists because `wsl -d Ubuntu -e bash -lc "NAME=... ./convert..."` does not survive
# the quoting (CLAUDE.md: run WSL as `wsl -d Ubuntu -e bash <script>`), and because the
# Windows environment already defines NAME -- so the variant is taken as a POSITIONAL
# argument here, never defaulted from the environment.
#
#   wsl -d Ubuntu -e bash work/qnn/run_stage.sh layout     0 mmdit_s0f
#   wsl -d Ubuntu -e bash work/qnn/run_stage.sh full       1 mmdit_s1f
#   wsl -d Ubuntu -e bash work/qnn/run_stage.sh androidlib 0 mmdit_s0f
set -uo pipefail
cd "$(dirname "$0")" || exit 1
STEP=${1:-full}
STAGE=${2:-0}
export NAME=${3:-mmdit_s${STAGE}f}

echo "step=$STEP stage=$STAGE NAME=$NAME"
case "$STEP" in
  layout)     exec ./convert_mmdit_w8a16.sh "$STAGE" --layout ;;
  probe)      exec ./convert_mmdit_w8a16.sh "$STAGE" --probe ;;
  full)       exec ./convert_mmdit_w8a16.sh "$STAGE" ;;
  androidlib) exec ./build_android_lib.sh "$STAGE" ;;
  *) echo "unknown step $STEP"; exit 1 ;;
esac
