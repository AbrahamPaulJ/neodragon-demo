#!/usr/bin/env bash
# Session 6 driver for the stage-0 AdaLN fix build (NAME=mmdit_s0f).
#
# Exists because `wsl -d Ubuntu -e bash -lc "NAME=... ./convert..."` does not survive
# the quoting (see CLAUDE.md: run WSL as `wsl -d Ubuntu -e bash <script>`), so the
# variant name is baked into a script instead of passed on a command line.
#
#   wsl -d Ubuntu -e bash work/qnn/run_s0f.sh layout
#   wsl -d Ubuntu -e bash work/qnn/run_s0f.sh full
#   wsl -d Ubuntu -e bash work/qnn/run_s0f.sh androidlib
set -uo pipefail
cd "$(dirname "$0")" || exit 1
# NOTE: assigned unconditionally, NOT with ${NAME:-...}. Launching this through
# `wsl -d Ubuntu -e bash ...` from PowerShell inherits the Windows environment, which
# already defines NAME (the machine/user name), so a default-if-unset would silently
# convert a graph called "<HOSTNAME>".
STEP=${1:-full}
export NAME=${2:-mmdit_s0f}

case "$STEP" in
  layout)     exec ./convert_mmdit_w8a16.sh 0 --layout ;;
  probe)      exec ./convert_mmdit_w8a16.sh 0 --probe ;;
  full)       exec ./convert_mmdit_w8a16.sh 0 ;;
  androidlib) exec ./build_android_lib.sh 0 ;;
  *) echo "unknown step $STEP"; exit 1 ;;
esac
