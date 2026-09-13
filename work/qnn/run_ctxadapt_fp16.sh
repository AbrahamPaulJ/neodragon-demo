#!/usr/bin/env bash
# FP16 build of the ContextAdapter -- see the FLOAT block in convert_ctxadapt_w8a16.sh.
export FLOAT=1
exec "$(dirname "$0")/convert_ctxadapt_w8a16.sh"