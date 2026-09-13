#!/usr/bin/env bash
# W16 weight build of the ContextAdapter (see the note in convert_ctxadapt_w8a16.sh).
export WBITS=16
exec "$(dirname "$0")/convert_ctxadapt_w8a16.sh"
