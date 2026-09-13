#!/system/bin/sh
# Run an MMDiT stage from the ONLINE-PREPARED model library instead of the context
# binary, which is the only way to get intermediate tensors (qnn-net-run refuses
# --debug and --set_output_tensors against --retrieve_context).
#
# Three modes, in the order they are worth running:
#
#   ./run_mmdit_lib.sh 0 acc      full test set, no dump -> compare_mmdit.py.
#                                 THE CONTROL: if this does not reproduce the context
#                                 binary's 19.48 dB, the fault is in graph PREP, not
#                                 in quantisation, and that is the whole answer.
#   ./run_mmdit_lib.sh 0 accdef   same, but WITHOUT the htp config file (default prep).
#   ./run_mmdit_lib.sh 0 blocks   one input, --set_output_tensors on the 35 block
#                                 boundaries -> block_snr.py.
#   ./run_mmdit_lib.sh 0 debug    one input, --debug: EVERY intermediate (~GBs). Slower
#                                 to write and to pull, but it answers the follow-up
#                                 question too, so it needs no second device run.
#
# Online prep of a 1.5 GB graph is slow (minutes). That is a one-off, not a per-run cost.
STAGE=${1:-0}
MODE=${2:-acc}
# target SoC of the prep config; see docs/porting-other-socs.md
HTP_ARCH=${HTP_ARCH:-v79}
SOC_MODEL=${SOC_MODEL:-69}
# NAME picks the variant; SUF is what every output directory is keyed on, so the
# session-5 baseline (NAME=mmdit_s0 -> SUF=s0) keeps exactly its old paths.
#   NAME=mmdit_s0f ./run_mmdit_lib.sh 0 acc
NAME=${NAME:-mmdit_s${STAGE}}
SUF=${NAME#mmdit_}
cd /data/local/tmp/nd || exit 1
export LD_LIBRARY_PATH=/data/local/tmp/nd/lib
export ADSP_LIBRARY_PATH=/data/local/tmp/nd/dsp

SO=lib/lib${NAME}.so
[ -f "$SO" ] || { echo "no $SO on device"; exit 1; }
LIST=mio_${SUF}/input_list.txt

# same graph-prep options the context binary was generated with, so the comparison
# isolates prep-vs-quantisation rather than confounding it with vtcm/O-level defaults.
cat > htp_cfg_${NAME}.json <<CFG
{ "graphs": [ { "vtcm_mb": 8, "graph_names": ["${NAME}"], "O": 3.0 } ],
  "devices": [ { "dsp_arch": "$HTP_ARCH", "soc_model": $SOC_MODEL,
                 "cores": [ { "perf_profile": "burst", "rpc_control_latency": 100 } ] } ] }
CFG
cat > htp_be_${NAME}.json <<BE
{ "backend_extensions": { "shared_library_path": "libQnnHtpNetRunExtensions.so",
                          "config_file_path": "/data/local/tmp/nd/htp_cfg_${NAME}.json" } }
BE

case "$MODE" in
  acc|accdef)
    OUT=lout_${SUF}_${MODE}
    rm -rf $OUT
    CFGARG="--config_file htp_be_${NAME}.json"
    [ "$MODE" = "accdef" ] && CFGARG=""
    echo "== online prep + accuracy ($MODE) =="; date
    ./qnn-net-run --backend lib/libQnnHtp.so --model $SO \
       --input_list $LIST --output_dir $OUT $CFGARG --perf_profile burst 2>&1 | tail -6
    echo "exit $?"; date
    echo "  results: $(ls $OUT 2>/dev/null | grep -c Result)"
    ;;
  debug)
    OUT=ddbg_${SUF}
    rm -rf $OUT
    head -1 $LIST > one_input.txt
    echo "== online prep + --debug (all intermediates) =="; date
    ./qnn-net-run --backend lib/libQnnHtp.so --model $SO        --input_list one_input.txt --output_dir $OUT        --config_file htp_be_${NAME}.json --perf_profile burst        --debug 2>&1 | tail -6
    echo "exit $?"; date
    echo "  dumped: $(find $OUT -name '*.raw' 2>/dev/null | wc -l) raws, $(du -sh $OUT 2>/dev/null | cut -f1)"
    ;;
  names)
    # arbitrary tensor list from a pushed file -- how block_tensors.py output is run.
    OUT=dnam_${SUF}
    F=${3:-names.txt}
    NAMES=$(cat $F)
    rm -rf $OUT
    head -1 $LIST > one_input.txt
    echo "== online prep + names from $F: $(echo $NAMES | tr ',' '
' | wc -l) tensors =="; date
    ./qnn-net-run --backend lib/libQnnHtp.so --model $SO --input_list one_input.txt --output_dir $OUT --config_file htp_be_${NAME}.json --perf_profile burst --set_output_tensors="$NAMES" 2>&1 | tail -6
    echo "exit $?"; date
    echo "  dumped: $(ls $OUT/Result_0 2>/dev/null | wc -l) raws, $(du -sh $OUT 2>/dev/null | cut -f1)"
    ;;
  blocks)
    OUT=dblk_${SUF}
    NAMES=$(cat blocknames_${SUF}.txt)
    rm -rf $OUT
    head -1 $LIST > one_input.txt
    echo "== online prep + ${MODE}: $(echo $NAMES | tr ',' '\n' | wc -l) tensors =="; date
    ./qnn-net-run --backend lib/libQnnHtp.so --model $SO \
       --input_list one_input.txt --output_dir $OUT \
       --config_file htp_be_${NAME}.json --perf_profile burst \
       --set_output_tensors="$NAMES" 2>&1 | tail -6
    echo "exit $?"; date
    echo "  dumped: $(ls $OUT/Result_0 2>/dev/null | wc -l) raws, $(du -sh $OUT 2>/dev/null | cut -f1)"
    ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac
