#!/system/bin/sh
# Min-of-N latency for the first-frame path, all in ONE thermal session.
#
# Trap #10: the DSP clock swings enough between sessions to reverse an A/B verdict, and
# this project has already seen a single build read 291.8 / 265.4 / 192.9 ms across three
# consecutive rounds purely from the clock ramping. So every module is measured here, back
# to back, and scored by min-of-N -- never averaged, never compared across sessions.
#
# Note the input-format split (trap #8): clipl is a FLOAT graph with an integer input so it
# needs --use_native_input_files; every quantised graph must omit it.
#
#   ./perf_firstframe.sh [N]
N=${1:-6}
cd /data/local/tmp/nd || exit 1
export LD_LIBRARY_PATH=/data/local/tmp/nd/lib
export ADSP_LIBRARY_PATH=/data/local/tmp/nd/dsp

run() {   # run <ctx-name> <io-dir> <list> <native>
  nm=$1; io=$2; lst=$3; nat=$4
  for r in 1 2 3; do
    rm -rf pf_out
    ./qnn-net-run --backend lib/libQnnHtp.so --retrieve_context ctx/${nm}_v79.bin \
       --input_list ${io}/${lst} --output_dir pf_out \
       --perf_profile burst --profiling_level basic \
       --num_inferences $N --keep_num_outputs 1 $nat >/dev/null 2>&1
    mv pf_out/qnn-profiling-data_0.log pf_${nm}_$r.log 2>/dev/null
  done
  echo "  $nm done"
}

echo "== first-frame path, $N inferences x 3 rounds, one session =="; date
run clipl        io_clipl        input_list.txt   --use_native_input_files
run clipg        io_clipg        input_list.txt   --use_native_input_files
run ssd1bunet    io_ssd1bunet    input_list.txt   ""
run ssd1bvaedec  io_ssd1bvaedec  input_list.txt   ""
date
ls pf_*.log
