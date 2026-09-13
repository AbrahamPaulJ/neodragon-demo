#!/system/bin/sh
# Interleaved A/B of two stage-2 MMDiT builds in ONE thermal session.
#
# Trap #10: the DSP throttles. Clock swung 776 -> 596 MHz between runs during Phase 4 and
# inflated cycle counts on untouched ops enough to REVERSE an A/B verdict. So the two
# binaries must alternate inside a single session and be compared min-of-N -- never one
# binary now against a number recorded on another day.
#
# Both builds read the SAME input list (mio_s2fs, absolute paths inside), so the only
# variable is the graph.
#
#   sh ab_s2.sh [N] [ROUNDS]        default N=6 inferences, 3 rounds
#
# Produces, in /data/local/tmp/nd:
#   dout_s2f/ dout_s2fs/            one pass over the 6 test samples, for SNR
#   dperf_ab_<name>_<r>.log         profiling logs -> pull, then perf_min.ps1
set -u
N=${1:-6}
ROUNDS=${2:-3}
cd /data/local/tmp/nd || exit 1
export LD_LIBRARY_PATH=/data/local/tmp/nd/lib
export ADSP_LIBRARY_PATH=/data/local/tmp/nd/dsp

LIST=mio_s2fs/input_list.txt
[ -f "$LIST" ] || { echo "no $LIST -- push the test inputs first"; exit 1; }
for n in mmdit_s2f mmdit_s2fs; do
  [ -f "ctx/${n}_v79.bin" ] || { echo "missing ctx/${n}_v79.bin"; exit 1; }
done

# ---- accuracy: one pass each over the 6 samples ----------------------------
# The graphs are QUANTISED (UFIXED_POINT_16 I/O) so --use_native_input_files must be
# OMITTED and fp32 raws fed directly (trap #8).
if [ "${SKIP_ACC:-0}" != "1" ]; then
for n in mmdit_s2f mmdit_s2fs; do
  suf=${n#mmdit_}
  echo "== accuracy: $n =="
  rm -rf dout_${suf}
  ./qnn-net-run --backend lib/libQnnHtp.so --retrieve_context ctx/${n}_v79.bin \
     --input_list $LIST --output_dir dout_${suf} \
     --perf_profile burst >/dev/null 2>&1
  echo "   results: $(ls dout_${suf} 2>/dev/null | grep -c Result)"
done
fi

# ---- latency: alternate the two builds, round by round ---------------------
echo ""
echo "== latency: $ROUNDS interleaved rounds of $N inferences =="
# ORDER IS ALTERNATED EVERY ROUND. The first version of this script always ran s2f
# first, so s2fs was always measured on a chip s2f had just heated -- a systematic bias
# worth ~20% and larger than the effect being measured. Swapping the order on odd/even
# rounds makes each build occupy the hot and cold slot equally often.
r=1
while [ "$r" -le "$ROUNDS" ]; do
  if [ $((r % 2)) -eq 1 ]; then ORDER="mmdit_s2f mmdit_s2fs"; else ORDER="mmdit_s2fs mmdit_s2f"; fi
  for n in $ORDER; do
    suf=${n#mmdit_}
    rm -rf dperf_tmp
    ./qnn-net-run --backend lib/libQnnHtp.so --retrieve_context ctx/${n}_v79.bin \
       --input_list $LIST --output_dir dperf_tmp \
       --perf_profile burst --profiling_level basic \
       --num_inferences $N --keep_num_outputs 1 >/dev/null 2>&1
    mv dperf_tmp/qnn-profiling-data_0.log dperf_ab_${suf}_${r}.log 2>/dev/null
    echo "   round $r  $n  done"
    # let the clock recover a little between builds so neither is systematically
    # measured hotter than the other
    sleep 45
  done
  r=$((r + 1))
done
rm -rf dperf_tmp
echo ""
ls -la dperf_ab_*.log
