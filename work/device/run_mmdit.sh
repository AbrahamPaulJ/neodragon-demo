#!/system/bin/sh
# One MMDiT pyramid stage: accuracy pass + min-of-N latency in ONE thermal session.
#
# The graph is QUANTISED (UFIXED_POINT_16 I/O), so --use_native_input_files must be
# OMITTED and fp32 raws fed directly (trap #8). compare_mmdit.py asserts two different
# inputs give two different outputs, which is the only reliable symptom when this is wrong.
#
# The DSP throttles (trap #10): min-of-N inside one session, never averages across sessions.
# Fewer inferences per round than the VAE scripts because a stage-2 call is ~1 s.
STAGE=${1:-0}
NAME=${NAME:-mmdit_s${STAGE}}
SUF=${NAME#mmdit_}
TAG=${2:-Q}
N=${3:-6}
cd /data/local/tmp/nd || exit 1
export LD_LIBRARY_PATH=/data/local/tmp/nd/lib
export ADSP_LIBRARY_PATH=/data/local/tmp/nd/dsp

LIST=mio_${SUF}/input_list.txt
[ -f "$LIST" ] || { echo "no $LIST on device -- push_mmdit.ps1 first"; exit 1; }

echo "== accuracy: $NAME =="
rm -rf dout_${SUF}
./qnn-net-run --backend lib/libQnnHtp.so --retrieve_context ctx/${NAME}_v79.bin \
   --input_list $LIST --output_dir dout_${SUF} \
   --perf_profile burst >/dev/null 2>&1
echo "  results: $(ls dout_${SUF} 2>/dev/null | grep -c Result)"

echo "== latency, 3 rounds of $N =="
for r in 1 2 3; do
  rm -rf dperf_${TAG}$r
  ./qnn-net-run --backend lib/libQnnHtp.so --retrieve_context ctx/${NAME}_v79.bin \
     --input_list $LIST --output_dir dperf_${TAG}$r \
     --perf_profile burst --profiling_level basic \
     --num_inferences $N --keep_num_outputs 1 >/dev/null 2>&1
  mv dperf_${TAG}$r/qnn-profiling-data_0.log dperf_${SUF}_${TAG}$r.log 2>/dev/null
  echo "  round $r done"
  sleep 20
done
ls dperf_${SUF}_*.log
