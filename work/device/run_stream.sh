#!/system/bin/sh
# Streaming VAE decoder: accuracy pass + interleaved min-of-N latency.
#
# The graph is QUANTISED (UFIXED_POINT_16 I/O), so --use_native_input_files must
# be OMITTED and fp32 raws fed directly -- the runtime quantises on the way in.
# Getting this backwards is trap #8.
#
# The DSP throttles (trap #10), so latency is min-of-N inside one thermal
# session, never an average across sessions.
NAME=${1:-vaedecs}
TAG=${2:-Q}
cd /data/local/tmp/nd || exit 1
export LD_LIBRARY_PATH=/data/local/tmp/nd/lib
export ADSP_LIBRARY_PATH=/data/local/tmp/nd/dsp

echo "== accuracy =="
rm -rf sout_${TAG}
./qnn-net-run --backend lib/libQnnHtp.so --retrieve_context ctx/${NAME}_v79.bin \
   --input_list vio_stream/input_list.txt --output_dir sout_${TAG} \
   --perf_profile burst >/dev/null 2>&1
echo "  results: $(ls sout_${TAG} 2>/dev/null | grep -c Result)"

echo "== latency, 3 rounds =="
for r in 1 2 3; do
  rm -rf sperf_${TAG}$r
  ./qnn-net-run --backend lib/libQnnHtp.so --retrieve_context ctx/${NAME}_v79.bin \
     --input_list vio_stream/input_list.txt --output_dir sperf_${TAG}$r \
     --perf_profile burst --profiling_level basic \
     --num_inferences 12 --keep_num_outputs 1 >/dev/null 2>&1
  mv sperf_${TAG}$r/qnn-profiling-data_0.log sperf_${TAG}$r.log 2>/dev/null
  echo "  round $r done"
  sleep 15
done
ls sperf_${TAG}*.log
