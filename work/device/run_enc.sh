#!/system/bin/sh
# 2-D VAE encoder: accuracy pass + min-of-N latency in ONE thermal session.
#
# The graph is QUANTISED (UFIXED_POINT_16 I/O), so --use_native_input_files must
# be OMITTED and fp32 raws fed directly -- the runtime quantises on the way in.
# Getting this backwards is trap #8, and its symptom is identical plausible
# outputs from every input, which compare_enc.py explicitly checks for.
#
# The DSP throttles (trap #10): latency is min-of-N inside one session, never an
# average across sessions.
NAME=${1:-vaeenc}
TAG=${2:-Q}
LIST=${3:-eio/input_list.txt}
cd /data/local/tmp/nd || exit 1
export LD_LIBRARY_PATH=/data/local/tmp/nd/lib
export ADSP_LIBRARY_PATH=/data/local/tmp/nd/dsp

echo "== accuracy: $NAME ($LIST) =="
rm -rf eout_${TAG}
./qnn-net-run --backend lib/libQnnHtp.so --retrieve_context ctx/${NAME}_v79.bin \
   --input_list ${LIST} --output_dir eout_${TAG} \
   --perf_profile burst >/dev/null 2>&1
echo "  results: $(ls eout_${TAG} 2>/dev/null | grep -c Result)"

echo "== latency, 3 rounds =="
for r in 1 2 3; do
  rm -rf eperf_${TAG}$r
  ./qnn-net-run --backend lib/libQnnHtp.so --retrieve_context ctx/${NAME}_v79.bin \
     --input_list ${LIST} --output_dir eperf_${TAG}$r \
     --perf_profile burst --profiling_level basic \
     --num_inferences 8 --keep_num_outputs 1 >/dev/null 2>&1
  mv eperf_${TAG}$r/qnn-profiling-data_0.log eperf_${TAG}$r.log 2>/dev/null
  echo "  round $r done"
  sleep 15
done
ls eperf_${TAG}*.log
