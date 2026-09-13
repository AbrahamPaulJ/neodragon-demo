#!/system/bin/sh
# Interleaved A/B: S = vaedecs (NCHW states), N = vaedecsn (NHWC states).
#
# Trap #10: the DSP throttles, so the only valid comparison is interleaved inside
# ONE thermal session, compared on min-of-N. Measuring the two builds in separate
# sessions gave -5% on min and -19% on average -- i.e. mostly thermal drift.
cd /data/local/tmp/nd || exit 1
export LD_LIBRARY_PATH=/data/local/tmp/nd/lib
export ADSP_LIBRARY_PATH=/data/local/tmp/nd/dsp

run() {
  tag=$1; bin=$2
  rm -rf ab_$tag
  ./qnn-net-run --backend lib/libQnnHtp.so --retrieve_context ctx/${bin}_v79.bin \
     --input_list vio_stream/input_list.txt --output_dir ab_$tag \
     --perf_profile burst --profiling_level basic \
     --num_inferences 12 --keep_num_outputs 1 >/dev/null 2>&1
  mv ab_$tag/qnn-profiling-data_0.log ab_$tag.log 2>/dev/null
  rm -rf ab_$tag
}

for r in 1 2 3; do
  run S$r vaedecs
  sleep 10
  run N$r vaedecsn
  sleep 10
  echo "round $r done"
done
ls ab_*.log
