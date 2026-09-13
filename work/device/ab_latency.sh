#!/system/bin/sh
# Interleaved A/B so both builds see the same thermal state.
# A = distilt5s (residual scaling as runtime Mul)
# B = distilt5f (residual scaling folded into weights)
cd /data/local/tmp/nd || exit 1
export LD_LIBRARY_PATH=/data/local/tmp/nd/lib
export ADSP_LIBRARY_PATH=/data/local/tmp/nd/dsp

run() {
  tag=$1; bin=$2
  rm -rf ab_$tag
  ./qnn-net-run --backend lib/libQnnHtp.so \
     --retrieve_context ctx/$bin \
     --input_list io/input_list.txt --output_dir ab_$tag \
     --use_native_input_files --perf_profile burst \
     --profiling_level basic --num_inferences 100 --keep_num_outputs 1 \
     >/dev/null 2>&1
  mv ab_$tag/qnn-profiling-data_0.log ab_$tag.log 2>/dev/null
}

for round in 1 2 3; do
  echo "round $round"
  run A${round} distilt5s_v79.bin
  sleep 20
  run B${round} distilt5f_v79.bin
  sleep 20
done
ls -la ab_*.log
