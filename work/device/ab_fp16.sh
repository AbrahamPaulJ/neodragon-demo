#!/system/bin/sh
# Interleaved: F = folded, fp32-declared graph   H = fp16-declared graph
cd /data/local/tmp/nd || exit 1
export LD_LIBRARY_PATH=/data/local/tmp/nd/lib
export ADSP_LIBRARY_PATH=/data/local/tmp/nd/dsp

run() {
  tag=$1; bin=$2; extra=$3
  rm -rf ab_$tag
  ./qnn-net-run --backend lib/libQnnHtp.so --retrieve_context ctx/$bin \
     --input_list io/input_list.txt --output_dir ab_$tag \
     --use_native_input_files --perf_profile burst \
     --profiling_level basic --num_inferences 100 $extra >/dev/null 2>&1
  mv ab_$tag/qnn-profiling-data_0.log ab_$tag.log 2>/dev/null
}

# accuracy pass for H (keep all 3 outputs)
rm -rf outh
./qnn-net-run --backend lib/libQnnHtp.so --retrieve_context ctx/distilt5h_v79.bin \
   --input_list io/input_list.txt --output_dir outh \
   --use_native_input_files --perf_profile burst >/dev/null 2>&1
echo "accuracy outputs:"; ls outh

for round in 1 2 3; do
  echo "round $round"
  run F$round distilt5f_v79.bin "--keep_num_outputs 1"
  sleep 20
  run H$round distilt5h_v79.bin "--keep_num_outputs 1"
  sleep 20
done
ls ab_F*.log ab_H*.log
