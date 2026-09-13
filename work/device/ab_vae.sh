#!/system/bin/sh
# F = float VAE decoder, Q = W8A16. Same held-out real latents for both.
cd /data/local/tmp/nd || exit 1
export LD_LIBRARY_PATH=/data/local/tmp/nd/lib
export ADSP_LIBRARY_PATH=/data/local/tmp/nd/dsp

# The float graph declares FLOAT_32 I/O, so raw files are read natively.
# The W8A16 graph declares UFIXED_POINT_16 I/O, so --use_native_input_files
# would demand pre-quantised 16-bit files. Feeding fp32 without the flag lets
# the runtime quantise on the way in and dequantise the output.
acc() {
  tag=$1; bin=$2; nat=$3
  rm -rf vout_$tag
  ./qnn-net-run --backend lib/libQnnHtp.so --retrieve_context ctx/$bin \
     --input_list vio_real/input_list.txt --output_dir vout_$tag \
     $nat --perf_profile burst >/dev/null 2>&1
  echo "  $tag accuracy: $(ls vout_$tag 2>/dev/null | grep -c Result) results"
}

perf() {
  tag=$1; bin=$2; nat=$3
  rm -rf vperf_$tag
  ./qnn-net-run --backend lib/libQnnHtp.so --retrieve_context ctx/$bin \
     --input_list vio_real/input_list.txt --output_dir vperf_$tag \
     $nat --perf_profile burst \
     --profiling_level basic --num_inferences 12 --keep_num_outputs 1 \
     >/dev/null 2>&1
  mv vperf_$tag/qnn-profiling-data_0.log vperf_$tag.log 2>/dev/null
}

echo "== accuracy =="
acc F vaedec1_v79.bin  --use_native_input_files
acc Q vaedec1q_v79.bin ""

echo "== interleaved perf =="
for r in 1 2; do
  echo "  round $r"
  perf F$r vaedec1_v79.bin  --use_native_input_files
  sleep 15
  perf Q$r vaedec1q_v79.bin ""
  sleep 15
done
ls vperf_*.log
