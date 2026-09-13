#!/system/bin/sh
# Interleaved A/B latency for two MMDiT context binaries, in ONE thermal session.
#
# Trap #10: the DSP clock swings enough between sessions to reverse a verdict, and the
# session-6 first attempt showed it plainly -- three consecutive rounds of the same
# build measured 291.8 / 265.4 / 192.9 ms as the clock ramped. So the two builds are
# alternated A B A B A B and each is scored by min-of-N ACROSS its rounds, never by an
# average and never across sessions.
#
# Each build reads its own input dir (mio_<suf>), because the session-6 graph takes
# temb_act where the session-5 one takes pooled_projections + timestep_ratio.
#
#   ./ab_mmdit.sh mmdit_s0 mmdit_s0f 3 6
A=${1:-mmdit_s0}
B=${2:-mmdit_s0f}
R=${3:-3}
N=${4:-6}
cd /data/local/tmp/nd || exit 1
export LD_LIBRARY_PATH=/data/local/tmp/nd/lib
export ADSP_LIBRARY_PATH=/data/local/tmp/nd/dsp

run() {                       # run <name> <tag>
  nm=$1; tag=$2
  suf=${nm#mmdit_}
  rm -rf abp_$tag
  ./qnn-net-run --backend lib/libQnnHtp.so --retrieve_context ctx/${nm}_v79.bin \
     --input_list mio_${suf}/input_list.txt --output_dir abp_$tag \
     --perf_profile burst --profiling_level basic \
     --num_inferences $N --keep_num_outputs 1 >/dev/null 2>&1
  mv abp_$tag/qnn-profiling-data_0.log ablat_$tag.log 2>/dev/null
  rm -rf abp_$tag
  echo "  $tag done ($nm)"
}

echo "== interleaved A/B: A=$A  B=$B  $R rounds of $N =="; date
r=1
while [ $r -le $R ]; do
  run "$A" "A$r"
  run "$B" "B$r"
  r=$((r+1))
done
date
ls ablat_*.log
