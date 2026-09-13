#!/system/bin/sh
# Interleaved A/B of two builds of ONE MMDiT stage, with the hygiene the first attempt
# lacked. Supersedes ab_s2.sh.
#
#   sh ab_stage.sh <stage> <old_name> <new_name> [N] [ROUNDS]
#   sh ab_stage.sh 2 mmdit_s2f mmdit_s2fs 12 6
#
# Three things this guards against, all of which actually happened:
#
#  1. THE APP. `com.neodragon.demo` drives the same HTP. A run of it during an A/B
#     competes for the NPU and heats the chip; one was left running through the first
#     stage-2 A/B and every number came out 15-30% above the known baseline with a 34%
#     within-binary spread. Force-stopped here, and checked.
#
#  2. ORDER BIAS. The first script ran the old build first in every round, so the new
#     build was always measured on a chip the old one had just heated -- worth ~20%,
#     larger than the effect. Order alternates by round here.
#
#  3. THERMAL DRIFT (trap #10). The DSP throttles; a clock swing already reversed an A/B
#     verdict once in this project. A cold-start round is discarded, cooldowns are long,
#     and the skin temperature is printed each round so drift is visible rather than
#     silently folded into the result.
#
# Compare min-of-N per round, and prefer the CYCLE counts from a detailed profile over
# wall-clock ms: cycles are frequency-independent, ms are not.
set -u
STAGE=${1:?stage}; OLD=${2:?old name}; NEW=${3:?new name}
N=${4:-12}; ROUNDS=${5:-6}
cd /data/local/tmp/nd || exit 1
export LD_LIBRARY_PATH=/data/local/tmp/nd/lib
export ADSP_LIBRARY_PATH=/data/local/tmp/nd/dsp

SUF_NEW=${NEW#mmdit_}
LIST=mio_${SUF_NEW}/input_list.txt
[ -f "$LIST" ] || { echo "no $LIST"; exit 1; }
for n in $OLD $NEW; do
  [ -f "ctx/${n}_v79.bin" ] || { echo "missing ctx/${n}_v79.bin"; exit 1; }
done

# --- 1. nothing else may touch the HTP -------------------------------------
am force-stop com.neodragon.demo 2>/dev/null
sleep 2
if pidof com.neodragon.demo >/dev/null 2>&1; then
  echo "WARNING: com.neodragon.demo still alive -- results will be contaminated"
fi

skin() {
  for z in /sys/class/thermal/thermal_zone*/; do
    t=$(cat "$z/type" 2>/dev/null)
    case "$t" in *skin*|*sdm*) cat "$z/temp" 2>/dev/null; return;; esac
  done
  echo 0
}
echo "start skin temp: $(skin)"

# --- 2. accuracy, one pass each --------------------------------------------
if [ "${SKIP_ACC:-0}" != "1" ]; then
  for n in $OLD $NEW; do
    suf=${n#mmdit_}
    rm -rf dout_${suf}
    ./qnn-net-run --backend lib/libQnnHtp.so --retrieve_context ctx/${n}_v79.bin \
       --input_list $LIST --output_dir dout_${suf} --perf_profile burst >/dev/null 2>&1
    echo "accuracy $n: $(ls dout_${suf} 2>/dev/null | grep -c Result) results"
  done
fi

# --- 3. latency, alternating order, round 0 discarded as warm-up -----------
echo ""
echo "latency: $ROUNDS rounds of $N (round 0 = warm-up, discarded)"
r=0
while [ "$r" -le "$ROUNDS" ]; do
  if [ $((r % 2)) -eq 0 ]; then ORDER="$OLD $NEW"; else ORDER="$NEW $OLD"; fi
  for n in $ORDER; do
    suf=${n#mmdit_}
    rm -rf dperf_tmp
    ./qnn-net-run --backend lib/libQnnHtp.so --retrieve_context ctx/${n}_v79.bin \
       --input_list $LIST --output_dir dperf_tmp \
       --perf_profile burst --profiling_level basic \
       --num_inferences $N --keep_num_outputs 1 >/dev/null 2>&1
    if [ "$r" -gt 0 ]; then
      mv dperf_tmp/qnn-profiling-data_0.log dperf_ab_${suf}_${r}.log 2>/dev/null
    fi
    sleep 45
  done
  echo "  round $r done, skin $(skin)"
  r=$((r + 1))
done
rm -rf dperf_tmp
echo ""
ls dperf_ab_*.log 2>/dev/null | wc -l
