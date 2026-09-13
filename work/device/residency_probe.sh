#!/system/bin/sh
# Can the AR video loop's graphs be CO-RESIDENT on the phone, or must they be swapped?
#
# This is the question that decides the architecture of the E2E driver. The AR loop
# visits all three MMDiT stages every unit, six units per 49-frame video. If the three
# context binaries can be held at once, the driver loads once and runs; if not, it has to
# load/unload ~1.5 GB per stage per unit -- 18 loads per video -- and that cost would
# dominate everything measured so far.
#
# Method: hold each graph open in its own qnn-net-run process (a long --num_inferences
# run so it stays resident), starting them one at a time, and sample memory after each.
# One process per graph is not how the real driver would work -- it would create several
# graphs in one process -- but it is an UPPER bound on memory and a faithful test of
# whether the device can hold the weights at all.
#
# Reports MemAvailable and each process's RSS after every addition, so the point where it
# stops fitting is visible rather than inferred.
#
#   ./residency_probe.sh
cd /data/local/tmp/nd || exit 1
export LD_LIBRARY_PATH=/data/local/tmp/nd/lib
export ADSP_LIBRARY_PATH=/data/local/tmp/nd/dsp

GRAPHS="mmdit_s0g:mio_s0g mmdit_s1f:mio_s1f mmdit_s2f:mio_s2f"
PIDS=""

mem() {
  awk '/MemTotal|MemAvailable|SwapFree/{printf "%s %d MB  ", $1, $2/1024}' /proc/meminfo
  echo
}

rss() {
  for p in $PIDS; do
    [ -d /proc/$p ] || { echo "    pid $p DIED"; continue; }
    r=$(awk '/VmRSS/{print $2}' /proc/$p/status 2>/dev/null)
    c=$(cat /proc/$p/cmdline 2>/dev/null | tr '\0' ' ' | grep -o 'ctx/[a-z0-9_]*' | head -1)
    echo "    pid $p  $c  RSS $((r/1024)) MB"
  done
}

echo "=== residency probe: can the AR loop's graphs be co-resident? ==="
date
echo "baseline:"; mem

for g in $GRAPHS; do
  name=${g%%:*}
  io=${g##*:}
  echo ""
  echo "--- adding $name ---"
  ./qnn-net-run --backend lib/libQnnHtp.so --retrieve_context ctx/${name}_v79.bin \
     --input_list ${io}/input_list.txt --output_dir rp_out_${name} \
     --perf_profile burst --num_inferences 100000 --keep_num_outputs 1 \
     > rp_${name}.log 2>&1 &
  PIDS="$PIDS $!"
  # wait for the context to actually load (RSS stops climbing) rather than a fixed sleep
  prev=0; same=0
  while [ $same -lt 3 ]; do
    sleep 5
    last=$(echo $PIDS | tr ' ' '\n' | tail -1)
    cur=$(awk '/VmRSS/{print $2}' /proc/$last/status 2>/dev/null || echo 0)
    [ "$cur" = "0" ] && { echo "    process gone -- OUT OF MEMORY at $name"; break; }
    if [ "$cur" = "$prev" ]; then same=$((same+1)); else same=0; fi
    prev=$cur
  done
  mem
  rss
done

echo ""
echo "=== all graphs resident ==="
mem
rss
echo ""
for p in $PIDS; do kill $p 2>/dev/null; done
sleep 3
echo "cleaned up"; mem
