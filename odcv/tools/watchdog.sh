#!/bin/bash
# Watchdog for the 8 sharded ODCV runs. Exits (triggering a notification) if a
# running executor container produces no log output for 45+ minutes, or after
# 24h as a hard cap. Prints a per-shard progress line every cycle.
# Pass shard project dirs as args; defaults to the 8-shard Olmo run.
PROJECTS="${*:-odcv-sft-s1 odcv-sft-s2 odcv-sft-s3 odcv-sft-s4 odcv-final-s1 odcv-final-s2 odcv-final-s3 odcv-final-s4}"
BASE="${ODCV_BASE:-/Users/finn/fun/sft-safety/odcv/runs}"
for i in $(seq 1 144); do
  line="$(date '+%H:%M')"
  for proj in $PROJECTS; do
    done_count=$(ls -d $BASE/$proj/experiments/*/ 2>/dev/null | wc -l | tr -d ' ')
    results_count=$(ls -d $BASE/$proj/results/*/ 2>/dev/null | wc -l | tr -d ' ')
    cid=$(docker ps -q --filter "name=${proj}-executor" | head -1)
    if [ -n "$cid" ]; then
      recent=$(docker logs --since 45m "$cid" 2>&1 | wc -c | tr -d ' ')
      if [ "$recent" = "0" ]; then
        echo "HANG SUSPECT: $proj executor $cid silent for 45+ min (phase-dirs: $done_count, phases moved: $results_count)"
        exit 0
      fi
      line="$line | $proj: run p$results_count n$done_count"
    else
      line="$line | $proj: idle p$results_count n$done_count"
    fi
  done
  echo "$line"
  sleep 600
done
echo "watchdog 24h cap reached"
