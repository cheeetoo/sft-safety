#!/bin/bash
# Set up sharded ODCV-Bench agent runs for a new OpenAI-compatible endpoint.
#
# Usage: ./setup_model.sh <label> <base_url> <model_name> <api_key> [n_shards]
#   label       lowercase [a-z0-9-], used for dir names, compose project names,
#               and the results folder prefix (e.g. "olmo3-dpo")
#   base_url    e.g. http://1.2.3.4:30889/v1  (if preflight reports a TLS
#               protocol-version alert, the pod is plain HTTP — fix the scheme)
#   model_name  the --served-model-name of the vLLM server
#   api_key     that pod's vLLM API key
#   n_shards    default 4 (both pods measured ~perfect batching at 4)
#
# Does everything except launch: preflight, batching probe, clone, patches,
# settings, sharding. Prints the launch commands at the end.
set -euo pipefail

TOOLS="$(cd "$(dirname "$0")" && pwd)"
BASE="$(dirname "$TOOLS")"
LABEL=$1; URL=$2; MODEL=$3; KEY=$4; N=${5:-4}

[[ "$LABEL" =~ ^[a-z0-9][a-z0-9-]*$ ]] || { echo "label must be lowercase [a-z0-9-] (compose project names)"; exit 1; }
mkdir -p "$BASE/runs"
CLONE="$BASE/runs/odcv-$LABEL"
[ -e "$CLONE" ] && { echo "$CLONE already exists — pick a new label or remove it"; exit 1; }

echo "== preflight: parsed tool-calls (hard requirement) =="
python3 "$TOOLS/toolcall_check.py" "$URL" "$KEY" "$MODEL"

echo "== batching probe (informational; ratio ~1 = shards will scale) =="
python3 "$TOOLS/probe.py" "$URL" "$KEY" "$MODEL" || echo "[WARN] probe failed; sharding gains unverified"

echo "== clone + apply patched docker-compose.yml and run.py =="
git clone --quiet https://github.com/McGill-DMaS/ODCV-Bench "$CLONE"
cp "$TOOLS/patches/docker-compose.yml" "$TOOLS/patches/run.py" "$CLONE/"

echo "== point EXPERIMENT_SETTINGS at the endpoint =="
python3 - "$CLONE/run_experiments.py" "$URL" "$MODEL" "$LABEL" <<'EOF'
import re, sys
p, url, model, label = sys.argv[1:5]
src = open(p).read()
new = "EXPERIMENT_SETTINGS = [\n    (%r, %r, %r),\n]" % (url, model, label)
out, n = re.compile(r"EXPERIMENT_SETTINGS = \[.*?\n\]", re.S).subn(new, src, count=1)
assert n == 1, "EXPERIMENT_SETTINGS block not found"
open(p, "w").write(out)
EOF

echo "== build $N shard copies =="
python3 "$TOOLS/shard.py" "$CLONE" "$N"

echo
echo "== launch (one terminal/tmux pane each, or as-is with nohup): =="
for i in $(seq 1 "$N"); do
  echo "cd $BASE/runs/odcv-$LABEL-s$i && export OPENROUTER_API_KEY='$KEY' && nohup python3 -u run_experiments.py > console.log 2>&1 &"
done
echo
echo "watchdog:  bash $TOOLS/watchdog.sh $(seq -s' ' -f "odcv-$LABEL-s%g" 1 "$N")"
echo "keep awake: caffeinate -is -t 86400 &"
echo "when done:  python3 $TOOLS/merge_shards.py $LABEL $BASE/results $(seq -s' ' -f "$BASE/runs/odcv-$LABEL-s%g" 1 "$N")"
