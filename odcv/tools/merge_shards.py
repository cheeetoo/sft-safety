#!/usr/bin/env python3
"""Merge sharded ODCV results back into one results tree per condition.

Usage:
  merge_shards.py <label> <out_dir> <shard_clone>... [--extra-mandated DIR]

For each phase (mandated, incentivized), unions
<shard>/results/<label>-<phase>/experiments/<scenario>/ from every shard into
<out_dir>/<label>-<phase>/experiments/. Duplicate scenario names abort the
merge. --extra-mandated adds completed scenario dirs from an interrupted
pre-shard run's experiments/ folder (only those with a messages_record.txt and
a finished run.log are taken).
"""
import shutil, sys
from pathlib import Path

args = sys.argv[1:]
extra_mandated = None
if "--extra-mandated" in args:
    i = args.index("--extra-mandated")
    extra_mandated = Path(args[i + 1])
    del args[i:i + 2]

label, out_dir, shards = args[0], Path(args[1]), [Path(p) for p in args[2:]]
assert shards, "need at least one shard clone"

for phase in ("mandated", "incentivized"):
    dest = out_dir / f"{label}-{phase}" / "experiments"
    dest.mkdir(parents=True, exist_ok=True)
    sources = []
    for shard in shards:
        src = shard / "results" / f"{label}-{phase}" / "experiments"
        if not src.is_dir():
            print(f"[WARN] missing {src} — shard incomplete?")
            continue
        sources += [p for p in src.iterdir() if p.is_dir()]
    if phase == "mandated" and extra_mandated and extra_mandated.is_dir():
        for p in extra_mandated.iterdir():
            runlog = p / "run.log"
            if (p / "messages_record.txt").is_file() and runlog.is_file() \
               and "Submission execution completed" in runlog.read_text():
                sources.append(p)
    seen = {}
    for p in sources:
        has_msgs = (p / "messages_record.txt").is_file()
        if p.name in seen:
            # a scenario can appear both as a failed shard entry (no
            # messages_record.txt) and as a successful retry — keep the copy
            # with a trajectory; two trajectory-bearing copies are ambiguous
            prev_has = (seen[p.name] / "messages_record.txt").is_file()
            if has_msgs and prev_has:
                sys.exit(f"ABORT: two trajectories for {p.name}:\n  {seen[p.name]}\n  {p}")
            if not has_msgs:
                continue
        seen[p.name] = p
        target = dest / p.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(p, target)
    n_msgs = sum(1 for d in dest.iterdir() if (d / "messages_record.txt").is_file())
    print(f"{label}-{phase}: {len(seen)} scenarios merged, {n_msgs} with messages_record.txt")
