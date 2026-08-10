#!/usr/bin/env python3
"""Build N shard copies of an ODCV clone, partitioning remaining scenarios."""
import subprocess, sys
from pathlib import Path

src = Path(sys.argv[1]).resolve()   # e.g. /Users/finn/fun/sft-safety/odcv-sft
n = int(sys.argv[2])                # number of shards

# Scenarios already completed in the original clone's interrupted mandated phase
completed = set()
exp = src / "experiments"
if exp.is_dir():
    for d in sorted(exp.iterdir()):
        runlog = d / "run.log"
        if (d / "messages_record.txt").is_file() and runlog.is_file() \
           and "Submission execution completed" in runlog.read_text():
            completed.add(d.name)
print(f"{src.name}: {len(completed)} completed mandated scenarios preserved: {sorted(completed)}")

phases = {}
for phase in ("mandated_scenarios", "incentivized_scenarios"):
    names = sorted([p.name for p in (src / phase).iterdir() if p.is_dir()], key=str.lower)
    if phase == "mandated_scenarios":
        names = [x for x in names if x not in completed]
    phases[phase] = names
    print(f"  {phase}: {len(names)} to run")

for i in range(n):
    dest = src.parent / f"{src.name}-s{i+1}"
    subprocess.run(["rsync", "-a", "--delete",
                    "--exclude", ".git", "--exclude", "experiments",
                    "--exclude", "results", "--exclude", "scenarios",
                    "--exclude", "console.log",
                    f"{src}/", f"{dest}/"], check=True)
    kept = {ph: [] for ph in phases}
    for ph, names in phases.items():
        mine = set(names[i::n])
        for p in (dest / ph).iterdir():
            if p.is_dir() and p.name not in mine:
                subprocess.run(["rm", "-rf", str(p)], check=True)
        kept[ph] = sorted(mine)
    print(f"  {dest.name}: mandated={len(kept['mandated_scenarios'])} incentivized={len(kept['incentivized_scenarios'])}")

# sanity: disjoint union across shards equals expected sets
for ph, names in phases.items():
    union, total = set(), 0
    for i in range(n):
        dest = src.parent / f"{src.name}-s{i+1}"
        s = {p.name for p in (dest / ph).iterdir() if p.is_dir()}
        union |= s
        total += len(s)
    assert total == len(union) == len(names), f"shard partition broken for {ph}"
print(f"  partition verified: disjoint, union complete")
