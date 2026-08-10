#!/bin/bash

uv run inspect eval src/impossiblebench/livecodebench_tasks.py@impossible_livecodebench \
  --task-config task_args.yaml \
  -T split=conflicting \
  --model vllm/$MODEL \
  --epochs 5 \
  --max-tokens 16384 \
  --max-connections 16 --max-sandboxes 16 \
  --log-dir logs/impossiblebench/$CKPT/$split
