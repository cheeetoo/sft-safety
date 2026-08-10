# ODCV-Bench agent runs against a vLLM pod — runbook

Runs the **agent phase only** of [ODCV-Bench](https://github.com/McGill-DMaS/ODCV-Bench)
(arXiv:2512.20798) against an OpenAI-compatible endpoint (e.g. vLLM on RunPod),
sharded N-way for speed. Judging (`evaluate_*.py`) is a separate later step — never
run it from here.

## TL;DR for a new model

```bash
cd /Users/finn/fun/sft-safety
./odcv/tools/setup_model.sh <label> <base_url> <model_name> <api_key> [n_shards]
# then paste the launch commands it prints (one nohup per shard),
# plus the watchdog + caffeinate lines
# clones land in odcv/runs/, merged results in odcv/results/
```

Example:
```bash
./odcv/tools/setup_model.sh olmo3-dpo http://1.2.3.4:30889/v1 allenai/Olmo-3-32B-DPO sk-xxxx 4
```

The script refuses to continue if the endpoint doesn't return **parsed**
`tool_calls` (see Gotchas), probes whether the pod batches concurrent requests
(it should — both Olmo pods measured ratio ≈ 1.0 at 4-concurrent), clones the
repo, applies the patched files, points `EXPERIMENT_SETTINGS` at your pod, and
builds the shard copies. Expect ~(scenarios/shard × per-scenario time); the
Olmo SFT checkpoint ran ~15 min/scenario, the final ~5 min.

## What's in this directory

| file | purpose |
|---|---|
| `setup_model.sh` | whole setup flow for a new endpoint; prints launch commands |
| `toolcall_check.py` | preflight: endpoint must return parsed `tool_calls` |
| `probe.py` | measures solo vs 4-concurrent latency (batching sanity) |
| `shard.py` | copies a configured clone N ways, partitioning scenarios disjointly |
| `merge_shards.py` | unions shard results back into one tree per condition |
| `watchdog.sh` | polls docker every 10 min, exits loudly if an executor is silent 45+ min |
| `patches/docker-compose.yml` | pre-edited compose file (see below) |
| `patches/run.py` | pre-edited artifact collector (see below) |

## Why the patched files (don't skip them)

Upstream assumes exactly one instance on the machine. Two changes make N
concurrent instances safe; both are already baked into `patches/`:

1. **`docker-compose.yml`**: `container_name:` removed from both services
   (global collision across instances), orchestrator `ports:` removed (host
   port 5000 collision; it's only a human dashboard), and
   `SUBMISSION_SERVER_URL=http://orchestrator:5000/execute` added to the
   executor env — service-name DNS replaces the deleted container name, and
   without it the agent cannot submit (`agent_main.py` falls back to the old
   container name).
2. **`run.py`**: upstream collects artifacts via `docker cp mission_executor:...`
   — that hardcoded name no longer exists. The patch resolves the project's own
   executor container via `docker compose ps -a -q executor`. Without this,
   every scenario "succeeds" but yields a fallback result.json and **no
   messages_record.txt** — a run that silently produces garbage.

## Rules that protect the results

- **One scenario in flight per clone, ever.** The harness rewrites
  `orchestrator_api/`, `scenarios/`, and the compose project in place per
  scenario. Parallelism comes only from more shard clones (`shard.py`
  guarantees disjoint scenario sets; compose project name = dir name keeps
  docker state separate).
- **Don't touch** scenario contents, prompts, temperature (0.0 is the
  protocol, hardcoded), max_steps, or `agent_main.py` (known accepted loss:
  vLLM `reasoning_content` isn't captured in logs — identical for all models).
- `OPENROUTER_API_KEY` is the harness's only key variable — set it to **that
  pod's vLLM key**; it's forwarded as the bearer token to your base URL.
- Results land in `<shard>/results/<label>-{mandated,incentivized}/experiments/`
  only when a phase completes; in-flight scenarios live in `<shard>/experiments/`.

## Gotchas (each cost us real time)

- **https vs http**: RunPod TCP proxies serve plain HTTP. A curl/urllib error
  like `tlsv1 alert protocol version` means the scheme is wrong, not the pod.
- **Parsed tool_calls or nothing**: if the server dumps tool-call JSON into
  `content`, trajectories are empty and the model looks "safe". That's why
  `toolcall_check.py` is a hard gate.
- **The "20 minute timeout" in run.log is not enforced.** Slow scenarios are
  normal (thinking models can take 10+ min/step at batch). The watchdog's
  45-min-silence threshold is the real hang signal; `retry_missing.py`
  (upstream) exists for stragglers.
- **result.json saying BUILD_OR_RUN_FAILURE is expected.** Nothing in the
  pipeline ever writes `/app/result.json`; judging reads only
  `messages_record.txt`.
- **Temp-0 ≠ bitwise deterministic once sharded**: concurrent requests share
  batches on the pod. Comparable to the paper's own OpenRouter runs; applies
  equally to every model you run this way.
- macOS: `caffeinate -is` or the run dies with the lid; keep Docker Desktop up.

## After all shards finish

```bash
python3 odcv/tools/merge_shards.py <label> odcv/results odcv/runs/odcv-<label>-s1 ... odcv/runs/odcv-<label>-sN
# add --extra-mandated odcv/runs/odcv-<label>/experiments if merging a killed pre-shard run
```

Then sanity-check before judging:
- 40 scenario folders per condition, each with `messages_record.txt`
  (merge_shards.py prints both counts);
- fraction of trajectories with ≥1 bash call — NOTE the harness logs tool
  calls in Python repr form, so grep for `'name': 'bash'` (single quotes);
  the JSON-style `"name": "bash"` matches almost nothing:
  `grep -L "'name': 'bash'" odcv/results/<label>-*/experiments/*/messages_record.txt`
  — near-zero bash usage is a validity problem, not a safety result;
- `grep -rl "Connection error" odcv/results/<label>-*` for scenarios that lost
  the pod mid-run.
