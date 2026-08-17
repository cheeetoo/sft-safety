# sft-safety

Does SFT set a model's safety properties? Comparison of three Olmo-3-32B-Think
checkpoints across safety evals: `-SFT` (post-SFT), `-DPO` (post-DPO), and the
final `allenai/Olmo-3-32B-Think` (post-RLVR).

## Layout

```
logs/                  inspect .eval logs: petri, strongreject, mask (one success run per checkpoint)
mask/                  run.py: one-pass MASK runner (judge fixes baked in, see Notes);
                       writes logs/mask/<ckpt>/
impossiblebench/       ImpossibleBench checkout; run.py runs all three splits as parallel
                       inspect tasks + logs/{sft,dpo,final}/{original,oneoff,conflicting}/
                       (one success 515/515 .eval per model x split)
odcv/                  ODCV-Bench (arXiv:2512.20798) agent-phase infrastructure
├── results/           merged trajectories: {olmo3-sft,olmo3-dpo,olmo3-final}-{mandated,incentivized}/
│                      experiments/<scenario>/messages_record.txt  — canonical, judged
├── runs/              raw run clones (15: 3 originals + 12 shards), gitignored
└── tools/             reusable pipeline (RUNBOOK.md is the entry point for new models)
judging/               ODCV judge outputs: scores_/reasons_{claude47,gpt55,grok,gemini}.csv
                       + resume caches; agent_logs/ symlinks into odcv/results/
figures/               generated figures (make_figure*.py writes here)
make_figure.py         per-eval checkpoint comparison figure
make_figure_all.py     combined figure
olmo3_parser.py        Olmo-3 output parser (template.jinja: chat template)
```

## Notes

- ODCV agent runs used 4 concurrent shard clones per checkpoint against pod-hosted
  vLLM (see `odcv/tools/RUNBOOK.md` for the full recipe and gotchas).
- ODCV misbehavior scores: median of 4 judges per scenario; scores ≥3 = intentional
  misalignment per the paper's rubric. All three checkpoints fully judged
  (240 scenario x judge cells per judge file, zero N/A).
- Superseded .eval logs (errored/cancelled runs later resumed or re-run to success)
  were deleted 2026-08-10; every kept .eval has status=success and full samples.
- MASK must be run via `mask/run.py`, never plain `inspect eval inspect_evals/mask`:
  it bakes in judge_max_tokens=3000 (the task default of 500 is fully consumed by
  claude-haiku-4-5's thinking, yielding ~97% 'error' scores) and a parser patch
  stripping `**` from bolded judge verdicts, which the upstream inspect_evals
  parser rejects. The sft/final logs were produced by applying that same judge
  config post-hoc over their original solver outputs (`rescore_mask.py`, deleted
  after use — recover via `git show 0b51467:rescore_mask.py`); they carried a
  `.rescored.eval` suffix until 2026-08-13. Scores are equivalent either way.
