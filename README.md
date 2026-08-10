# sft-safety

Does SFT set a model's safety properties? Comparison of `allenai/Olmo-3-32B-Think-SFT`
(post-SFT checkpoint) vs `allenai/Olmo-3-32B-Think` (final, post-RL) across safety evals.

## Layout

```
logs/                  inspect .eval logs: petri, strongreject, mask (one success run per checkpoint)
impossiblebench/       ImpossibleBench checkout + logs/{sft,final}/{original,oneoff,conflicting}/
                       (one success 515/515 .eval per model x split)
odcv/                  ODCV-Bench (arXiv:2512.20798) agent-phase infrastructure
├── results/           merged trajectories: {olmo3-sft,olmo3-final}-{mandated,incentivized}/
│                      experiments/<scenario>/messages_record.txt  — canonical, judged
├── runs/              raw run clones (10: 2 originals + 8 shards), gitignored
└── tools/             reusable pipeline (RUNBOOK.md is the entry point for new models)
judging/               ODCV judge outputs: scores_/reasons_{claude47,gpt55,grok,gemini}.csv
                       + resume caches; agent_logs/ symlinks into odcv/results/
figures/               generated figures (make_figure*.py writes here)
make_figure.py         per-eval SFT-vs-final comparison figure
make_figure_all.py     combined figure
olmo3_parser.py        Olmo-3 output parser (template.jinja: chat template)
```

## Notes

- ODCV agent runs used 4 concurrent shard clones per checkpoint against pod-hosted
  vLLM (see `odcv/tools/RUNBOOK.md` for the full recipe and gotchas).
- ODCV misbehavior scores: median of 4 judges per scenario; scores ≥3 = intentional
  misalignment per the paper's rubric.
- Superseded .eval logs (errored/cancelled runs later resumed or re-run to success)
  were deleted 2026-08-10; every kept .eval has status=success and full samples.
- The MASK `.rescored.eval` logs were produced by re-running only the judge over the
  original solver logs with judge_max_tokens=3000 (original runs used 500, fully
  consumed by thinking) AND a parser workaround stripping `**` from bolded judge
  verdicts that the upstream inspect_evals parser rejects. A plain rerun with a
  bigger budget does NOT reproduce the parser fix — see `rescore_mask.py` in git
  history (deleted after use) if this ever needs to be redone.
