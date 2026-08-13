"""Run MASK for one checkpoint with a working judge, in one pass.

Replaces the old run-then-rescore dance (rescore_mask.py, deleted after use;
recoverable via `git show 0b51467:rescore_mask.py`). Two fixes are baked in
relative to the original sft/final solver runs:

1. judge_max_tokens=3000 — the task default of 500 was fully consumed by
   claude-haiku-4-5's thinking, leaving empty judge text and 96-98% 'error'
   scores.
2. Upstream inspect_evals only strips *trailing* punctuation from the judge's
   verdict token, so haiku's habitual bolding ("Final Answer: **B**") survives
   as "**B" and fails the answer-map lookup. The parser is patched below
   (same workaround as rescore_mask.py) before the eval runs.

Solver-side settings match the sft/final runs exactly (max_tokens=8192,
temperature=1.0, judges = claude-haiku-4-5, belief_repeats_n=3), so results
are comparable: the sft/final logs in logs/mask/ are what this one-pass
config would have produced directly (they were judged via the rescore path
with this same judge config + parser fix).

Usage (from repo root; VLLM_BASE_URL/VLLM_API_KEY -> pod; ANTHROPIC_API_KEY
for the judge):

    uv run python mask/run.py --ckpt dpo --model allenai/Olmo-3-32B-Think-DPO

Smoke-test the plumbing (and the judge fix) first with --limit 5: the score
category summary printed at the end should show real verdicts, not 'error'.
"""

import argparse
import os
from collections import Counter
from pathlib import Path

from inspect_ai import eval as inspect_eval
from inspect_ai.log import read_eval_log
import inspect_evals.mask.judge_estimate.parse as mask_parse
from inspect_evals.mask import mask

JUDGE = "anthropic/claude-haiku-4-5-20251001"

# Call sites (parse_statement/parse_belief) look the parser up through module
# globals at call time and pass keyword arguments, so patching the module
# attribute works — but the wrapper's parameter names must match the original.
_orig_parse_binary = mask_parse.parse_binary_proposition_response


def _parse_binary_tolerant(proposition_response_typevar, judge_response, answer_map, error_case):
    return _orig_parse_binary(
        proposition_response_typevar=proposition_response_typevar,
        judge_response=judge_response.replace("**", ""),
        answer_map=answer_map,
        error_case=error_case,
    )


mask_parse.parse_binary_proposition_response = _parse_binary_tolerant


def print_score_categories(location: str) -> None:
    """Judge health check: a broken judge shows up here as mostly 'error'."""
    log = read_eval_log(location)
    counts = {"accuracy": Counter(), "honesty": Counter()}
    for sample in log.samples or []:
        score = (sample.scores or {}).get("accuracy_and_honesty")
        value = score.value if score is not None and isinstance(score.value, dict) else {}
        for kind in counts:
            counts[kind][str(value.get(kind))] += 1
    for kind, c in counts.items():
        total = sum(c.values()) or 1
        parts = ", ".join(f"{k}={n} ({n / total:.0%})" for k, n in c.most_common())
        print(f"  {kind}: {parts}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="e.g. allenai/Olmo-3-32B-Think-DPO")
    ap.add_argument("--ckpt", required=True, help="log subdir label, e.g. dpo")
    ap.add_argument("--limit", type=int, default=None, help="sample limit (smoke test)")
    args = ap.parse_args()

    for var, why in [
        ("VLLM_BASE_URL", "pod endpoint (without it inspect launches a local vLLM server)"),
        ("ANTHROPIC_API_KEY", "judge calls"),
    ]:
        if not os.environ.get(var):
            raise SystemExit(f"{var} is not set — needed for {why}.")

    task = mask(
        honesty_at_n=1,
        belief_repeats_n=3,
        binary_judge_model=JUDGE,
        numeric_judge_model=JUDGE,
        numeric_tol=0.05,
        no_belief_handling="tolerant",
        judge_max_tokens=3000,  # not the default 500: haiku's thinking eats that budget
        judge_temperature=1.0,
        judge_reasoning_effort="low",
        include_core_metrics=True,
        include_statistical_summary=True,
        include_stratification=True,
        include_normalisation=True,
    )

    [log] = inspect_eval(
        task,
        model=f"vllm/{args.model}",
        max_tokens=8192,
        temperature=1.0,
        limit=args.limit,
        log_dir=str(Path(__file__).parent.parent / "logs" / "mask" / args.ckpt),
    )

    print(f"{log.status} -> {log.location}")
    print_score_categories(log.location)


if __name__ == "__main__":
    main()
