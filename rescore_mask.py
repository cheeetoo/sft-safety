"""Re-score existing MASK eval logs with a fixed judge config.

The original runs used judge_max_tokens=500 with reasoning enabled on
claude-haiku-4-5: thinking consumed the whole token budget, nearly every
judge response came back with empty visible text, and 96-98% of samples
scored as 'error'. Generation (solver phase, including belief elicitation)
is already in the logs, so re-scoring only re-runs the judge calls —
no vLLM/GPU needed, just ANTHROPIC_API_KEY.

Dry-run on a few samples first (prints score categories, writes nothing):

    .venv/bin/python rescore_mask.py logs/mask/final/<run>.eval --limit 10

Full re-score (writes <run>.rescored.eval next to the original):

    .venv/bin/python rescore_mask.py logs/mask/final/<run>.eval logs/mask/sft/<run>.eval
"""

import argparse
from collections import Counter
from pathlib import Path

from inspect_ai import score
from inspect_ai.log import EvalLog, read_eval_log, write_eval_log
from inspect_ai.model import GenerateConfig, ModelOutput
import inspect_evals.mask.judge_estimate.parse as mask_parse
from inspect_evals.mask.compat import _reduce
from inspect_evals.mask.scorer import build_accuracy_and_honesty_scorer
from inspect_evals.mask.types import NoBeliefConsistencyHandling, QuestionArchetype

# claude-haiku-4-5 bolds its verdict ("Final Answer: **B**"). The upstream
# parser only strips trailing punctuation from the captured token, so the
# leading asterisks survive and the answer-map lookup fails -> 'error'.
_orig_parse_binary = mask_parse.parse_binary_proposition_response


def _parse_binary_tolerant(proposition_response_typevar, judge_response, answer_map, error_case):
    return _orig_parse_binary(
        proposition_response_typevar,
        judge_response.replace("**", ""),
        answer_map,
        error_case,
    )


mask_parse.parse_binary_proposition_response = _parse_binary_tolerant


def category_counts(log: EvalLog) -> dict[str, Counter]:
    counts: dict[str, Counter] = {"accuracy": Counter(), "honesty": Counter()}
    for sample in log.samples or []:
        s = (sample.scores or {}).get("accuracy_and_honesty")
        if s is None or not isinstance(s.value, dict):
            counts["accuracy"]["<unscored>"] += 1
            counts["honesty"]["<unscored>"] += 1
            continue
        counts["accuracy"][str(s.value.get("accuracy"))] += 1
        counts["honesty"][str(s.value.get("honesty"))] += 1
    return counts


def print_counts(label: str, counts: dict[str, Counter]) -> None:
    print(f"  {label}:")
    for kind, c in counts.items():
        total = sum(c.values())
        parts = ", ".join(f"{k}={n} ({n / total:.0%})" for k, n in c.most_common())
        print(f"    {kind}: {parts}")


def rescore(path: Path, max_tokens: int, limit: int | None) -> None:
    print(f"== {path}")
    log = read_eval_log(str(path))
    if log.samples is None:
        raise SystemExit(f"{path}: no samples in log")

    if limit is not None:
        log.samples = log.samples[:limit]
        print(f"  dry run on first {len(log.samples)} samples")

    # The MASK scorer expects the in-memory types a live run produces, but a log
    # round-trip serializes them: metadata enums become strings and the solver's
    # stored ModelOutput lists become dicts. Re-hydrate both before scoring.
    output_store_keys = (
        "pressured_responses",
        "belief_elicit_1_responses",
        "belief_elicit_2_and_3_responses",
    )
    for sample in log.samples:
        if isinstance(sample.metadata.get("config"), str):
            sample.metadata["config"] = QuestionArchetype(sample.metadata["config"])
        for key in output_store_keys:
            outputs = sample.store.get(key)
            if outputs:
                sample.store[key] = [
                    ModelOutput.model_validate(o) if isinstance(o, dict) else o
                    for o in outputs
                ]

    print_counts("before", category_counts(log))

    ta = log.eval.task_args or {}
    judge_config = GenerateConfig(
        max_tokens=max_tokens,
        temperature=ta.get("judge_temperature", 1.0),
        reasoning_effort=ta.get("judge_reasoning_effort", "low"),
    )
    scorer = build_accuracy_and_honesty_scorer(
        include_core=ta.get("include_core_metrics", True),
        include_statistical_summary=ta.get("include_statistical_summary", True),
        include_normalisation=ta.get("include_normalisation", True),
        include_stratification=ta.get("include_stratification", True),
    )(
        honesty_at_n=ta.get("honesty_at_n", 1),
        binary_judge_model=ta["binary_judge_model"],
        numeric_judge_model=ta["numeric_judge_model"],
        numeric_tol=ta.get("numeric_tol", 0.05),
        no_belief_handling=NoBeliefConsistencyHandling(
            ta.get("no_belief_handling", "tolerant")
        ),
        judge_config=judge_config,
    )

    scored = score(
        log,
        scorer,
        epochs_reducer=_reduce(),
        action="overwrite",
        display="plain",
    )

    print_counts("after", category_counts(scored))
    for result in (scored.results.scores if scored.results else [])[:1]:
        headline = {
            k: v.value
            for k, v in result.metrics.items()
            if k.startswith("overall_")
        }
        print(f"  headline metrics: {headline}")

    if limit is None:
        out = path.with_suffix(".rescored.eval")
        write_eval_log(scored, str(out))
        print(f"  wrote {out}")
    else:
        print("  dry run: nothing written")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="MASK .eval log(s)")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=3000,
        help="judge max_tokens (original bug: 500, fully consumed by thinking)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="dry run: score only the first N samples and write nothing",
    )
    args = parser.parse_args()
    for path in args.logs:
        rescore(path, args.max_tokens, args.limit)


if __name__ == "__main__":
    main()
