# impossiblebench (vendored)

Trimmed vendored copy of [ImpossibleBench](https://github.com/fjzzq2002/impossiblebench)
(MIT, see LICENSE), installed as an editable path dependency of this repo.

Kept: the `impossiblebench` package in `src/` (Inspect AI tasks, agents, scorers,
analysis helpers). Removed: upstream README, demo script, figure, and the `gen/`
dataset-generation scripts — the mutated benchmark datasets are downloaded from
HuggingFace at runtime, so `gen/` is only needed to regenerate them from scratch.
