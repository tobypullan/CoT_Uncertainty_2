# Experiments

Put each new experiment script in a task-specific subfolder.

Current:

- `benchmarks/local_model.py`: local model throughput and latency checks.
- `mmlu_pro/no_think.py`: MMLU-Pro validation evaluation with `/nothink` prompting.
- `mmlu_pro/CoT_questioning.py`: CoT-fraction probing on questions missed in the no-think run.
