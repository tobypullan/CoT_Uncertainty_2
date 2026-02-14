# CoT_Uncertainty_2

Local experiments for Qwen3-32B focused on MMLU-Pro `/nothink` behavior and CoT-fraction probing.

## Quick start on a fresh machine

```bash
git clone <your-repo-url>
cd CoT_Uncertainty_2
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Then run:

```bash
python experiments/benchmarks/local_model.py --model-id Qwen/Qwen3-32B
python experiments/mmlu_pro/no_think.py
python experiments/mmlu_pro/CoT_questioning.py --max-new-tokens-cot 4096
```

## Repo layout

- `experiments/benchmarks/local_model.py`: local throughput/latency benchmark.
- `experiments/mmlu_pro/no_think.py`: full validation split with `/nothink`.
- `experiments/mmlu_pro/CoT_questioning.py`: CoT baseline + fraction probing on prior misses.
- `results/mmlu_pro/validation/`: run artifacts (`.jsonl`, `.json`, `.png`, progress logs).
- `README_CODEX.md`: handoff/context doc for a new Codex instance.

## Output naming

Use:

`<dataset_or_task>_<split_or_setting>_<model_or_variant>_<experiment_tag>.<ext>`

Examples:

- `mmlu_pro_validation_nothink_qwen3_32b.jsonl`
- `cot_questioning_qwen3_32b.json`
- `cot_questioning_qwen3_32b.progress.jsonl`
