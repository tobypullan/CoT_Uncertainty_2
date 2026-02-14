# Codex Handoff

## What this repo is

This repo runs local Qwen3-32B experiments for MMLU-Pro:

1. `/nothink` baseline over validation split.
2. CoT follow-up on questions missed by the `/nothink` run.
3. Fractional CoT probing (1/10 ... 10/10) with forced immediate answers.

## Important scripts

- `experiments/benchmarks/local_model.py`
- `experiments/mmlu_pro/no_think.py`
- `experiments/mmlu_pro/CoT_questioning.py`

## Expected data flow

1. Run `no_think.py` first.
2. It writes:
   - `results/mmlu_pro/validation/mmlu_pro_validation_nothink_qwen3_32b.jsonl`
3. Run `CoT_questioning.py` next.
4. It reads incorrect question IDs from the no-think JSONL, then writes:
   - `results/mmlu_pro/validation/cot_questioning_qwen3_32b.json`
   - `results/mmlu_pro/validation/cot_questioning_qwen3_32b.png`
   - `results/mmlu_pro/validation/cot_questioning_qwen3_32b.progress.jsonl`

## Full run command used for CoT questioning

```bash
PYTHONUNBUFFERED=1 python experiments/mmlu_pro/CoT_questioning.py \
  --max-new-tokens-cot 4096 \
  --cot-batch-size 4 \
  --fraction-batch-size 10 \
  --attn-impl auto \
  --cache-implementation auto \
  --checkpoint-every 1 \
  --output-json results/mmlu_pro/validation/cot_questioning_qwen3_32b.json \
  --output-figure results/mmlu_pro/validation/cot_questioning_qwen3_32b.png \
  --progress-jsonl results/mmlu_pro/validation/cot_questioning_qwen3_32b.progress.jsonl
```

## Current behavioral notes

- `CoT_questioning.py` now guards against repeated turn/template artifacts in baseline CoT:
  - trims apparent restart markers (`Human:`, `User:`, `<|im_start|>user`, etc.)
  - trims repeated `Final answer:` loops
  - stores both raw and cleaned baseline CoT text in output JSON
- If chat template rendering fails (often `jinja2` mismatch), code falls back to explicit manual chat formatting and emits a warning.
- `requirements.txt` pins `jinja2==3.1.6` to avoid known template issues.

## Quick sanity checks

```bash
python -m py_compile experiments/benchmarks/local_model.py \
  experiments/mmlu_pro/no_think.py \
  experiments/mmlu_pro/CoT_questioning.py

python - <<'PY'
import jinja2
print("jinja2", jinja2.__version__)
PY
```
