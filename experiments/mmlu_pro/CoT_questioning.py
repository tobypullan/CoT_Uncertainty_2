#!/usr/bin/env python3
"""CoT-fraction probing with configurable question pools and fraction-stage scope."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

LETTER_POOL = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CACHE_IMPL_FALLBACK_WARNED = False
CHAT_TEMPLATE_FALLBACK_WARNED = False
SYSTEM_PROMPT = "You are a precise assistant for multiple-choice questions."

TURN_RESTART_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    ("human_tag", re.compile(r"\n\s*Human\s*:", flags=re.IGNORECASE)),
    ("user_tag", re.compile(r"\n\s*User\s*:", flags=re.IGNORECASE)),
    ("assistant_tag", re.compile(r"\n\s*Assistant\s*:", flags=re.IGNORECASE)),
    ("im_start_user", re.compile(r"<\|im_start\|>\s*user", flags=re.IGNORECASE)),
    ("im_start_assistant", re.compile(r"<\|im_start\|>\s*assistant", flags=re.IGNORECASE)),
]
FINAL_ANSWER_LINE_RE = re.compile(r"(?im)^\s*final\s*answer\s*:\s*[A-Z](?:[^\n]*)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe how partial CoT affects /nothink answers on selected MMLU-Pro questions."
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3-32B", help="HF model ID")
    parser.add_argument(
        "--nothink-results",
        default="results/mmlu_pro/validation/mmlu_pro_validation_nothink_qwen3_32b.jsonl",
        help="Path to prior no-think jsonl result file",
    )
    parser.add_argument("--dataset-id", default="TIGER-Lab/MMLU-Pro", help="HF dataset ID")
    parser.add_argument("--split", default="validation", help="Dataset split")
    parser.add_argument(
        "--output-json",
        default="results/mmlu_pro/validation/cot_questioning_qwen3_32b.json",
        help="Output JSON with all experiment records",
    )
    parser.add_argument(
        "--output-figure",
        default="results/mmlu_pro/validation/cot_questioning_qwen3_32b.png",
        help="Output figure path",
    )
    parser.add_argument("--device-map", default="auto", help='HF device_map (e.g. "auto")')
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="Model dtype",
    )
    parser.add_argument(
        "--attn-impl",
        choices=["auto", "sdpa", "flash_attention_2", "eager"],
        default="auto",
        help="Attention implementation",
    )
    parser.add_argument(
        "--max-new-tokens-cot",
        type=int,
        default=4096,
        help="Generation budget for full CoT baseline",
    )
    parser.add_argument(
        "--max-new-tokens-answer",
        type=int,
        default=12,
        help="Generation budget for forced immediate-answer runs",
    )
    parser.add_argument(
        "--max-new-tokens-recovery",
        type=int,
        default=16,
        help="Generation budget for truncated-CoT answer recovery",
    )
    parser.add_argument(
        "--limit-questions",
        type=int,
        default=None,
        help="Optional cap on number of selected questions to process",
    )
    parser.add_argument(
        "--question-pool",
        default="incorrect",
        choices=["incorrect", "all"],
        help="Which question pool to process: prior /nothink incorrect only, or all dataset questions.",
    )
    parser.add_argument(
        "--fraction-stage-scope",
        default="baseline_correct_only",
        choices=["baseline_correct_only", "all"],
        help="Run fraction-stage probes only when baseline CoT is correct, or for all selected questions.",
    )
    parser.add_argument(
        "--cot-batch-size",
        type=int,
        default=4,
        help="Batch size for baseline CoT (increase to improve throughput if memory allows).",
    )
    parser.add_argument(
        "--fraction-batch-size",
        type=int,
        default=10,
        help="Batch size for fraction-stage /nothink probes.",
    )
    parser.add_argument(
        "--cache-implementation",
        default="auto",
        choices=["auto", "dynamic", "static", "offloaded", "offloaded_static", "quantized"],
        help="KV cache backend for generation. 'static' can improve decode throughput.",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Compile model forward for speed (only when model is on a single device).",
    )
    parser.add_argument(
        "--compile-mode",
        default="reduce-overhead",
        choices=["reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
        help="torch.compile mode",
    )
    parser.add_argument(
        "--progress-jsonl",
        default="",
        help="Streaming JSONL path for incremental per-question results (default: <output-json>.progress.jsonl).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Write checkpoint JSON after this many processed questions.",
    )
    parser.add_argument(
        "--cuda-sync-timing",
        action="store_true",
        help="Synchronize CUDA around generation calls for stricter timing (slower).",
    )
    parser.add_argument(
        "--skip-fraction-stage",
        action="store_true",
        help="Only run baseline CoT step (skip 1/10..10/10 fraction probes).",
    )
    parser.add_argument(
        "--write-baseline-to-nothink-jsonl",
        action="store_true",
        help="Write baseline CoT fields for processed questions back into --nothink-results file.",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Disable trust_remote_code",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else torch.float32


def parse_incorrect_qids(path: Path) -> List[int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    qids: List[int] = []
    for line in lines:
        row = json.loads(line)
        if row.get("type") == "example" and not row.get("is_correct", False):
            qids.append(int(row["question_id"]))
    return sorted(set(qids))


def build_chat_text(tokenizer: Any, user_prompt: str) -> str:
    global CHAT_TEMPLATE_FALLBACK_WARNED
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as exc:
        if not CHAT_TEMPLATE_FALLBACK_WARNED:
            print(
                "Warning: tokenizer.apply_chat_template failed. "
                "Falling back to manual chat template formatting."
            )
            print(f"  Cause: {type(exc).__name__}: {exc}")
            CHAT_TEMPLATE_FALLBACK_WARNED = True
        im_start = "<|im_start|>"
        im_end = "<|im_end|>"
        return (
            f"{im_start}system\n{SYSTEM_PROMPT}{im_end}\n"
            f"{im_start}user\n{user_prompt}{im_end}\n"
            f"{im_start}assistant\n"
        )


def clean_cot_response_text(text: str) -> Tuple[str, Dict[str, Any]]:
    raw = text or ""
    cleaned = raw.strip()
    meta: Dict[str, Any] = {
        "raw_char_length": len(raw),
        "cleaned_char_length_before": len(cleaned),
        "turn_marker_trimmed": False,
        "turn_marker_type": None,
        "repeated_final_answer_trimmed": False,
        "final_answer_line_count_before": len(FINAL_ANSWER_LINE_RE.findall(cleaned)),
        "final_answer_line_count_after": None,
        "cleanup_applied": False,
    }

    marker_hits: List[Tuple[int, str]] = []
    for marker_name, pattern in TURN_RESTART_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            marker_hits.append((match.start(), marker_name))
    if marker_hits:
        marker_hits.sort(key=lambda x: x[0])
        cut_pos, marker_name = marker_hits[0]
        cleaned = cleaned[:cut_pos].rstrip()
        meta["turn_marker_trimmed"] = True
        meta["turn_marker_type"] = marker_name
        meta["cleanup_applied"] = True

    answer_line_matches = list(FINAL_ANSWER_LINE_RE.finditer(cleaned))
    if len(answer_line_matches) >= 2:
        cleaned = cleaned[: answer_line_matches[1].start()].rstrip()
        meta["repeated_final_answer_trimmed"] = True
        meta["cleanup_applied"] = True

    meta["final_answer_line_count_after"] = len(FINAL_ANSWER_LINE_RE.findall(cleaned))
    meta["cleaned_char_length_after"] = len(cleaned)
    return cleaned, meta


def format_options(options: Sequence[str]) -> str:
    letters = LETTER_POOL[: len(options)]
    return "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(options))


def build_cot_prompt(question: str, options: Sequence[str]) -> str:
    return (
        "Solve this multiple-choice question with full reasoning.\n"
        "Be concise but complete, and end with exactly: Final answer: <LETTER>\n\n"
        f"Question: {question.strip()}\n\n"
        f"Options:\n{format_options(options)}\n"
    )


def build_fraction_prompt(question: str, options: Sequence[str], cot_snippet: str) -> str:
    return (
        "/nothink\n"
        "You are given partial reasoning context. Do not add extra reasoning.\n"
        "Immediately answer the multiple-choice question.\n"
        "Output exactly one uppercase letter only.\n\n"
        f"Question: {question.strip()}\n\n"
        f"Options:\n{format_options(options)}\n\n"
        "Partial reasoning context:\n"
        f"{cot_snippet}\n\n"
        "Answer:"
    )


def build_recovery_prompt(question: str, options: Sequence[str], cot_response: str) -> str:
    return (
        "/nothink\n"
        "The prior reasoning may have been cut off. Using the reasoning context below, output the final answer letter.\n"
        "Return exactly one uppercase letter only.\n\n"
        f"Question: {question.strip()}\n\n"
        f"Options:\n{format_options(options)}\n\n"
        "Reasoning context:\n"
        f"{cot_response}\n\n"
        "Final answer:"
    )


def extract_prediction(response: str, num_options: int) -> Tuple[Optional[str], str]:
    allowed = LETTER_POOL[:num_options]
    text = response.upper()
    cleaned = text.strip()
    escaped = re.escape(allowed)

    if re.fullmatch(rf"[{escaped}]", cleaned):
        return cleaned, "exact_single_letter"

    patterns = [
        rf"FINAL\s*ANSWER\s*[:\-]?\s*\(?\s*([{escaped}])\s*\)?",
        rf"THE\s+CORRECT\s+ANSWER\s+IS\s*\(?\s*([{escaped}])\s*\)?",
        rf"THE\s+ANSWER\s+IS\s*\(?\s*([{escaped}])\s*\)?",
        rf"\bANSWER\s*[:\-]?\s*\(?\s*([{escaped}])\s*\)?",
        rf"\bOPTION\s*[:\-]?\s*\(?\s*([{escaped}])\s*\)?",
        rf"\bCHOICE\s*[:\-]?\s*\(?\s*([{escaped}])\s*\)?",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1], "explicit_pattern"

    tail = text[-600:]
    tail_matches = re.findall(rf"\b([{escaped}])\b", tail)
    if tail_matches:
        return tail_matches[-1], "tail_standalone_letter"

    all_matches = re.findall(rf"\b([{escaped}])\b", text)
    if all_matches:
        return all_matches[-1], "last_standalone_letter"

    start_match = re.search(rf"^\s*\(?\s*([{escaped}])\s*\)?(?:[\)\.\:\-\s]|$)", text)
    if start_match:
        return start_match.group(1), "start_letter"

    return None, "none"


def model_uses_cuda(model: Any) -> bool:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for location in device_map.values():
            if isinstance(location, str) and location.startswith("cuda"):
                return True
            if isinstance(location, int):
                return True
            if isinstance(location, torch.device) and location.type == "cuda":
                return True
        return False
    try:
        return next(model.parameters()).device.type == "cuda"
    except StopIteration:
        return False


def sync_if_needed(enabled: bool) -> None:
    if enabled:
        torch.cuda.synchronize()


def load_model_and_tokenizer(args: argparse.Namespace) -> Tuple[Any, Any, torch.dtype, bool]:
    trust_remote_code = not args.no_trust_remote_code
    dtype = resolve_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=trust_remote_code)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {
        "dtype": dtype,
        "device_map": args.device_map,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if args.attn_impl != "auto":
        model_kwargs["attn_implementation"] = args.attn_impl

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    except TypeError as exc:
        if "unexpected keyword argument 'dtype'" not in str(exc):
            raise
        fallback = dict(model_kwargs)
        fallback.pop("dtype", None)
        fallback["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(args.model_id, **fallback)

    model.eval()
    if args.torch_compile:
        if hasattr(torch, "compile") and getattr(model, "hf_device_map", None) is None:
            print(f"Compiling model with torch.compile(mode={args.compile_mode})...")
            model = torch.compile(model, mode=args.compile_mode)
        else:
            print("Skipping torch.compile (unsupported with current device_map/model dispatch).")
    use_cuda_sync = args.cuda_sync_timing and model_uses_cuda(model)
    return model, tokenizer, dtype, use_cuda_sync


def run_generate(
    model: Any,
    inputs: Dict[str, Any],
    tokenizer: Any,
    max_new_tokens: int,
    cache_implementation: str,
) -> Any:
    global CACHE_IMPL_FALLBACK_WARNED
    kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if cache_implementation != "auto":
        kwargs["cache_implementation"] = cache_implementation

    try:
        return model.generate(**inputs, **kwargs)
    except TypeError as exc:
        if "cache_implementation" not in str(exc) or "cache_implementation" not in kwargs:
            raise
        if not CACHE_IMPL_FALLBACK_WARNED:
            print(
                "Warning: cache_implementation is unsupported by this model/transformers build. "
                "Falling back to default cache."
            )
            CACHE_IMPL_FALLBACK_WARNED = True
        kwargs.pop("cache_implementation", None)
        return model.generate(**inputs, **kwargs)


def generate_batch_completions(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    max_new_tokens: int,
    use_cuda_sync: bool,
    cache_implementation: str,
) -> Tuple[List[Dict[str, Any]], float]:
    if not prompts:
        return [], 0.0

    chat_texts = [build_chat_text(tokenizer, p) for p in prompts]
    inputs = tokenizer(chat_texts, return_tensors="pt", padding=True, truncation=False)
    if not getattr(model, "hf_device_map", None):
        target_device = next(model.parameters()).device
        inputs = {k: v.to(target_device) for k, v in inputs.items()}

    gen_t0 = time.perf_counter()
    sync_if_needed(use_cuda_sync)
    with torch.inference_mode():
        out = run_generate(
            model=model,
            inputs=inputs,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            cache_implementation=cache_implementation,
        )
    sync_if_needed(use_cuda_sync)
    generation_seconds = time.perf_counter() - gen_t0

    prompt_len = inputs["input_ids"].shape[1]
    gen_tokens = out[:, prompt_len:]
    pad_id = tokenizer.pad_token_id
    results: List[Dict[str, Any]] = []
    for i in range(len(prompts)):
        seq = gen_tokens[i]
        if pad_id is not None:
            generated_tokens = int((seq != pad_id).sum().item())
        else:
            generated_tokens = int(seq.shape[0])
        decoded = tokenizer.decode(seq, skip_special_tokens=True).strip()
        results.append(
            {
                "text": decoded,
                "generated_tokens": generated_tokens,
                "max_new_tokens": max_new_tokens,
                "truncated": generated_tokens >= max_new_tokens,
            }
        )
    return results, generation_seconds


def cot_fraction_snippet(tokenizer: Any, cot_text: str, numer: int, denom: int) -> Tuple[str, int, int]:
    tokens = tokenizer.encode(cot_text, add_special_tokens=False)
    total = len(tokens)
    if total == 0:
        return cot_text, 0, 0
    keep = max(1, math.ceil(total * numer / denom))
    keep = min(total, keep)
    snippet = tokenizer.decode(tokens[:keep], skip_special_tokens=True)
    return snippet, keep, total


def render_figure(
    output_path: Path,
    records: List[Dict[str, Any]],
    title_suffix: str,
) -> Dict[str, float]:
    eligible = [r for r in records if r["fraction_trials"]]
    if not eligible:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "No questions reached fraction stage.", ha="center", va="center")
        ax.set_title(f"CoT Fraction Probing ({title_suffix})")
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return {}

    eligible = sorted(eligible, key=lambda x: x["question_id"])
    matrix = np.array([[1 if t["is_correct"] else 0 for t in r["fraction_trials"]] for r in eligible], dtype=float)
    frac_accuracy = matrix.mean(axis=0)
    x = np.arange(1, 11)

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1, 3], hspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, frac_accuracy, marker="o", linewidth=2)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xticks(x)
    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel("CoT Fraction (k/10)")
    ax1.set_title(f"Fraction Accuracy ({title_suffix})")
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = fig.add_subplot(gs[1, 0])
    im = ax2.imshow(matrix, aspect="auto", interpolation="nearest", cmap="RdYlGn", vmin=0, vmax=1)
    ax2.set_xticks(np.arange(10))
    ax2.set_xticklabels([f"{i}/10" for i in range(1, 11)])
    ax2.set_yticks(np.arange(len(eligible)))
    ax2.set_yticklabels([str(r["question_id"]) for r in eligible])
    ax2.set_xlabel("Provided CoT Fraction")
    ax2.set_ylabel("Question ID")
    ax2.set_title("Per-question correctness (1=correct, 0=incorrect)")
    cbar = fig.colorbar(im, ax=ax2, fraction=0.02, pad=0.02)
    cbar.set_label("Correctness")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {f"{i}/10": float(frac_accuracy[i - 1]) for i in range(1, 11)}


def update_nothink_file(
    path: Path,
    records: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> None:
    by_qid = {int(r["question_id"]): r for r in records}
    updated_lines: List[str] = []
    lines = path.read_text(encoding="utf-8").splitlines()

    for line in lines:
        row = json.loads(line)
        if row.get("type") == "example":
            qid = int(row["question_id"])
            if qid in by_qid:
                rec = by_qid[qid]
                row.update(
                    {
                        "baseline_cot_response_raw": rec["baseline_cot_response_raw"],
                        "baseline_cot_response": rec["baseline_cot_response"],
                        "baseline_cot_cleanup": rec["baseline_cot_cleanup"],
                        "baseline_cot_cleanup_applied": rec["baseline_cot_cleanup_applied"],
                        "baseline_cot_generated_tokens": rec["baseline_cot_generated_tokens"],
                        "baseline_cot_max_new_tokens": rec["baseline_cot_max_new_tokens"],
                        "baseline_cot_truncated": rec["baseline_cot_truncated"],
                        "baseline_cot_prediction_primary": rec["baseline_cot_prediction_primary"],
                        "baseline_cot_parse_method_primary": rec["baseline_cot_parse_method_primary"],
                        "baseline_cot_recovery_used": rec["baseline_cot_recovery_used"],
                        "baseline_cot_recovery_response": rec["baseline_cot_recovery_response"],
                        "baseline_cot_recovery_prediction": rec["baseline_cot_recovery_prediction"],
                        "baseline_cot_recovery_parse_method": rec["baseline_cot_recovery_parse_method"],
                        "baseline_cot_prediction": rec["baseline_cot_prediction"],
                        "baseline_cot_prediction_source": rec["baseline_cot_prediction_source"],
                        "baseline_cot_is_correct": rec["baseline_cot_is_correct"],
                    }
                )
        elif row.get("type") == "summary":
            row["baseline_cot_audit"] = {
                "model_id": summary["model_id"],
                "question_pool": summary.get("question_pool", "incorrect"),
                "fraction_stage_scope": summary.get("fraction_stage_scope", "baseline_correct_only"),
                "num_processed_from_selected_pool": summary.get("num_initial_questions"),
                "num_processed_from_incorrect_pool": summary["num_initial_incorrect_questions"],
                "num_baseline_cot_correct": summary["num_baseline_cot_correct"],
                "num_baseline_cot_incorrect": summary["num_baseline_cot_incorrect"],
                "num_baseline_cot_truncated": summary["num_baseline_cot_truncated"],
                "num_recovery_used": summary["num_recovery_used"],
                "num_cot_cleanup_applied": summary["num_cot_cleanup_applied"],
                "num_cot_turn_marker_trimmed": summary["num_cot_turn_marker_trimmed"],
                "num_cot_repeated_final_answer_trimmed": summary["num_cot_repeated_final_answer_trimmed"],
                "fraction_accuracy": summary["fraction_accuracy"],
            }
        updated_lines.append(json.dumps(row, ensure_ascii=False))

    path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")


def write_json_payload(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def build_summary(
    args: argparse.Namespace,
    nothink_path: Path,
    dtype: torch.dtype,
    records: List[Dict[str, Any]],
    total_questions: int,
    truncated_count: int,
    recovery_used_count: int,
    generated_tokens_total: int,
    baseline_cot_tokens_total: int,
    recovery_tokens_total: int,
    fraction_tokens_total: int,
    generation_wall_seconds: float,
    runtime_seconds: float,
    frac_accuracy: Dict[str, float],
    fig_path: Path,
    is_final: bool,
) -> Dict[str, Any]:
    eligible = [r for r in records if r["baseline_cot_is_correct"]]
    cleanup_applied_count = sum(1 for r in records if r.get("baseline_cot_cleanup_applied"))
    cleanup_turn_marker_count = sum(
        1
        for r in records
        if isinstance(r.get("baseline_cot_cleanup"), dict) and r["baseline_cot_cleanup"].get("turn_marker_trimmed")
    )
    cleanup_repeat_final_answer_count = sum(
        1
        for r in records
        if isinstance(r.get("baseline_cot_cleanup"), dict)
        and r["baseline_cot_cleanup"].get("repeated_final_answer_trimmed")
    )
    return {
        "nothink_results_path": str(nothink_path),
        "dataset_id": args.dataset_id,
        "split": args.split,
        "model_id": args.model_id,
        "question_pool": args.question_pool,
        "fraction_stage_scope": args.fraction_stage_scope,
        "dtype": str(dtype),
        "device_map": args.device_map,
        "attn_implementation": args.attn_impl,
        "cache_implementation": args.cache_implementation,
        "torch_compile": args.torch_compile,
        "compile_mode": args.compile_mode if args.torch_compile else None,
        "cot_batch_size": args.cot_batch_size,
        "fraction_batch_size": args.fraction_batch_size,
        "max_new_tokens_cot": args.max_new_tokens_cot,
        "max_new_tokens_recovery": args.max_new_tokens_recovery,
        "max_new_tokens_answer": args.max_new_tokens_answer,
        "num_initial_questions": total_questions,
        "num_initial_incorrect_questions": total_questions,
        "num_processed_questions": len(records),
        "is_final": is_final,
        "num_baseline_cot_correct": len(eligible),
        "num_baseline_cot_incorrect": len(records) - len(eligible),
        "num_fraction_stage_questions": sum(1 for r in records if r["fraction_trials"]),
        "num_baseline_cot_truncated": truncated_count,
        "num_recovery_used": recovery_used_count,
        "num_cot_cleanup_applied": cleanup_applied_count,
        "num_cot_turn_marker_trimmed": cleanup_turn_marker_count,
        "num_cot_repeated_final_answer_trimmed": cleanup_repeat_final_answer_count,
        "generated_tokens_total": generated_tokens_total,
        "generated_tokens_baseline_cot": baseline_cot_tokens_total,
        "generated_tokens_recovery": recovery_tokens_total,
        "generated_tokens_fraction_stage": fraction_tokens_total,
        "generation_seconds": generation_wall_seconds,
        "overall_tokens_per_second": (generated_tokens_total / runtime_seconds) if runtime_seconds > 0 else 0.0,
        "generation_tokens_per_second": (
            (generated_tokens_total / generation_wall_seconds) if generation_wall_seconds > 0 else 0.0
        ),
        "fraction_accuracy": frac_accuracy,
        "figure_path": str(fig_path),
        "runtime_seconds": runtime_seconds,
    }


def main() -> None:
    args = parse_args()
    run_start = time.perf_counter()
    if args.cot_batch_size < 1:
        raise ValueError("--cot-batch-size must be >= 1")
    if args.fraction_batch_size < 1:
        raise ValueError("--fraction-batch-size must be >= 1")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be >= 1")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    output_json = Path(args.output_json)
    fig_path = Path(args.output_figure)
    progress_path = (
        Path(args.progress_jsonl)
        if args.progress_jsonl
        else output_json.with_suffix(".progress.jsonl")
    )

    nothink_path = Path(args.nothink_results)
    print(f"Streaming progress to: {progress_path}")
    print("Loading dataset...")
    dataset = load_dataset(args.dataset_id, split=args.split)
    ds_by_qid: Dict[int, Dict[str, Any]] = {int(row["question_id"]): row for row in dataset}

    if args.question_pool == "incorrect":
        if not nothink_path.exists():
            raise FileNotFoundError(f"Missing no-think result file: {nothink_path}")
        selected_qids = parse_incorrect_qids(nothink_path)
        if not selected_qids:
            raise RuntimeError("No incorrect questions found in no-think result file.")
        print(f"Loaded incorrect-question pool from {nothink_path}")
    else:
        selected_qids = sorted(ds_by_qid.keys())
        if not selected_qids:
            raise RuntimeError(f"No questions found in dataset split {args.split}.")
        print(f"Loaded full-question pool from {args.dataset_id}:{args.split}")

    if args.limit_questions is not None:
        selected_qids = selected_qids[: args.limit_questions]
    if not selected_qids:
        raise RuntimeError("No questions selected after applying limits.")

    total_q = len(selected_qids)
    print(f"Selected {total_q} questions (question_pool={args.question_pool})")

    missing = [qid for qid in selected_qids if qid not in ds_by_qid]
    if missing:
        raise RuntimeError(f"Question IDs missing from dataset split {args.split}: {missing[:8]}")

    print("Loading model/tokenizer...")
    model_t0 = time.perf_counter()
    model, tokenizer, dtype, use_cuda_sync = load_model_and_tokenizer(args)
    print(
        f"Model ready in {time.perf_counter() - model_t0:.2f}s | "
        f"dtype={dtype} | device_map={args.device_map} | cuda_sync={use_cuda_sync}"
    )

    records: List[Dict[str, Any]] = []
    baseline_cot_correct_count = 0
    truncated_count = 0
    recovery_used_count = 0
    generated_tokens_total = 0
    baseline_cot_tokens_total = 0
    recovery_tokens_total = 0
    fraction_tokens_total = 0
    generation_wall_seconds = 0.0

    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("w", encoding="utf-8") as f_progress:
        f_progress.write(
            json.dumps(
                {
                    "type": "run_start",
                    "timestamp": time.time(),
                    "dataset_id": args.dataset_id,
                    "split": args.split,
                    "model_id": args.model_id,
                    "total_questions": total_q,
                    "question_pool": args.question_pool,
                    "fraction_stage_scope": args.fraction_stage_scope,
                    "max_new_tokens_cot": args.max_new_tokens_cot,
                    "cot_batch_size": args.cot_batch_size,
                    "fraction_batch_size": args.fraction_batch_size,
                    "attn_implementation": args.attn_impl,
                    "cache_implementation": args.cache_implementation,
                    "checkpoint_every": args.checkpoint_every,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        f_progress.flush()

        for batch_start in range(0, total_q, args.cot_batch_size):
            batch_qids = selected_qids[batch_start : batch_start + args.cot_batch_size]
            batch_examples = [ds_by_qid[qid] for qid in batch_qids]
            batch_end = batch_start + len(batch_qids)
            print(f"\n[Batch] baseline CoT questions {batch_start + 1}-{batch_end}/{total_q}")

            cot_prompts = [build_cot_prompt(ex["question"], ex["options"]) for ex in batch_examples]
            cot_outs, cot_gen_s = generate_batch_completions(
                model=model,
                tokenizer=tokenizer,
                prompts=cot_prompts,
                max_new_tokens=args.max_new_tokens_cot,
                use_cuda_sync=use_cuda_sync,
                cache_implementation=args.cache_implementation,
            )
            generation_wall_seconds += cot_gen_s
            batch_cot_tokens = sum(int(out["generated_tokens"]) for out in cot_outs)
            generated_tokens_total += batch_cot_tokens
            baseline_cot_tokens_total += batch_cot_tokens

            cot_cleaned_texts: List[str] = []
            cot_cleanup_metas: List[Dict[str, Any]] = []
            for cot_out in cot_outs:
                cleaned_text, cleanup_meta = clean_cot_response_text(cot_out["text"])
                cot_cleaned_texts.append(cleaned_text)
                cot_cleanup_metas.append(cleanup_meta)

            recovery_indices: List[int] = []
            recovery_prompts: List[str] = []
            for local_i, cot_out in enumerate(cot_outs):
                if cot_out["truncated"]:
                    truncated_count += 1
                    recovery_used_count += 1
                    ex = batch_examples[local_i]
                    recovery_indices.append(local_i)
                    recovery_prompts.append(
                        build_recovery_prompt(ex["question"], ex["options"], cot_cleaned_texts[local_i])
                    )

            recovery_by_local_i: Dict[int, Dict[str, Any]] = {}
            if recovery_prompts:
                recovery_outs, recovery_gen_s = generate_batch_completions(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=recovery_prompts,
                    max_new_tokens=args.max_new_tokens_recovery,
                    use_cuda_sync=use_cuda_sync,
                    cache_implementation=args.cache_implementation,
                )
                generation_wall_seconds += recovery_gen_s
                batch_recovery_tokens = sum(int(out["generated_tokens"]) for out in recovery_outs)
                generated_tokens_total += batch_recovery_tokens
                recovery_tokens_total += batch_recovery_tokens
                for idx_in_recovery, local_i in enumerate(recovery_indices):
                    recovery_by_local_i[local_i] = recovery_outs[idx_in_recovery]

            for local_i, qid in enumerate(batch_qids):
                global_idx = batch_start + local_i + 1
                question_tag = f"[Q {global_idx}/{total_q} qid={qid}]"
                print(f"{question_tag} processing")

                ex = batch_examples[local_i]
                question = ex["question"]
                options = ex["options"]
                gold = ex["answer"].strip().upper()

                cot_out = cot_outs[local_i]
                cot_response_raw = cot_out["text"]
                cot_response = cot_cleaned_texts[local_i]
                cot_cleanup_meta = cot_cleanup_metas[local_i]
                cot_pred_primary, cot_parse_method_primary = extract_prediction(cot_response, len(options))

                recovery_used = local_i in recovery_by_local_i
                recovery_response: Optional[str] = None
                recovery_pred: Optional[str] = None
                recovery_parse_method: Optional[str] = None
                if recovery_used:
                    recovery_out = recovery_by_local_i[local_i]
                    recovery_response = recovery_out["text"]
                    recovery_pred, recovery_parse_method = extract_prediction(recovery_response, len(options))

                if recovery_used and recovery_pred is not None:
                    cot_pred_final = recovery_pred
                    cot_pred_source = "recovery"
                else:
                    cot_pred_final = cot_pred_primary
                    cot_pred_source = "primary"

                baseline_cot_is_correct = cot_pred_final == gold
                if baseline_cot_is_correct:
                    baseline_cot_correct_count += 1

                q_record: Dict[str, Any] = {
                    "question_id": qid,
                    "question": question,
                    "options": options,
                    "gold_answer": gold,
                    "baseline_cot_response_raw": cot_response_raw,
                    "baseline_cot_response": cot_response,
                    "baseline_cot_cleanup": cot_cleanup_meta,
                    "baseline_cot_cleanup_applied": bool(cot_cleanup_meta.get("cleanup_applied", False)),
                    "baseline_cot_generated_tokens": cot_out["generated_tokens"],
                    "baseline_cot_max_new_tokens": cot_out["max_new_tokens"],
                    "baseline_cot_truncated": cot_out["truncated"],
                    "baseline_cot_prediction_primary": cot_pred_primary,
                    "baseline_cot_parse_method_primary": cot_parse_method_primary,
                    "baseline_cot_recovery_used": recovery_used,
                    "baseline_cot_recovery_response": recovery_response,
                    "baseline_cot_recovery_prediction": recovery_pred,
                    "baseline_cot_recovery_parse_method": recovery_parse_method,
                    "baseline_cot_prediction": cot_pred_final,
                    "baseline_cot_prediction_source": cot_pred_source,
                    "baseline_cot_is_correct": baseline_cot_is_correct,
                    "fraction_trials": [],
                }

                should_run_fraction_stage = (
                    not args.skip_fraction_stage
                    and (args.fraction_stage_scope == "all" or baseline_cot_is_correct)
                )
                if should_run_fraction_stage:
                    fraction_metadata: List[Tuple[int, int, int, str]] = []
                    for fraction_idx in range(1, 11):
                        snippet, used_tok, total_tok = cot_fraction_snippet(tokenizer, cot_response, fraction_idx, 10)
                        fraction_prompt = build_fraction_prompt(question, options, snippet)
                        fraction_metadata.append((fraction_idx, used_tok, total_tok, fraction_prompt))

                    for frac_start in range(0, len(fraction_metadata), args.fraction_batch_size):
                        frac_batch = fraction_metadata[frac_start : frac_start + args.fraction_batch_size]
                        frac_begin_idx = frac_batch[0][0]
                        frac_end_idx = frac_batch[-1][0]
                        print(f"{question_tag} resamples {frac_begin_idx}/10 to {frac_end_idx}/10")

                        frac_outs, frac_gen_s = generate_batch_completions(
                            model=model,
                            tokenizer=tokenizer,
                            prompts=[item[3] for item in frac_batch],
                            max_new_tokens=args.max_new_tokens_answer,
                            use_cuda_sync=use_cuda_sync,
                            cache_implementation=args.cache_implementation,
                        )
                        generation_wall_seconds += frac_gen_s
                        batch_fraction_tokens = sum(int(out["generated_tokens"]) for out in frac_outs)
                        generated_tokens_total += batch_fraction_tokens
                        fraction_tokens_total += batch_fraction_tokens

                        for meta, frac_out in zip(frac_batch, frac_outs):
                            fraction_idx, used_tok, total_tok, _ = meta
                            response = frac_out["text"]
                            pred, parse_method = extract_prediction(response, len(options))
                            is_correct = pred == gold
                            q_record["fraction_trials"].append(
                                {
                                    "fraction": fraction_idx / 10.0,
                                    "fraction_label": f"{fraction_idx}/10",
                                    "cot_tokens_used": used_tok,
                                    "cot_tokens_total": total_tok,
                                    "model_response": response,
                                    "parsed_prediction": pred,
                                    "parse_method": parse_method,
                                    "is_correct": is_correct,
                                }
                            )
                            print(f"{question_tag} resample {fraction_idx}/10 correct={is_correct}")

                records.append(q_record)

                elapsed = time.perf_counter() - run_start
                tps = (generated_tokens_total / elapsed) if elapsed > 0 else 0.0
                gen_only_tps = (
                    (generated_tokens_total / generation_wall_seconds) if generation_wall_seconds > 0 else 0.0
                )
                print(
                    f"{question_tag} baseline_cot_correct={baseline_cot_is_correct} "
                    f"truncated={cot_out['truncated']} baseline_cot_correct_total={baseline_cot_correct_count} "
                    f"fraction_trials={len(q_record['fraction_trials'])} "
                    f"gen_tokens={generated_tokens_total} tps={tps:.1f} "
                    f"gen_only_tps={gen_only_tps:.1f} elapsed={elapsed:.1f}s"
                )

                streamed_record = {
                    "type": "question_result",
                    "question_index": global_idx,
                    "total_questions": total_q,
                    **q_record,
                }
                f_progress.write(json.dumps(streamed_record, ensure_ascii=False) + "\n")
                f_progress.flush()

                if global_idx % args.checkpoint_every == 0:
                    checkpoint_runtime = time.perf_counter() - run_start
                    checkpoint_summary = build_summary(
                        args=args,
                        nothink_path=nothink_path,
                        dtype=dtype,
                        records=records,
                        total_questions=total_q,
                        truncated_count=truncated_count,
                        recovery_used_count=recovery_used_count,
                        generated_tokens_total=generated_tokens_total,
                        baseline_cot_tokens_total=baseline_cot_tokens_total,
                        recovery_tokens_total=recovery_tokens_total,
                        fraction_tokens_total=fraction_tokens_total,
                        generation_wall_seconds=generation_wall_seconds,
                        runtime_seconds=checkpoint_runtime,
                        frac_accuracy={},
                        fig_path=fig_path,
                        is_final=False,
                    )
                    write_json_payload(
                        output_json,
                        {
                            "summary": checkpoint_summary,
                            "questions": records,
                        },
                    )
                    f_progress.write(
                        json.dumps(
                            {
                                "type": "checkpoint",
                                "question_index": global_idx,
                                "total_questions": total_q,
                                "runtime_seconds": checkpoint_runtime,
                                "generated_tokens_total": generated_tokens_total,
                                "overall_tokens_per_second": checkpoint_summary["overall_tokens_per_second"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    f_progress.flush()

    frac_accuracy = render_figure(
        output_path=fig_path,
        records=records,
        title_suffix=f"{args.model_id} on {args.dataset_id}:{args.split}",
    )

    runtime_seconds = time.perf_counter() - run_start
    summary = build_summary(
        args=args,
        nothink_path=nothink_path,
        dtype=dtype,
        records=records,
        total_questions=total_q,
        truncated_count=truncated_count,
        recovery_used_count=recovery_used_count,
        generated_tokens_total=generated_tokens_total,
        baseline_cot_tokens_total=baseline_cot_tokens_total,
        recovery_tokens_total=recovery_tokens_total,
        fraction_tokens_total=fraction_tokens_total,
        generation_wall_seconds=generation_wall_seconds,
        runtime_seconds=runtime_seconds,
        frac_accuracy=frac_accuracy,
        fig_path=fig_path,
        is_final=True,
    )

    write_json_payload(
        output_json,
        {
            "summary": summary,
            "questions": records,
        },
    )

    with progress_path.open("a", encoding="utf-8") as f_progress:
        f_progress.write(
            json.dumps(
                {
                    "type": "run_complete",
                    "timestamp": time.time(),
                    "summary": summary,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        f_progress.flush()

    if args.write_baseline_to_nothink_jsonl:
        update_nothink_file(nothink_path, records, summary)

    print("\nDone.")
    print(f"Output JSON: {output_json}")
    print(f"Streaming JSONL: {progress_path}")
    print(f"Output figure: {fig_path}")
    print(f"Question pool: {summary['question_pool']}")
    print(f"Fraction-stage scope: {summary['fraction_stage_scope']}")
    print(f"Initial pool size: {summary['num_initial_questions']}")
    print(f"Baseline-CoT correct: {summary['num_baseline_cot_correct']}")
    print(f"Baseline-CoT incorrect: {summary['num_baseline_cot_incorrect']}")
    print(f"Fraction-stage questions: {summary['num_fraction_stage_questions']}")
    print(f"Baseline-CoT truncated: {summary['num_baseline_cot_truncated']}")
    print(f"Recovery used: {summary['num_recovery_used']}")
    print(f"CoT cleanup applied: {summary['num_cot_cleanup_applied']}")
    print(f"CoT turn-marker trims: {summary['num_cot_turn_marker_trimmed']}")
    print(f"CoT repeated-final trims: {summary['num_cot_repeated_final_answer_trimmed']}")
    print(f"Generated tokens total: {summary['generated_tokens_total']}")
    print(f"Overall throughput (tokens/s): {summary['overall_tokens_per_second']:.1f}")
    print(f"Generation-only throughput (tokens/s): {summary['generation_tokens_per_second']:.1f}")
    if frac_accuracy:
        print("Fraction accuracy:")
        for k, v in frac_accuracy.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
