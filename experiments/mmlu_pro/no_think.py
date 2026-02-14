#!/usr/bin/env python3
"""Run /nothink evaluation on TIGER-Lab/MMLU-Pro validation with Qwen3-32B."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

LETTER_POOL = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen/Qwen3-32B on TIGER-Lab/MMLU-Pro using /nothink prompts."
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3-32B", help="HF model ID")
    parser.add_argument(
        "--dataset-id",
        default="TIGER-Lab/MMLU-Pro",
        help="HF dataset ID",
    )
    parser.add_argument("--split", default="validation", help="Dataset split to run")
    parser.add_argument(
        "--output",
        default="results/mmlu_pro/validation/mmlu_pro_validation_nothink_qwen3_32b.jsonl",
        help="Output jsonl path",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--max-new-tokens", type=int, default=16, help="Max generated tokens")
    parser.add_argument("--device-map", default="auto", help='HF device_map, e.g. "auto"')
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
        help="Attention backend",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Disable trust_remote_code",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of questions (default: full split)",
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


def build_prompt(question: str, options: Sequence[str]) -> str:
    letters = LETTER_POOL[: len(options)]
    option_lines = [f"{letters[i]}. {opt}" for i, opt in enumerate(options)]
    lines = [
        "/nothink",
        "Answer the multiple-choice question immediately.",
        "Return only one uppercase option letter and nothing else.",
        "",
        f"Question: {question.strip()}",
        "",
        "Options:",
        *option_lines,
        "",
        "Answer:",
    ]
    return "\n".join(lines)


def build_chat_text(tokenizer: Any, prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You are a precise assistant for multiple-choice questions."},
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return prompt


def extract_prediction(response: str, num_options: int) -> Optional[str]:
    allowed = LETTER_POOL[:num_options]
    cleaned = response.strip().upper()
    if cleaned in allowed:
        return cleaned

    pattern = rf"\b([{re.escape(allowed)}])\b"
    match = re.search(pattern, cleaned)
    if match:
        return match.group(1)

    pattern_start = rf"^\s*([{re.escape(allowed)}])(?:[\).\:\-\s]|$)"
    match = re.search(pattern_start, cleaned)
    if match:
        return match.group(1)

    for ch in cleaned:
        if ch in allowed:
            return ch
    return None


def batched_indices(total: int, batch_size: int) -> List[Tuple[int, int]]:
    return [(i, min(i + batch_size, total)) for i in range(0, total, batch_size)]


def sync_if_cuda(model: Any) -> None:
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        for device in model.hf_device_map.values():
            if isinstance(device, str) and device.startswith("cuda"):
                torch.cuda.synchronize()
                return
            if isinstance(device, int):
                torch.cuda.synchronize()
                return
    elif next(model.parameters()).device.type == "cuda":
        torch.cuda.synchronize()


def load_model_and_tokenizer(args: argparse.Namespace) -> Tuple[Any, Any, torch.dtype]:
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
        fallback_kwargs = dict(model_kwargs)
        fallback_kwargs.pop("dtype", None)
        fallback_kwargs["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(args.model_id, **fallback_kwargs)

    model.eval()
    return model, tokenizer, dtype


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be >= 1")

    start_time = time.perf_counter()
    print("Loading dataset...")
    dataset = load_dataset(args.dataset_id, split=args.split)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    total = len(dataset)
    print(f"Loaded split={args.split} with {total} questions.")

    print("Loading model/tokenizer...")
    model_load_t0 = time.perf_counter()
    model, tokenizer, dtype = load_model_and_tokenizer(args)
    model_load_s = time.perf_counter() - model_load_t0
    print(f"Model loaded in {model_load_s:.2f}s with dtype={dtype} device_map={args.device_map}")

    correct_questions: List[Dict[str, Any]] = []
    incorrect_questions: List[Dict[str, Any]] = []
    num_correct = 0
    num_answer_parsing_fail = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f_out, torch.inference_mode():
        for batch_start, batch_end in batched_indices(total, args.batch_size):
            batch = [dataset[i] for i in range(batch_start, batch_end)]
            prompts = [build_prompt(ex["question"], ex["options"]) for ex in batch]
            chat_texts = [build_chat_text(tokenizer, p) for p in prompts]

            tokenized = tokenizer(
                chat_texts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )

            if not getattr(model, "hf_device_map", None):
                target_device = next(model.parameters()).device
                tokenized = {k: v.to(target_device) for k, v in tokenized.items()}

            sync_if_cuda(model)
            generated = model.generate(
                **tokenized,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            sync_if_cuda(model)

            prompt_len = tokenized["input_ids"].shape[1]
            new_tokens = generated[:, prompt_len:]
            responses = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            for local_i, ex in enumerate(batch):
                response = responses[local_i].strip()
                pred = extract_prediction(response, len(ex["options"]))
                gold = ex["answer"].strip().upper()
                is_correct = pred == gold
                if pred is None:
                    num_answer_parsing_fail += 1
                if is_correct:
                    num_correct += 1
                    correct_questions.append(
                        {
                            "question_id": int(ex["question_id"]),
                            "question": ex["question"],
                            "gold": gold,
                            "prediction": pred,
                        }
                    )
                else:
                    incorrect_questions.append(
                        {
                            "question_id": int(ex["question_id"]),
                            "question": ex["question"],
                            "gold": gold,
                            "prediction": pred,
                            "response": response,
                        }
                    )

                row = {
                    "type": "example",
                    "question_id": int(ex["question_id"]),
                    "question": ex["question"],
                    "options": ex["options"],
                    "gold_answer": gold,
                    "gold_answer_index": int(ex["answer_index"]),
                    "model_response": response,
                    "parsed_prediction": pred,
                    "is_correct": is_correct,
                    "category": ex.get("category"),
                    "src": ex.get("src"),
                }
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")

            processed = batch_end
            elapsed = time.perf_counter() - start_time
            print(f"Processed {processed}/{total} | accuracy_so_far={num_correct / processed:.4f} | elapsed={elapsed:.1f}s")

        summary = {
            "type": "summary",
            "dataset_id": args.dataset_id,
            "split": args.split,
            "model_id": args.model_id,
            "dtype": str(dtype),
            "device_map": args.device_map,
            "num_questions": total,
            "num_correct": num_correct,
            "num_incorrect": total - num_correct,
            "accuracy": (num_correct / total) if total else 0.0,
            "num_answer_parsing_fail": num_answer_parsing_fail,
            "correct_questions": correct_questions,
            "incorrect_questions": incorrect_questions,
        }
        f_out.write(json.dumps(summary, ensure_ascii=False) + "\n")

    total_time_s = time.perf_counter() - start_time
    print("\nDone.")
    print(f"Output: {output_path}")
    print(f"Questions: {total}")
    print(f"Correct: {num_correct}")
    print(f"Incorrect: {total - num_correct}")
    print(f"Accuracy: {num_correct / total:.4f}" if total else "Accuracy: 0.0000")
    print(f"Parse failures: {num_answer_parsing_fail}")
    print(f"Total runtime: {total_time_s:.2f}s")


if __name__ == "__main__":
    main()
