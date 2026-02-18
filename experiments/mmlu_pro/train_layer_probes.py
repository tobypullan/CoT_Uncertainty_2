#!/usr/bin/env python3
"""Train layer-wise probes for per-fraction correctness on CoT-fraction runs."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = "You are a precise assistant for multiple-choice questions."
LETTER_POOL = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train layer-wise probes from CoT fraction outputs.",
    )
    parser.add_argument(
        "--cot-json",
        default="results/mmlu_pro/validation/cot_questioning_qwen3_32b.json",
        help="Path to CoT fraction JSON output.",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3-32B", help="HF model ID.")
    parser.add_argument(
        "--splitter",
        default="group_kfold",
        choices=["group_kfold"],
        help="Cross-validation splitter.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of GroupKFold splits.",
    )
    parser.add_argument(
        "--balance",
        default="class_weight",
        choices=["class_weight", "none"],
        help="Class balancing strategy for logistic regression probes.",
    )
    parser.add_argument(
        "--hook-point",
        default="post_mlp_residual",
        choices=["post_mlp_residual"],
        help="Activation hook point to probe.",
    )
    parser.add_argument(
        "--output-json",
        default="results/mmlu_pro/validation/layer_probes_qwen3_32b.json",
        help="Output JSON path for probe metrics.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for activation extraction forward passes.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="Model dtype.",
    )
    parser.add_argument("--device-map", default="auto", help='HF device_map (e.g. "auto").')
    parser.add_argument(
        "--attn-impl",
        choices=["auto", "sdpa", "flash_attention_2", "eager"],
        default="auto",
        help="Attention backend.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic probe training.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of fraction trials after filtering.",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Disable trust_remote_code.",
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


def format_options(options: Sequence[str]) -> str:
    letters = LETTER_POOL[: len(options)]
    return "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(options))


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


def build_chat_text(tokenizer: Any, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        im_start = "<|im_start|>"
        im_end = "<|im_end|>"
        return (
            f"{im_start}system\n{SYSTEM_PROMPT}{im_end}\n"
            f"{im_start}user\n{user_prompt}{im_end}\n"
            f"{im_start}assistant\n"
        )


def cot_fraction_snippet(tokenizer: Any, cot_text: str, numer: int, denom: int) -> Tuple[str, int, int]:
    tokens = tokenizer.encode(cot_text, add_special_tokens=False)
    total = len(tokens)
    if total == 0:
        return cot_text, 0, 0
    keep = max(1, math.ceil(total * numer / denom))
    keep = min(total, keep)
    snippet = tokenizer.decode(tokens[:keep], skip_special_tokens=True)
    return snippet, keep, total


def parse_fraction_index(trial: Dict[str, Any]) -> Optional[int]:
    label = trial.get("fraction_label")
    if isinstance(label, str):
        match = re.fullmatch(r"\s*(\d+)\s*/\s*10\s*", label)
        if match:
            value = int(match.group(1))
            if 1 <= value <= 10:
                return value
    fraction = trial.get("fraction")
    if isinstance(fraction, (float, int)):
        value = int(round(float(fraction) * 10.0))
        if 1 <= value <= 10:
            return value
    return None


def load_probe_rows(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload.get("questions", [])
    summary = payload.get("summary", {})

    rows: List[Dict[str, Any]] = []
    parse_failures_excluded = 0
    invalid_fraction_entries = 0
    label_type_errors = 0
    non_ten_trial_questions: List[int] = []

    for question_row in questions:
        qid = int(question_row["question_id"])
        trials = list(question_row.get("fraction_trials", []))
        if len(trials) != 10:
            non_ten_trial_questions.append(qid)
        cot_text = question_row.get("baseline_cot_response", "") or ""
        q_text = question_row.get("question", "")
        options = question_row.get("options", [])

        for trial in trials:
            if trial.get("parsed_prediction") is None:
                parse_failures_excluded += 1
                continue

            fraction_index = parse_fraction_index(trial)
            if fraction_index is None:
                invalid_fraction_entries += 1
                continue

            raw_label = trial.get("is_correct")
            if not isinstance(raw_label, bool):
                label_type_errors += 1
            y = int(bool(raw_label))

            rows.append(
                {
                    "question_id": qid,
                    "question": q_text,
                    "options": options,
                    "cot_text": cot_text,
                    "fraction_index": fraction_index,
                    "fraction_label": f"{fraction_index}/10",
                    "y": y,
                }
            )

    diagnostics = {
        "summary_question_pool": summary.get("question_pool"),
        "summary_fraction_stage_scope": summary.get("fraction_stage_scope"),
        "summary_num_processed_questions": summary.get("num_processed_questions"),
        "questions_in_json": len(questions),
        "rows_after_filtering": len(rows),
        "parse_failures_excluded": parse_failures_excluded,
        "invalid_fraction_entries_excluded": invalid_fraction_entries,
        "label_type_errors": label_type_errors,
        "questions_with_non_10_fraction_trials": sorted(non_ten_trial_questions),
    }
    return rows, diagnostics


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


def find_transformer_layers(model: Any) -> Sequence[Any]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    raise RuntimeError("Unable to locate transformer block list on the loaded model.")


def extract_post_mlp_residual_features(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    batch_size: int,
) -> Tuple[List[np.ndarray], np.ndarray, int, int]:
    layers = find_transformer_layers(model)
    forward_module = model.model if hasattr(model, "model") else model
    num_layers = len(layers)
    if num_layers < 1:
        raise RuntimeError("Model has no transformer layers.")

    hidden_size = int(getattr(model.config, "hidden_size"))
    feature_chunks: List[List[np.ndarray]] = [[] for _ in range(num_layers)]
    prompt_lengths: List[int] = []
    hook_outputs: Dict[int, torch.Tensor] = {}
    handles: List[Any] = []

    def make_hook(layer_idx: int):
        def _hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor):
                raise RuntimeError(f"Layer {layer_idx} produced non-tensor output.")
            hook_outputs[layer_idx] = hidden

        return _hook

    for idx, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(idx)))

    try:
        total = len(prompts)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_prompts = prompts[start:end]
            chat_texts = [build_chat_text(tokenizer, prompt) for prompt in batch_prompts]
            inputs = tokenizer(chat_texts, return_tensors="pt", padding=True, truncation=False)
            attention_mask = inputs["attention_mask"]
            lengths = attention_mask.sum(dim=1)
            prompt_lengths.extend(int(x) for x in lengths.tolist())

            if not getattr(model, "hf_device_map", None):
                target_device = next(model.parameters()).device
                inputs = {k: v.to(target_device) for k, v in inputs.items()}

            hook_outputs.clear()
            with torch.inference_mode():
                _ = forward_module(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                )

            if len(hook_outputs) != num_layers:
                missing = [idx for idx in range(num_layers) if idx not in hook_outputs]
                raise RuntimeError(f"Missing layer hook outputs for layers: {missing[:8]}")

            for layer_idx in range(num_layers):
                hidden = hook_outputs[layer_idx]
                last_idx = (lengths - 1).to(hidden.device)
                batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
                vectors = hidden[batch_idx, last_idx, :].detach().to(torch.float32).cpu().numpy()
                feature_chunks[layer_idx].append(vectors.astype(np.float16, copy=False))

            print(f"Extracted activations for {end}/{total} prompts")
    finally:
        for handle in handles:
            handle.remove()

    features_by_layer: List[np.ndarray] = []
    for layer_idx, chunks in enumerate(feature_chunks):
        if not chunks:
            raise RuntimeError(f"No activations captured for layer {layer_idx}.")
        layer_features = np.concatenate(chunks, axis=0)
        if layer_features.shape[1] != hidden_size:
            raise RuntimeError(
                f"Layer {layer_idx} hidden-size mismatch: expected {hidden_size}, got {layer_features.shape[1]}."
            )
        features_by_layer.append(layer_features)

    return features_by_layer, np.asarray(prompt_lengths, dtype=np.int32), num_layers, hidden_size


def train_and_predict_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    class_weight: Optional[str],
    seed: int,
) -> np.ndarray:
    if np.unique(y_train).size < 2:
        fill_value = float(np.mean(y_train))
        return np.full(x_val.shape[0], fill_value, dtype=np.float64)

    x_train = x_train.astype(np.float32, copy=False)
    x_val = x_val.astype(np.float32, copy=False)

    train_mean = np.mean(x_train, axis=0, keepdims=True)
    train_std = np.std(x_train, axis=0, keepdims=True)
    train_std = np.where(train_std < 1e-6, 1.0, train_std)
    x_train_scaled = (x_train - train_mean) / train_std
    x_val_scaled = (x_val - train_mean) / train_std

    y_train_float = y_train.astype(np.float32)
    if class_weight == "balanced":
        pos_count = float(np.sum(y_train_float == 1.0))
        neg_count = float(np.sum(y_train_float == 0.0))
        if pos_count > 0.0 and neg_count > 0.0:
            n_samples = float(y_train_float.shape[0])
            pos_weight = n_samples / (2.0 * pos_count)
            neg_weight = n_samples / (2.0 * neg_count)
            sample_weights = np.where(y_train_float > 0.5, pos_weight, neg_weight).astype(np.float32)
        else:
            sample_weights = np.ones_like(y_train_float, dtype=np.float32)
    else:
        sample_weights = np.ones_like(y_train_float, dtype=np.float32)

    torch.manual_seed(seed)
    x_train_t = torch.from_numpy(x_train_scaled)
    y_train_t = torch.from_numpy(y_train_float)
    w_train_t = torch.from_numpy(sample_weights)
    x_val_t = torch.from_numpy(x_val_scaled)

    weight = torch.zeros(x_train_t.shape[1], dtype=torch.float32, requires_grad=True)
    bias = torch.zeros(1, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=0.05)

    best_loss = float("inf")
    stale_epochs = 0
    for _ in range(120):
        optimizer.zero_grad()
        logits = x_train_t.matmul(weight) + bias
        per_example_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y_train_t, reduction="none"
        )
        loss = (per_example_loss * w_train_t).mean() + 1e-4 * weight.pow(2).mean()
        loss.backward()
        optimizer.step()

        current_loss = float(loss.detach().cpu().item())
        if best_loss - current_loss > 1e-6:
            best_loss = current_loss
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= 10:
            break

    with torch.inference_mode():
        probs = torch.sigmoid(x_val_t.matmul(weight) + bias).cpu().numpy()
    return probs.astype(np.float64, copy=False)


def compute_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    positive_scores = y_prob[y_true == 1]
    negative_scores = y_prob[y_true == 0]
    if positive_scores.size == 0 or negative_scores.size == 0:
        return None
    diffs = positive_scores[:, None] - negative_scores[None, :]
    wins = np.sum(diffs > 0)
    ties = np.sum(diffs == 0)
    auc = (wins + 0.5 * ties) / float(positive_scores.size * negative_scores.size)
    return float(auc)


def compute_average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    num_pos = int(np.sum(y_true == 1))
    num_neg = int(np.sum(y_true == 0))
    if num_pos == 0 or num_neg == 0:
        return None
    order = np.argsort(-y_prob, kind="mergesort")
    y_sorted = y_true[order]
    tp_cumsum = np.cumsum(y_sorted == 1)
    precision_at_k = tp_cumsum / (np.arange(y_sorted.shape[0]) + 1)
    ap = np.sum(precision_at_k[y_sorted == 1]) / float(num_pos)
    return float(ap)


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Optional[float]]:
    y_true = y_true.astype(int)
    y_prob = y_prob.astype(float)
    y_pred = (y_prob >= 0.5).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    tpr = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / float(tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = 0.5 * (tpr + tnr)
    f1_denom = (2 * tp + fp + fn)
    f1 = (2.0 * tp / float(f1_denom)) if f1_denom > 0 else 0.0
    brier = float(np.mean((y_prob - y_true) ** 2))

    return {
        "auroc": compute_auroc(y_true, y_prob),
        "auprc": compute_average_precision(y_true, y_prob),
        "balanced_acc": float(balanced_acc),
        "f1": float(f1),
        "brier": brier,
    }


def mean_or_none(values: Sequence[Optional[float]]) -> Optional[float]:
    cleaned = [v for v in values if v is not None]
    if not cleaned:
        return None
    return float(np.mean(cleaned))


def std_or_none(values: Sequence[Optional[float]]) -> Optional[float]:
    cleaned = [v for v in values if v is not None]
    if not cleaned:
        return None
    return float(np.std(cleaned))


def iter_group_kfold_splits(groups: np.ndarray, n_splits: int) -> Sequence[Tuple[int, np.ndarray, np.ndarray]]:
    group_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, group_id in enumerate(groups.tolist()):
        group_to_indices[int(group_id)].append(idx)

    unique_groups = sorted(group_to_indices.keys())
    if n_splits > len(unique_groups):
        raise ValueError(f"n_splits={n_splits} exceeds number of unique groups={len(unique_groups)}")

    sorted_groups = sorted(unique_groups, key=lambda g: (-len(group_to_indices[g]), g))
    fold_groups: List[List[int]] = [[] for _ in range(n_splits)]
    fold_sizes = [0 for _ in range(n_splits)]

    for group_id in sorted_groups:
        fold_idx = int(np.argmin(fold_sizes))
        fold_groups[fold_idx].append(group_id)
        fold_sizes[fold_idx] += len(group_to_indices[group_id])

    splits: List[Tuple[int, np.ndarray, np.ndarray]] = []
    all_group_ids = np.asarray(groups.tolist(), dtype=np.int64)
    for fold_idx, val_group_ids in enumerate(fold_groups, start=1):
        val_set = set(val_group_ids)
        val_mask = np.asarray([int(g) in val_set for g in all_group_ids], dtype=bool)
        val_idx = np.nonzero(val_mask)[0]
        train_idx = np.nonzero(~val_mask)[0]
        if val_idx.size == 0 or train_idx.size == 0:
            raise RuntimeError(f"Invalid fold split at fold={fold_idx} (empty train or val).")
        splits.append((fold_idx, train_idx.astype(np.int64), val_idx.astype(np.int64)))
    return splits


def run_group_kfold(
    features_by_layer: Sequence[np.ndarray],
    metadata_features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    fraction_indices: np.ndarray,
    n_splits: int,
    class_weight: Optional[str],
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[int, List[Dict[str, Any]]], bool, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    predictions_by_layer: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    leakage_failures: List[Dict[str, Any]] = []

    splits = iter_group_kfold_splits(groups=groups, n_splits=n_splits)
    for fold_idx, train_idx, val_idx in splits:
        train_groups = set(groups[train_idx].tolist())
        val_groups = set(groups[val_idx].tolist())
        overlap = train_groups.intersection(val_groups)
        if overlap:
            leakage_failures.append(
                {
                    "fold": fold_idx,
                    "overlap_question_ids": sorted(int(x) for x in overlap),
                }
            )

        y_train = labels[train_idx]
        y_val = labels[val_idx]

        metadata_probs = train_and_predict_logistic(
            x_train=metadata_features[train_idx],
            y_train=y_train,
            x_val=metadata_features[val_idx],
            class_weight=class_weight,
            seed=seed,
        )
        metadata_metrics = compute_binary_metrics(y_val, metadata_probs)

        for layer_idx, layer_features in enumerate(features_by_layer):
            layer_probs = train_and_predict_logistic(
                x_train=layer_features[train_idx].astype(np.float32, copy=False),
                y_train=y_train,
                x_val=layer_features[val_idx].astype(np.float32, copy=False),
                class_weight=class_weight,
                seed=seed,
            )
            metrics = compute_binary_metrics(y_val, layer_probs)

            row = {
                "layer": layer_idx,
                "fold": fold_idx,
                "n_train": int(train_idx.shape[0]),
                "n_val": int(val_idx.shape[0]),
                "pos_rate_train": float(np.mean(y_train)),
                "pos_rate_val": float(np.mean(y_val)),
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "balanced_acc": metrics["balanced_acc"],
                "f1": metrics["f1"],
                "brier": metrics["brier"],
                "metadata_baseline_auroc": metadata_metrics["auroc"],
            }
            rows.append(row)

            predictions_by_layer[layer_idx].append(
                {
                    "fold": fold_idx,
                    "y_true": y_val.astype(int).tolist(),
                    "y_prob": layer_probs.astype(float).tolist(),
                    "fraction_indices": fraction_indices[val_idx].astype(int).tolist(),
                }
            )

        print(
            f"Completed fold {fold_idx}/{n_splits} "
            f"(n_train={train_idx.shape[0]}, n_val={val_idx.shape[0]}, metadata_auroc={metadata_metrics['auroc']})"
        )

    leakage_pass = len(leakage_failures) == 0
    return rows, predictions_by_layer, leakage_pass, leakage_failures


def summarize_layers(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_layer: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_layer[int(row["layer"])].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for layer_idx in sorted(by_layer):
        layer_rows = by_layer[layer_idx]
        mean_auroc = mean_or_none([r["auroc"] for r in layer_rows])
        mean_metadata = mean_or_none([r["metadata_baseline_auroc"] for r in layer_rows])
        summary_rows.append(
            {
                "layer": layer_idx,
                "mean_auroc": mean_auroc,
                "std_auroc": std_or_none([r["auroc"] for r in layer_rows]),
                "mean_auprc": mean_or_none([r["auprc"] for r in layer_rows]),
                "mean_balanced_acc": mean_or_none([r["balanced_acc"] for r in layer_rows]),
                "mean_f1": mean_or_none([r["f1"] for r in layer_rows]),
                "mean_brier": mean_or_none([r["brier"] for r in layer_rows]),
                "mean_metadata_baseline_auroc": mean_metadata,
                "auroc_gap_vs_metadata": (
                    (mean_auroc - mean_metadata)
                    if (mean_auroc is not None and mean_metadata is not None)
                    else None
                ),
            }
        )
    return summary_rows


def choose_best_layer(layer_summary: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    candidates = [row for row in layer_summary if row.get("mean_auroc") is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row["mean_auroc"]))


def robustness_by_fraction(predictions: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    bucket: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: {"y": [], "p": []})
    for fold_block in predictions:
        for y_true, y_prob, frac_idx in zip(
            fold_block["y_true"], fold_block["y_prob"], fold_block["fraction_indices"]
        ):
            frac = int(frac_idx)
            bucket[frac]["y"].append(int(y_true))
            bucket[frac]["p"].append(float(y_prob))

    rows: List[Dict[str, Any]] = []
    for frac_idx in range(1, 11):
        ys = np.asarray(bucket[frac_idx]["y"], dtype=np.int64)
        ps = np.asarray(bucket[frac_idx]["p"], dtype=np.float64)
        if ys.size == 0:
            rows.append(
                {
                    "fraction_label": f"{frac_idx}/10",
                    "n": 0,
                    "pos_rate": None,
                    "auroc": None,
                    "balanced_acc": None,
                    "f1": None,
                    "brier": None,
                }
            )
            continue
        metrics = compute_binary_metrics(ys, ps)
        rows.append(
            {
                "fraction_label": f"{frac_idx}/10",
                "n": int(ys.size),
                "pos_rate": float(np.mean(ys)),
                "auroc": metrics["auroc"],
                "balanced_acc": metrics["balanced_acc"],
                "f1": metrics["f1"],
                "brier": metrics["brier"],
            }
        )
    return rows


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.n_splits < 2:
        raise ValueError("--n-splits must be >= 2")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    cot_json_path = Path(args.cot_json)
    if not cot_json_path.exists():
        raise FileNotFoundError(f"Missing CoT JSON: {cot_json_path}")

    print(f"Loading CoT data from {cot_json_path}")
    probe_rows, diagnostics = load_probe_rows(cot_json_path)
    if args.max_samples is not None:
        probe_rows = probe_rows[: args.max_samples]
    if not probe_rows:
        raise RuntimeError("No valid fraction trials available after filtering.")

    labels = np.asarray([row["y"] for row in probe_rows], dtype=np.int64)
    groups = np.asarray([row["question_id"] for row in probe_rows], dtype=np.int64)
    fraction_indices = np.asarray([row["fraction_index"] for row in probe_rows], dtype=np.int64)

    unique_groups = int(np.unique(groups).shape[0])
    if args.n_splits > unique_groups:
        raise ValueError(
            f"--n-splits={args.n_splits} exceeds number of unique question groups ({unique_groups})."
        )

    print("Loading model/tokenizer for activation extraction...")
    model, tokenizer, dtype = load_model_and_tokenizer(args)
    print(f"Model ready with dtype={dtype} device_map={args.device_map}")

    prompts: List[str] = []
    for row in probe_rows:
        snippet, _, _ = cot_fraction_snippet(
            tokenizer=tokenizer,
            cot_text=row["cot_text"],
            numer=row["fraction_index"],
            denom=10,
        )
        prompts.append(
            build_fraction_prompt(
                question=row["question"],
                options=row["options"],
                cot_snippet=snippet,
            )
        )

    print(
        f"Extracting {args.hook_point} activations for {len(prompts)} prompts "
        f"across {unique_groups} question groups..."
    )
    features_by_layer, prompt_lengths, num_layers, hidden_size = extract_post_mlp_residual_features(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        batch_size=args.batch_size,
    )

    hook_consistency_pass = True
    hook_consistency_errors: List[str] = []
    for layer_idx, feats in enumerate(features_by_layer):
        if feats.shape[0] != len(probe_rows):
            hook_consistency_pass = False
            hook_consistency_errors.append(
                f"layer={layer_idx}: expected {len(probe_rows)} rows, got {feats.shape[0]}"
            )
        if feats.shape[1] != hidden_size:
            hook_consistency_pass = False
            hook_consistency_errors.append(
                f"layer={layer_idx}: expected hidden_size={hidden_size}, got {feats.shape[1]}"
            )

    metadata_features = np.stack(
        [
            fraction_indices.astype(np.float32),
            prompt_lengths.astype(np.float32),
        ],
        axis=1,
    )
    class_weight = "balanced" if args.balance == "class_weight" else None
    rows, predictions_by_layer, leakage_pass, leakage_failures = run_group_kfold(
        features_by_layer=features_by_layer,
        metadata_features=metadata_features,
        labels=labels,
        groups=groups,
        fraction_indices=fraction_indices,
        n_splits=args.n_splits,
        class_weight=class_weight,
        seed=args.seed,
    )

    layer_summary = summarize_layers(rows)
    best_layer = choose_best_layer(layer_summary)
    per_layer_curve = [
        {"layer": row["layer"], "mean_auroc": row["mean_auroc"]}
        for row in layer_summary
    ]

    metadata_aurocs = [row["metadata_baseline_auroc"] for row in rows if row["metadata_baseline_auroc"] is not None]
    metadata_mean_auroc = mean_or_none(metadata_aurocs)
    best_layer_mean_auroc = best_layer["mean_auroc"] if best_layer is not None else None
    baseline_sanity_pass = (
        best_layer_mean_auroc is not None
        and metadata_mean_auroc is not None
        and float(best_layer_mean_auroc) > float(metadata_mean_auroc)
    )

    best_layer_robustness: List[Dict[str, Any]] = []
    if best_layer is not None:
        best_layer_idx = int(best_layer["layer"])
        best_layer_robustness = robustness_by_fraction(predictions_by_layer[best_layer_idx])

    data_coverage_pass = len(diagnostics["questions_with_non_10_fraction_trials"]) == 0
    label_integrity_pass = diagnostics["label_type_errors"] == 0
    robustness_report_pass = len(best_layer_robustness) == 10 and all(
        row["n"] > 0 for row in best_layer_robustness
    )

    tests = {
        "data_coverage": {
            "pass": data_coverage_pass,
            "questions_with_non_10_fraction_trials": diagnostics["questions_with_non_10_fraction_trials"],
            "questions_in_json": diagnostics["questions_in_json"],
            "summary_num_processed_questions": diagnostics["summary_num_processed_questions"],
        },
        "label_integrity": {
            "pass": label_integrity_pass,
            "label_type_errors": diagnostics["label_type_errors"],
            "parse_failures_excluded": diagnostics["parse_failures_excluded"],
            "invalid_fraction_entries_excluded": diagnostics["invalid_fraction_entries_excluded"],
        },
        "leakage": {
            "pass": leakage_pass,
            "failures": leakage_failures,
        },
        "hook_consistency": {
            "pass": hook_consistency_pass,
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "errors": hook_consistency_errors,
        },
        "baseline_sanity": {
            "pass": baseline_sanity_pass,
            "best_layer_mean_auroc": best_layer_mean_auroc,
            "metadata_mean_auroc": metadata_mean_auroc,
        },
        "robustness_by_fraction": {
            "pass": robustness_report_pass,
            "rows_reported": len(best_layer_robustness),
        },
    }

    output_payload = {
        "summary": {
            "cot_json": str(cot_json_path),
            "model_id": args.model_id,
            "dtype": str(dtype),
            "device_map": args.device_map,
            "hook_point": args.hook_point,
            "splitter": args.splitter,
            "n_splits": args.n_splits,
            "balance": args.balance,
            "num_samples": len(probe_rows),
            "num_question_groups": unique_groups,
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "metadata_mean_auroc": metadata_mean_auroc,
            "best_layer": best_layer,
            "source_question_pool": diagnostics["summary_question_pool"],
            "source_fraction_stage_scope": diagnostics["summary_fraction_stage_scope"],
        },
        "tests": tests,
        "fold_results": rows,
        "layer_summary": layer_summary,
        "per_layer_curve": per_layer_curve,
        "best_layer_robustness_by_fraction": best_layer_robustness,
    }

    output_path = Path(args.output_json)
    write_json(output_path, output_payload)
    print(f"Saved probe results to {output_path}")


if __name__ == "__main__":
    main()
