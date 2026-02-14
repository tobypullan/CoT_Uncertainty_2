#!/usr/bin/env python3
"""Benchmark local forward-pass speed for Qwen/Qwen3-32B.

Example:
  python local_model.py \
    --model-id Qwen/Qwen3-32B \
    --batch-size 1 \
    --seq-len 1024 \
    --runs 8 \
    --decode-steps 64
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Any, Dict, Iterable, List, Tuple

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local inference speed tests for a causal LM (default: Qwen/Qwen3-32B)."
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3-32B", help="HF model ID")
    parser.add_argument(
        "--prompt",
        default="Explain in one paragraph why batching improves GPU throughput.",
        help="Prompt used to build a fixed-length token batch.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for both benchmarks")
    parser.add_argument(
        "--seq-len",
        type=int,
        default=1024,
        help="Context length for prefill benchmark (token count per sequence)",
    )
    parser.add_argument("--warmup-runs", type=int, default=2, help="Warmup forward passes")
    parser.add_argument("--runs", type=int, default=8, help="Measured prefill forward passes")
    parser.add_argument(
        "--decode-steps",
        type=int,
        default=64,
        help="Token-by-token decode steps after one prefill pass",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="Model dtype",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help='HF device_map (e.g. "auto", "cpu", "cuda:0")',
    )
    parser.add_argument(
        "--attn-impl",
        choices=["auto", "sdpa", "flash_attention_2", "eager"],
        default="auto",
        help="Attention kernel implementation",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Use bitsandbytes 4-bit quantization (recommended if VRAM is limited)",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Compile the model forward (PyTorch 2.x)",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Disable trust_remote_code when loading tokenizer/model",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32

    if torch.cuda.is_available():
        bf16_supported = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        return torch.bfloat16 if bf16_supported else torch.float16
    return torch.float32


def sync_if_needed(enabled: bool) -> None:
    if not enabled:
        return
    torch.cuda.synchronize()


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


def percentile(values: Iterable[float], p: float) -> float:
    vals = sorted(values)
    if not vals:
        return float("nan")
    idx = (len(vals) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(vals) - 1)
    if lo == hi:
        return vals[lo]
    frac = idx - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def build_fixed_length_batch(
    tokenizer: Any,
    prompt: str,
    batch_size: int,
    seq_len: int,
) -> Dict[str, torch.Tensor]:
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not token_ids:
        if tokenizer.eos_token_id is None:
            raise ValueError("Prompt tokenized to empty and tokenizer has no eos_token_id.")
        token_ids = [tokenizer.eos_token_id]

    repeats = (seq_len + len(token_ids) - 1) // len(token_ids)
    fixed = (token_ids * repeats)[:seq_len]

    input_ids = torch.tensor([fixed] * batch_size, dtype=torch.long)
    return {"input_ids": input_ids}


def move_inputs_if_single_device(
    inputs: Dict[str, torch.Tensor], model: Any
) -> Dict[str, torch.Tensor]:
    # With HF dispatch/device_map=auto, keep inputs on CPU and let hooks move them.
    if getattr(model, "hf_device_map", None) is not None:
        return inputs

    model_device = next(model.parameters()).device
    return {k: v.to(model_device) for k, v in inputs.items()}


def run_prefill_benchmark(
    model: Any,
    batch: Dict[str, torch.Tensor],
    warmup_runs: int,
    runs: int,
    use_cuda_sync: bool,
) -> List[float]:
    latencies = []

    with torch.inference_mode():
        for _ in range(warmup_runs):
            _ = model(**batch, use_cache=True)

        for _ in range(runs):
            sync_if_needed(use_cuda_sync)
            t0 = time.perf_counter()
            out = model(**batch, use_cache=True)
            _ = out.logits[:, -1, :].argmax(dim=-1)
            sync_if_needed(use_cuda_sync)
            latencies.append(time.perf_counter() - t0)

    return latencies


def run_decode_benchmark(
    model: Any,
    batch: Dict[str, torch.Tensor],
    decode_steps: int,
    use_cuda_sync: bool,
) -> Tuple[float, List[float]]:
    if decode_steps <= 0:
        return 0.0, []

    with torch.inference_mode():
        sync_if_needed(use_cuda_sync)
        t0 = time.perf_counter()
        prefill = model(**batch, use_cache=True)
        _ = prefill.logits[:, -1, :].argmax(dim=-1)
        sync_if_needed(use_cuda_sync)
        prefill_latency = time.perf_counter() - t0

        past_key_values = prefill.past_key_values
        next_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        step_latencies = []
        for _ in range(decode_steps):
            sync_if_needed(use_cuda_sync)
            t1 = time.perf_counter()
            out = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
            _ = out.logits[:, -1, :].argmax(dim=-1)
            sync_if_needed(use_cuda_sync)
            step_latencies.append(time.perf_counter() - t1)

            past_key_values = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    return prefill_latency, step_latencies


def maybe_print_cuda_memory(enabled: bool) -> None:
    if not enabled:
        return

    print("\\nCUDA peak memory by device:")
    for i in range(torch.cuda.device_count()):
        try:
            alloc_gb = torch.cuda.max_memory_allocated(i) / (1024**3)
            reserved_gb = torch.cuda.max_memory_reserved(i) / (1024**3)
            print(f"  cuda:{i}: allocated={alloc_gb:.2f} GB, reserved={reserved_gb:.2f} GB")
        except RuntimeError as exc:
            print(f"  cuda:{i}: unable to read memory stats ({exc})")


def main() -> None:
    args = parse_args()
    trust_remote_code = not args.no_trust_remote_code

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency: transformers. Install with `pip install transformers` "
            "and rerun."
        ) from exc

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.seq_len < 1:
        raise ValueError("--seq-len must be >= 1")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    dtype = resolve_dtype(args.dtype)

    print("Loading tokenizer...")
    tokenizer_t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer_load_s = time.perf_counter() - tokenizer_t0

    model_kwargs = {
        "dtype": dtype,
        "device_map": args.device_map,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if args.attn_impl != "auto":
        model_kwargs["attn_implementation"] = args.attn_impl

    if args.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "--load-in-4bit requires bitsandbytes and a compatible transformers install."
            ) from exc

        compute_dtype = dtype if dtype in (torch.float16, torch.bfloat16) else torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    print("Loading model...")
    model_t0 = time.perf_counter()
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    except TypeError as exc:
        if "unexpected keyword argument 'dtype'" not in str(exc):
            raise
        fallback_kwargs = dict(model_kwargs)
        fallback_kwargs.pop("dtype", None)
        fallback_kwargs["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(args.model_id, **fallback_kwargs)
    model_load_s = time.perf_counter() - model_t0
    model.eval()
    use_cuda_sync = model_uses_cuda(model)

    if args.torch_compile:
        if hasattr(torch, "compile") and getattr(model, "hf_device_map", None) is None:
            print("Compiling model forward (torch.compile)...")
            model = torch.compile(model, mode="reduce-overhead")
        else:
            print("Skipping torch.compile (unsupported with current setup/device_map).")

    batch = build_fixed_length_batch(
        tokenizer=tokenizer,
        prompt=args.prompt,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )
    batch = move_inputs_if_single_device(batch, model)

    if use_cuda_sync:
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.reset_peak_memory_stats(i)
            except RuntimeError:
                pass

    print("\\nBenchmark config:")
    print(f"  model_id: {args.model_id}")
    print(f"  dtype: {dtype}")
    print(f"  device_map: {args.device_map}")
    print(f"  attn_impl: {args.attn_impl}")
    print(f"  load_in_4bit: {args.load_in_4bit}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  seq_len: {args.seq_len}")
    print(f"  warmup_runs: {args.warmup_runs}")
    print(f"  runs: {args.runs}")
    print(f"  decode_steps: {args.decode_steps}")
    print(f"  tokenizer_load_s: {tokenizer_load_s:.2f}")
    print(f"  model_load_s: {model_load_s:.2f}")

    print("\\nRunning prefill benchmark...")
    prefill_latencies = run_prefill_benchmark(
        model=model,
        batch=batch,
        warmup_runs=args.warmup_runs,
        runs=args.runs,
        use_cuda_sync=use_cuda_sync,
    )

    prefill_mean = statistics.mean(prefill_latencies)
    prefill_p50 = percentile(prefill_latencies, 0.50)
    prefill_p95 = percentile(prefill_latencies, 0.95)
    prefill_tokens = args.batch_size * args.seq_len
    prefill_tokens_per_sec = prefill_tokens / prefill_mean

    print("Prefill results:")
    print(f"  mean_latency_s: {prefill_mean:.4f}")
    print(f"  p50_latency_s: {prefill_p50:.4f}")
    print(f"  p95_latency_s: {prefill_p95:.4f}")
    print(f"  throughput_tokens_per_s: {prefill_tokens_per_sec:,.1f}")

    print("\\nRunning decode benchmark...")
    decode_prefill_s, decode_step_latencies = run_decode_benchmark(
        model=model,
        batch=batch,
        decode_steps=args.decode_steps,
        use_cuda_sync=use_cuda_sync,
    )

    if decode_step_latencies:
        decode_total_s = sum(decode_step_latencies)
        decode_tps = (args.batch_size * len(decode_step_latencies)) / decode_total_s
        print("Decode results:")
        print(f"  prefill_for_decode_s: {decode_prefill_s:.4f}")
        print(f"  mean_step_latency_s: {statistics.mean(decode_step_latencies):.4f}")
        print(f"  p50_step_latency_s: {percentile(decode_step_latencies, 0.50):.4f}")
        print(f"  p95_step_latency_s: {percentile(decode_step_latencies, 0.95):.4f}")
        print(f"  throughput_tokens_per_s: {decode_tps:,.1f}")
    else:
        print("Decode results: skipped (decode_steps <= 0)")

    maybe_print_cuda_memory(use_cuda_sync)


if __name__ == "__main__":
    main()
