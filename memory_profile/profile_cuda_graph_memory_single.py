#!/usr/bin/env python3
"""
Single-run CUDA graph memory profiler.

This script executes exactly one experiment per invocation so allocator/context
state is fresh on each run (no subprocess isolation logic needed).

Usage examples:

  # Run with speculation and capture graphs only for max_draft_len (always-on)
  python profile_cuda_graph_memory_single.py \
    --model_path /path/to/model \
    --model_name MyModel \
    --max_batch_size 32 \
    --max_draft_len 4 \
    --max_concurrency None

  # Run with speculation allowing draft_len=0 (doubles graphs)
  python profile_cuda_graph_memory_single.py \
    --model_path /path/to/model \
    --model_name MyModel \
    --max_batch_size 32 \
    --max_draft_len 4 \
    --max_concurrency 16

The output is a single JSON file under memory_profile/results/, named by
parameters, containing:
  - parameters, memory_before/after/diff
  - number of CUDA graphs and breakdown
  - allocator stats
  - full terminal output as log_lines
"""

import argparse
import contextlib
import gc
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# Force single-process mode for memory profiling access to model_engine
os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"

# Add TensorRT-LLM to path (parent directory since we're in memory_profile/)
sys.path.append(str(Path(__file__).parent.parent))

from tensorrt_llm.llmapi import LLM, CudaGraphConfig, NGramDecodingConfig


def get_memory_info() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}
    device = torch.cuda.current_device()
    return {
        "allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved(device) / 1024**2,
        "max_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
    }


def reset_memory_stats() -> None:
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"Debug: Resetting memory stats on CUDA device {device}")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        gc.collect()


def create_llm(
    model_path: str,
    max_concurrency: Optional[int],
    max_batch_size: int,
    max_draft_len: int,
) -> LLM:
    cuda_graph_config = CudaGraphConfig(
        max_batch_size=max_batch_size,
        enable_padding=False,
    )
    spec_config = NGramDecodingConfig(
        max_draft_len=max_draft_len,
        max_concurrency=max_concurrency,
        max_matching_ngram_size=2,
        is_keep_all=True,
        is_use_oldest=True,
        is_public_pool=True,
    )
    llm = LLM(
        model=model_path,
        max_batch_size=max_batch_size,
        max_num_tokens=1024,
        max_seq_len=2048,
        cuda_graph_config=cuda_graph_config,
        speculative_config=spec_config,
        disable_overlap_scheduler=True,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )
    return llm


def run_experiment(
    model_path: str,
    model_name: str,
    max_batch_size: int,
    max_draft_len: int,
    max_concurrency_arg: Optional[str],
    output_file: Optional[str],
    force_separate_pools: bool,
) -> str:
    # Parse max_concurrency from string allowing "None"
    max_concurrency: Optional[int]
    if max_concurrency_arg is None or max_concurrency_arg.strip().lower() == "none":
        max_concurrency = None
        experiment_name = "mc_None"
    else:
        max_concurrency = int(max_concurrency_arg)
        experiment_name = f"mc_{max_concurrency}"

    # Capture logs while we run, then print at the end and store in JSON
    log_buf = io.StringIO()
    with contextlib.redirect_stdout(log_buf), contextlib.redirect_stderr(log_buf):
        # Configure optional separate CUDA graph pools for this run
        if force_separate_pools:
            os.environ["TLLM_FORCE_SEPARATE_CUDA_GRAPH_POOLS"] = "1"
        else:
            os.environ.pop("TLLM_FORCE_SEPARATE_CUDA_GRAPH_POOLS", None)
        print(f"CUDA Memory Profiling (single run) for {model_name}")
        print(f"Model path: {model_path}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            print(f"Current CUDA device: {current_device}")
            print(f"CUDA device name: {torch.cuda.get_device_name(current_device)}")
            print(
                f"CUDA memory: {torch.cuda.get_device_properties(current_device).total_memory / 1024**3:.1f} GB"
            )

        print(f"\n{'='*60}")
        print(f"Running experiment: {experiment_name}")
        print(f"max_concurrency: {max_concurrency}")
        print(f"max_batch_size: {max_batch_size}")
        print(f"max_draft_len: {max_draft_len}")
        print(f"{'='*60}")

        reset_memory_stats()
        time.sleep(1)
        memory_before = get_memory_info()
        print(f"Memory before LLM: {memory_before}")

        try:
            print("Creating LLM instance (includes warmup & CUDA graph capture)...")
            t0 = time.time()
            llm = create_llm(
                model_path=model_path,
                max_concurrency=max_concurrency,
                max_batch_size=max_batch_size,
                max_draft_len=max_draft_len,
            )
            creation_time = time.time() - t0
            print(f"LLM creation took: {creation_time:.2f} seconds")

            memory_after = get_memory_info()
            print(f"Memory after LLM: {memory_after}")

            memory_diff = {
                "allocated_mb": memory_after["allocated_mb"] - memory_before["allocated_mb"],
                "reserved_mb": memory_after["reserved_mb"] - memory_before["reserved_mb"],
                "max_allocated_mb": memory_after["max_allocated_mb"]
                - memory_before["max_allocated_mb"],
            }
            print(f"Memory difference: {memory_diff}")

            # Access model_engine if available to introspect graphs
            if hasattr(llm._executor, "model_engine"):
                model_engine = llm._executor.model_engine
            elif hasattr(llm._executor, "engine") and hasattr(llm._executor.engine, "model_engine"):
                model_engine = llm._executor.engine.model_engine
            else:
                print(f"Warning: Cannot access model_engine from {type(llm._executor)}")
                model_engine = None

            num_cuda_graphs = 0
            cuda_graph_batch_sizes = []
            graph_breakdown: Dict[str, int] = {}
            if model_engine is not None and hasattr(model_engine, "_cuda_graphs"):
                num_cuda_graphs = sum(
                    len(graphs_by_draft) for graphs_by_draft in model_engine._cuda_graphs.values()
                )
                cuda_graph_batch_sizes = list(getattr(model_engine, "_cuda_graph_batch_sizes", []))
                for bs, graphs_by_draft in model_engine._cuda_graphs.items():
                    for draft_len, _runner in graphs_by_draft.items():
                        key = f"draft_len_{draft_len}"
                        graph_breakdown[key] = graph_breakdown.get(key, 0) + 1

                print(f"Number of CUDA graphs captured (total): {num_cuda_graphs}")
                print(f"CUDA graph batch sizes: {cuda_graph_batch_sizes}")
                print(f"CUDA graphs by draft length (counts): {graph_breakdown}")

                # Allocator stats snapshot
                device = torch.cuda.current_device()
                stats = torch.cuda.memory_stats(device)
                reserved = stats.get("reserved_bytes.all.current", 0) / (1024**2)
                allocated = stats.get("allocated_bytes.all.current", 0) / (1024**2)
                fragmentation = reserved - allocated
                print("Memory allocator state:")
                print(f"  - Reserved: {reserved:.1f} MB")
                print(f"  - Allocated: {allocated:.1f} MB")
                print(f"  - Fragmentation: {fragmentation:.1f} MB")
                print(f"  - Efficiency: {allocated/reserved*100:.1f}%" if reserved > 0 else "N/A")

            # Cleanup LLM to release references
            del llm

            # Prepare results payload
            result: Dict[str, Any] = {
                "model_name": model_name,
                "experiment_name": experiment_name,
                "max_concurrency": max_concurrency,
                "max_batch_size": max_batch_size,
                "max_draft_len": max_draft_len,
                "force_separate_pools": force_separate_pools,
                "memory_before": memory_before,
                "memory_after": memory_after,
                "memory_diff": memory_diff,
                "creation_time_seconds": creation_time,
                "num_cuda_graphs": num_cuda_graphs,
                "cuda_graph_batch_sizes": cuda_graph_batch_sizes,
                "graph_breakdown": graph_breakdown,
            }

        except Exception as e:
            print(f"Error during experiment {experiment_name}: {e}")
            import traceback

            traceback.print_exc()
            result = {
                "model_name": model_name,
                "experiment_name": experiment_name,
                "max_concurrency": max_concurrency,
                "max_batch_size": max_batch_size,
                "max_draft_len": max_draft_len,
                "force_separate_pools": force_separate_pools,
                "error": str(e),
            }
        finally:
            reset_memory_stats()
            time.sleep(1)

    # Finalize logs
    log_text = log_buf.getvalue()
    log_lines = log_text.splitlines()
    result["log_lines"] = log_lines

    # Save single JSON (named by parameters)
    os.makedirs("results", exist_ok=True)
    try:
        os.chmod("results", 0o777)
    except PermissionError:
        pass

    if output_file:
        output_path = output_file
    else:
        mc_str = "None" if max_concurrency is None else str(max_concurrency)
        ts = time.strftime("%Y%m%d_%H%M%S")
        fp = "sepPools1" if force_separate_pools else "sepPools0"
        output_path = f"memory_profile/results/{model_name}_bs{max_batch_size}_draft{max_draft_len}_mc{mc_str}_{fp}_{ts}.json"

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    # Also print logs to terminal
    print(log_text, end="")
    print(f"\nResult saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Single-run CUDA graph memory profiler")
    parser.add_argument("--model_path", required=True, help="Path to model directory")
    parser.add_argument("--model_name", default="llama3_70b", help="Model name for output")
    parser.add_argument("--max_batch_size", type=int, default=32, help="Max batch size")
    parser.add_argument("--max_draft_len", type=int, default=4, help="Max draft length")
    parser.add_argument(
        "--max_concurrency",
        default="None",
        help="Use 'None' for speculation always-on (graphs only for max_draft_len), or an integer >0 to allow draft_len=0 and double graphs",
    )
    parser.add_argument("--output_file", default=None, help="Optional explicit output JSON path")
    parser.add_argument("--force_separate_pools", action="store_true", help="Force each CUDA graph to use a separate memory pool for this run")

    args = parser.parse_args()

    run_experiment(
        model_path=args.model_path,
        model_name=args.model_name,
        max_batch_size=args.max_batch_size,
        max_draft_len=args.max_draft_len,
        max_concurrency_arg=args.max_concurrency,
        output_file=args.output_file,
        force_separate_pools=args.force_separate_pools,
    )


if __name__ == "__main__":
    main()


