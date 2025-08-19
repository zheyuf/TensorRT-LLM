#!/usr/bin/env python3
"""
Memory profiling script for CUDA graph overhead analysis using natural warmup process.

This script measures the memory difference using TensorRT-LLM's natural CUDA graph creation:
1. max_concurrency=None: Creates CUDA graphs only for max_draft_len (speculation always on)  
2. max_concurrency>0: Creates CUDA graphs for both max_draft_len AND draft_len=0 (can disable speculation)

This follows the natural behavior in model_engine.warmup() where max_concurrency setting
determines whether additional CUDA graphs are created for draft_len=0.

Usage:
    python profile_cuda_graph_memory.py --model_path /path/to/model --model_name model_name
"""

import argparse
import torch
import gc
import time
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import json
import multiprocessing as mp
import io
import contextlib

# Force single-process mode for memory profiling access to model_engine
os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"

# Add TensorRT-LLM to path (parent directory since we're in memory_profile/)
sys.path.append(str(Path(__file__).parent.parent))

from tensorrt_llm.llmapi import LLM, CudaGraphConfig, NGramDecodingConfig


def get_memory_info() -> Dict[str, float]:
    """Get current CUDA memory usage in MB from the current device."""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "max_allocated": 0}
    
    # Use the current CUDA device (don't assume device 0)
    device = torch.cuda.current_device()
    print(f"Debug: Measuring memory on CUDA device {device}")
    
    return {
        "allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved(device) / 1024**2, 
        "max_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
    }


def reset_memory_stats():
    """Reset CUDA memory statistics on current device."""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"Debug: Resetting memory stats on CUDA device {device}")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        gc.collect()


def create_llm_with_natural_warmup(
    model_path: str,
    max_concurrency: Optional[int] = None,
    max_batch_size: int = 32,
    max_draft_len: int = 4,
) -> LLM:
    """Create LLM using authentic max_concurrency approach with NGram speculative decoding."""
    
    # Use natural CUDA graph generation (don't specify batch_sizes manually)
    cuda_graph_config = CudaGraphConfig(
        max_batch_size=max_batch_size,  # Let it generate natural batch sizes
        enable_padding=False,
    )
    
    # NGram speculative decoding - more compatible than MTP
    spec_config = NGramDecodingConfig(
        max_draft_len=max_draft_len,
        max_concurrency=max_concurrency,  # This is the key variable!
        max_matching_ngram_size=2,  # From official examples
        is_keep_all=True,
        is_use_oldest=True,
        is_public_pool=True,
    )
    
    # Follow the exact pattern from quickstart_advanced.py
    # Force single-process mode so we can access model_engine for memory profiling
    llm = LLM(
        model=model_path,
        max_batch_size=max_batch_size,
        max_num_tokens=1024,
        max_seq_len=2048,
        cuda_graph_config=cuda_graph_config,
        speculative_config=spec_config,
        disable_overlap_scheduler=True,  # NGram requires this
        tensor_parallel_size=1,  # Single GPU
        pipeline_parallel_size=1,  # Single process
    )
    
    return llm


def run_memory_experiment(
    model_path: str,
    max_concurrency: Optional[int],
    experiment_name: str,
    max_batch_size: int = 32,
    max_draft_len: int = 4,
) -> Dict[str, Any]:
    """Run one memory experiment and return results."""
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {experiment_name}")
    print(f"max_concurrency: {max_concurrency}")
    print(f"max_batch_size: {max_batch_size}")
    print(f"max_draft_len: {max_draft_len}")
    print(f"{'='*60}")
    
    # Reset memory before experiment
    reset_memory_stats()
    time.sleep(2)  # Let cleanup settle
    
    # Measure memory before LLM creation
    memory_before = get_memory_info()
    print(f"Memory before LLM: {memory_before}")
    
    try:
        print("Creating LLM instance (this includes model loading and CUDA graph capture)...")
        start_time = time.time()
        
        # Create LLM using natural warmup process
        llm = create_llm_with_natural_warmup(
            model_path=model_path,
            max_concurrency=max_concurrency,
            max_batch_size=max_batch_size,
            max_draft_len=max_draft_len,
        )
        
        creation_time = time.time() - start_time
        print(f"LLM creation took: {creation_time:.2f} seconds")
        
        # Measure memory after LLM creation (including warmup)
        memory_after = get_memory_info()
        print(f"Memory after LLM: {memory_after}")
        
        # Calculate memory difference
        memory_diff = {
            "allocated_mb": memory_after["allocated_mb"] - memory_before["allocated_mb"],
            "reserved_mb": memory_after["reserved_mb"] - memory_before["reserved_mb"],
            "max_allocated_mb": memory_after["max_allocated_mb"] - memory_before["max_allocated_mb"],
        }
        
        print(f"Memory difference: {memory_diff}")
        
        # Get CUDA graph info from the LLM's executor (now accessible due to single-process mode)
        if hasattr(llm._executor, 'model_engine'):
            model_engine = llm._executor.model_engine
        elif hasattr(llm._executor, 'engine') and hasattr(llm._executor.engine, 'model_engine'):
            model_engine = llm._executor.engine.model_engine
        else:
            print(f"Warning: Cannot access model_engine from {type(llm._executor)}")
            model_engine = None
            
        if model_engine is not None:
            # Correct total count across batch sizes and draft lengths
            if hasattr(model_engine, '_cuda_graphs'):
                num_cuda_graphs = sum(len(graphs_by_draft_len)
                                      for graphs_by_draft_len in model_engine._cuda_graphs.values())
            else:
                num_cuda_graphs = 0
            cuda_graph_batch_sizes = getattr(model_engine, '_cuda_graph_batch_sizes', [])
            
            print(f"Number of CUDA graphs captured (total): {num_cuda_graphs}")
            print(f"CUDA graph batch sizes: {cuda_graph_batch_sizes}")
            
            # Analyze CUDA graphs by draft length (correct nested structure)
            graph_breakdown = {}
            if hasattr(model_engine, '_cuda_graphs'):
                for batch_size, graphs_by_draft_len in model_engine._cuda_graphs.items():
                    for draft_len, graph in graphs_by_draft_len.items():
                        # Count graphs by draft length
                        key = f"draft_len_{draft_len}"
                        graph_breakdown[key] = graph_breakdown.get(key, 0) + 1
            
            print(f"CUDA graphs by draft length (counts): {graph_breakdown}")
            
            # 🔍 INVESTIGATE THE MEMORY PARADOX
            print(f"\n🔬 INVESTIGATING MEMORY PARADOX:")
            
            # 1. Memory pool analysis
            if hasattr(model_engine, '_cuda_graph_mem_pool'):
                print(f"CUDA graph memory pool: {model_engine._cuda_graph_mem_pool}")
            
            # 2. Estimate individual CUDA graph buffer sizes
            total_graph_buffer_memory = 0
            if len(model_engine._cuda_graphs) > 0:
                sample_count = 0
                for bs, graphs_by_draft in model_engine._cuda_graphs.items():
                    for draft_len, graph in graphs_by_draft.items():
                        if hasattr(graph, 'input_ids') and hasattr(graph, 'position_ids'):
                            input_mem = graph.input_ids.numel() * graph.input_ids.element_size()
                            pos_mem = graph.position_ids.numel() * graph.position_ids.element_size()
                            graph_buffer_mem = (input_mem + pos_mem) / (1024**2)  # MB
                            total_graph_buffer_memory += graph_buffer_mem
                            
                            # Show first few for debugging
                            if sample_count < 3:
                                print(f"  Graph[bs={bs}, draft_len={draft_len}]: {graph_buffer_mem:.2f} MB buffers")
                                sample_count += 1
                        else:
                            print(f"  Graph[bs={bs}, draft_len={draft_len}]: No input_ids or position_ids")

                
                print(f"Total CUDA graph buffer memory: {total_graph_buffer_memory:.1f} MB")
                avg_per_graph = total_graph_buffer_memory / num_cuda_graphs if num_cuda_graphs > 0 else 0
                print(f"Average buffer memory per graph: {avg_per_graph:.2f} MB")
            
            # 3. Check memory allocator patterns
            device = torch.cuda.current_device()
            stats = torch.cuda.memory_stats(device)
            reserved = stats.get("reserved_bytes.all.current", 0) / (1024**2)
            allocated = stats.get("allocated_bytes.all.current", 0) / (1024**2)
            fragmentation = reserved - allocated
            
            print(f"Memory allocator state:")
            print(f"  - Reserved: {reserved:.1f} MB")
            print(f"  - Allocated: {allocated:.1f} MB") 
            print(f"  - Fragmentation: {fragmentation:.1f} MB")
            print(f"  - Efficiency: {allocated/reserved*100:.1f}%" if reserved > 0 else "N/A")
            
            # Debug: Show detailed CUDA graph structure
            if hasattr(model_engine, '_cuda_graphs'):
                print(f"Debug: _cuda_graphs keys: {list(model_engine._cuda_graphs.keys())}")
                for bs, graphs_by_draft in model_engine._cuda_graphs.items():
                    print(f"Debug: batch_size {bs} has draft_lens: {list(graphs_by_draft.keys())}")
            
            # Debug: Show speculative config details
            if hasattr(model_engine, 'spec_config') and model_engine.spec_config:
                print(f"Debug: spec_dec_mode: {model_engine.spec_config.spec_dec_mode}")
                print(f"Debug: max_concurrency: {getattr(model_engine.spec_config, 'max_concurrency', 'N/A')}")
                print(f"Debug: use_one_engine: {model_engine.spec_config.spec_dec_mode.use_one_engine()}")
                print(f"Debug: max_draft_len: {getattr(model_engine.spec_config, 'max_draft_len', 'N/A')}")
                print(f"Debug: is_draft_model: {getattr(model_engine, 'is_draft_model', 'N/A')}")
                
                # Debug: Check warmup conditions from model_engine.py lines 747-752
                is_draft_model = getattr(model_engine, 'is_draft_model', False)
                max_draft_len = getattr(model_engine.spec_config, 'max_draft_len', 0)
                use_one_engine = model_engine.spec_config.spec_dec_mode.use_one_engine()
                max_concurrency = getattr(model_engine.spec_config, 'max_concurrency', None)
                
                condition_met = (not is_draft_model and max_draft_len > 0 
                               and not use_one_engine 
                               and max_concurrency is not None)
                               
                print(f"Debug: Warmup condition for extra graphs: {condition_met}")
                print(f"Debug:   - not is_draft_model: {not is_draft_model}")
                print(f"Debug:   - max_draft_len > 0: {max_draft_len > 0}")  
                print(f"Debug:   - not use_one_engine: {not use_one_engine}")
                print(f"Debug:   - max_concurrency is not None: {max_concurrency is not None}")
            else:
                print("Debug: No spec_config found")
        else:
            num_cuda_graphs = 0
            cuda_graph_batch_sizes = []
            graph_breakdown = {}
        
        # Cleanup
        del llm
        
        return {
            "experiment_name": experiment_name,
            "max_concurrency": max_concurrency,
            "max_batch_size": max_batch_size,
            "max_draft_len": max_draft_len,
            "memory_before": memory_before,
            "memory_after": memory_after,
            "memory_diff": memory_diff,
            "creation_time_seconds": creation_time,
            "num_cuda_graphs": num_cuda_graphs,
            "cuda_graph_batch_sizes": cuda_graph_batch_sizes,
            "graph_breakdown": graph_breakdown,
        }
        
    except Exception as e:
        print(f"Error in experiment {experiment_name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "experiment_name": experiment_name,
            "max_concurrency": max_concurrency,
            "error": str(e),
        }
    finally:
        # Cleanup memory
        reset_memory_stats()
        time.sleep(2)


def _child_run_memory_experiment(model_path: str,
                                 max_concurrency: Optional[int],
                                 experiment_name: str,
                                 max_batch_size: int,
                                 max_draft_len: int,
                                 result_queue: "mp.Queue") -> None:
    """Top-level target for multiprocessing spawn to run one experiment and return result + logs via queue."""
    log_buf = io.StringIO()
    with contextlib.redirect_stdout(log_buf), contextlib.redirect_stderr(log_buf):
        res = run_memory_experiment(
            model_path=model_path,
            max_concurrency=max_concurrency,
            experiment_name=experiment_name,
            max_batch_size=max_batch_size,
            max_draft_len=max_draft_len,
        )
    _log_text = log_buf.getvalue()
    res["log"] = _log_text
    res["log_lines"] = _log_text.splitlines()
    result_queue.put(res)


def _wrapped_child_run(model_path: str,
                       max_concurrency: Optional[int],
                       experiment_name: str,
                       max_batch_size: int,
                       max_draft_len: int,
                       result_queue: "mp.Queue",
                       force_separate_pools: bool) -> None:
    """Spawn-safe wrapper that sets env flags before running the experiment."""
    if force_separate_pools:
        os.environ["TLLM_FORCE_SEPARATE_CUDA_GRAPH_POOLS"] = "1"
    _child_run_memory_experiment(model_path, max_concurrency, experiment_name, max_batch_size, max_draft_len, result_queue)


def main():
    parser = argparse.ArgumentParser(description="Profile CUDA graph memory overhead")
    parser.add_argument("--model_path", required=True, help="Path to model directory")
    parser.add_argument("--model_name", default="llama3_70b", help="Model name for output")
    parser.add_argument("--max_batch_size", type=int, default=32, help="Max batch size")
    parser.add_argument("--max_draft_len", type=int, default=4, help="Max draft length")
    parser.add_argument("--max_concurrency_test", type=int, default=16, help="max_concurrency value for test")
    parser.add_argument("--output_file", help="Output JSON file path")
    parser.add_argument("--isolate_subprocess", action="store_true",
                        help="Run each experiment in a fresh subprocess to avoid allocator/JIT reuse effects")
    parser.add_argument("--force_separate_pools", action="store_true",
                        help="Force each CUDA graph to use a separate memory pool (for measurement)")
    
    args = parser.parse_args()

    # Ensure CUDA-safe multiprocessing when isolation is requested
    if args.isolate_subprocess:
        try:
            if mp.get_start_method(allow_none=True) != "spawn":
                mp.set_start_method("spawn", force=True)
        except RuntimeError:
            # Start method already set by parent; ignore
            pass
    
    print(f"CUDA Memory Profiling for {args.model_name}")
    print(f"Model path: {args.model_path}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        print(f"Current CUDA device: {current_device}")
        print(f"CUDA device name: {torch.cuda.get_device_name(current_device)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(current_device).total_memory / 1024**3:.1f} GB")
    
    experiments = []
    summary = {}

    # Ensure results directory exists and is writable early
    os.makedirs("results", exist_ok=True)
    try:
        os.chmod("results", 0o777)
    except PermissionError:
        pass

    def _run_in_subprocess(model_path, max_concurrency, experiment_name, max_batch_size, max_draft_len):
        """Run one experiment in a child process to isolate CUDA context and allocator state."""
        result_queue: mp.Queue = mp.Queue()
        proc = mp.Process(target=_wrapped_child_run,
                          args=(model_path, max_concurrency, experiment_name, max_batch_size, max_draft_len, result_queue, args.force_separate_pools))
        proc.start()
        proc.join()
        if not result_queue.empty():
            return result_queue.get()
        return {"experiment_name": experiment_name, "max_concurrency": max_concurrency, "error": "no_result"}
    
    # Experiment 1: max_concurrency=None (only max_draft_len CUDA graphs)
    if args.isolate_subprocess:
        result1 = _run_in_subprocess(
            args.model_path, None, "max_concurrency_None", args.max_batch_size, args.max_draft_len
        )
        # Echo child logs to terminal for visibility
        if isinstance(result1, dict):
            if "log" in result1:
                print(result1["log"], end="")
            elif "log_lines" in result1:
                print("\n".join(result1["log_lines"]) + "\n", end="")
    else:
        buf1 = io.StringIO()
        with contextlib.redirect_stdout(buf1), contextlib.redirect_stderr(buf1):
            result1 = run_memory_experiment(
                model_path=args.model_path,
                max_concurrency=None,
                experiment_name="max_concurrency_None",
                max_batch_size=args.max_batch_size,
                max_draft_len=args.max_draft_len,
            )
        _log_text1 = buf1.getvalue()
        result1["log"] = _log_text1
        result1["log_lines"] = _log_text1.splitlines()
        print(_log_text1, end="")
    experiments.append(result1)
     
    # Wait between experiments
    time.sleep(5)
    
    # Experiment 2: max_concurrency>0 (both max_draft_len + draft_len=0 CUDA graphs)
    if args.isolate_subprocess:
        result2 = _run_in_subprocess(
            args.model_path, args.max_concurrency_test, f"max_concurrency_{args.max_concurrency_test}", args.max_batch_size, args.max_draft_len
        )
        if isinstance(result2, dict):
            if "log" in result2:
                print(result2["log"], end="")
            elif "log_lines" in result2:
                print("\n".join(result2["log_lines"]) + "\n", end="")
    else:
        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2), contextlib.redirect_stderr(buf2):
            result2 = run_memory_experiment(
                model_path=args.model_path,
                max_concurrency=args.max_concurrency_test,
                experiment_name=f"max_concurrency_{args.max_concurrency_test}",
                max_batch_size=args.max_batch_size,
                max_draft_len=args.max_draft_len,
            )
        _log_text2 = buf2.getvalue()
        result2["log"] = _log_text2
        result2["log_lines"] = _log_text2.splitlines()
        print(_log_text2, end="")
    experiments.append(result2)
    
    # Analysis
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}")
    
    if "error" not in result1 and "error" not in result2:
        # Calculate overhead (report both allocated and reserved)
        mem1_alloc = result1["memory_diff"]["allocated_mb"]
        mem2_alloc = result2["memory_diff"]["allocated_mb"] 
        overhead_alloc_mb = mem2_alloc - mem1_alloc
        overhead_alloc_pct = (overhead_alloc_mb / mem1_alloc * 100) if mem1_alloc > 0 else 0

        mem1_resv = result1["memory_diff"]["reserved_mb"]
        mem2_resv = result2["memory_diff"]["reserved_mb"] 
        overhead_resv_mb = mem2_resv - mem1_resv
        overhead_resv_pct = (overhead_resv_mb / mem1_resv * 100) if mem1_resv > 0 else 0
        
        graphs1 = result1["num_cuda_graphs"]
        graphs2 = result2["num_cuda_graphs"]
        
        print(f"Allocated delta (max_concurrency=None):     {mem1_alloc:.1f} MB")
        print(f"Allocated delta (max_concurrency={args.max_concurrency_test}): {mem2_alloc:.1f} MB")
        print(f"Allocated overhead: {overhead_alloc_mb:.1f} MB ({overhead_alloc_pct:.1f}% change)")
        print(f"Reserved delta  (max_concurrency=None):     {mem1_resv:.1f} MB")
        print(f"Reserved delta  (max_concurrency={args.max_concurrency_test}): {mem2_resv:.1f} MB")
        print(f"Reserved overhead: {overhead_resv_mb:.1f} MB ({overhead_resv_pct:.1f}% change)")
        print(f"CUDA graphs (max_concurrency=None):     {graphs1}")
        print(f"CUDA graphs (max_concurrency={args.max_concurrency_test}): {graphs2}")
        additional_graphs = graphs2 - graphs1
        print(f"Additional CUDA graphs: {additional_graphs}")
        
        # Possible explanations:
        # print(f"\n💡 POSSIBLE EXPLANATIONS:")
        # print(f"1. Memory Pool Sharing: CUDA graphs may share memory pools")
        # print(f"2. KV Cache Efficiency: max_concurrency affects KV allocation")
        # print(f"3. Allocator Optimization: PyTorch allocator behaves differently")
        # print(f"4. Buffer Reuse: draft_len=0 graphs may reuse existing buffers")
        
        if graphs2 > graphs1:
            actual_per_graph_resv = overhead_resv_mb / (graphs2 - graphs1)
            print(f"Actual memory per additional CUDA graph (reserved): {actual_per_graph_resv:.1f} MB")
        
        # Summary
        summary = {
            "model_name": args.model_name,
            "allocated_overhead_mb": overhead_alloc_mb,
            "allocated_overhead_pct": overhead_alloc_pct,
            "reserved_overhead_mb": overhead_resv_mb,
            "reserved_overhead_pct": overhead_resv_pct,
            "additional_cuda_graphs": graphs2 - graphs1,
            "reserved_memory_per_graph_mb": overhead_resv_mb / (graphs2 - graphs1) if graphs2 > graphs1 else 0,
        }
        
        print(f"\nSUMMARY: {summary}")
        experiments.append({"summary": summary})
    
    # Save a single master JSON containing both runs and full logs
    master_file = args.output_file or f"results/results_{args.model_name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(master_file, 'w') as f:
            json.dump({
                "model_name": args.model_name,
                "experiments": experiments,
                "summary": summary
            }, f, indent=2)
        print(f"\nMaster summary saved to: {master_file}")
    except PermissionError as e:
        print(f"Warning: Could not save master summary to {master_file}: {e}")


if __name__ == "__main__":
    main()
