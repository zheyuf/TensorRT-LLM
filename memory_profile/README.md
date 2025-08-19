# Memory Profiling Tools

This directory contains tools for analyzing CUDA graph memory overhead in TensorRT-LLM using **natural warmup processes**.

## What This Measures

Uses TensorRT-LLM's authentic behavior to measure real speculative decoding memory overhead:
- **max_concurrency=None**: CUDA graphs only for `max_draft_len` (speculation always-on)
- **max_concurrency>0**: CUDA graphs for both `max_draft_len` AND `draft_len=0` (runtime control)

This follows the exact logic in `model_engine.warmup()` where `max_concurrency` setting determines whether additional CUDA graphs are created for `draft_len=0`. Uses NGram speculative decoding for better compatibility.

## Files

- `profile_cuda_graph_memory.py`: Main memory profiling script
- `run_memory_profile.sh`: Executable bash wrapper
- `setup_container.sh`: Container dependency setup script
- `README.md`: This documentation
- `results/`: Directory for output JSON files (auto-created)

## Quick Start

```bash
# Navigate to the memory profiling directory
cd memory_profile

# First time in container: Set up dependencies
./setup_container.sh

# Run with your model (example with mounted model)
./run_memory_profile.sh /models/DeepSeek-R1-Distill-Llama-8B DeepSeek-R1-Distill-Llama-8B
```

## Usage

### Method 1: Using the bash script (recommended)
```bash
cd memory_profile

# First time in container: Set up dependencies
./setup_container.sh

# Run memory profiling
./run_memory_profile.sh /path/to/your/model model_name
```

### Method 2: Direct Python execution
```bash
cd memory_profile

# First time in container: Set up dependencies
./setup_container.sh

# Run memory profiling
python profile_cuda_graph_memory.py \
    --model_path /path/to/your/model \
    --model_name model_name \
    --max_batch_size 32 \
    --max_draft_len 4 \
    --max_concurrency_test 16
```

## Expected Output

The script will run two experiments and output something like:

```
Running experiment: max_concurrency_None
Memory before LLM: {'allocated_mb': 0.0, 'reserved_mb': 0.0, 'max_allocated_mb': 0.0}
Memory after LLM: {'allocated_mb': 3456.7, 'reserved_mb': 4000.0, ...}
Number of CUDA graphs captured: 32
CUDA graphs by draft length: {'draft_len_4': 32}

Running experiment: max_concurrency_16
Memory before LLM: {'allocated_mb': 0.0, 'reserved_mb': 0.0, 'max_allocated_mb': 0.0}
Memory after LLM: {'allocated_mb': 6789.1, 'reserved_mb': 8000.0, ...}
Number of CUDA graphs captured: 64
CUDA graphs by draft length: {'draft_len_4': 32, 'draft_len_0': 32}

ANALYSIS
Memory usage (max_concurrency_None):     3456.7 MB
Memory usage (max_concurrency_16):       6789.1 MB
Memory overhead: 3332.4 MB (96.4% increase)
Additional CUDA graphs: 32
Memory per additional CUDA graph: 104.1 MB
```

## Key Features

✅ **Authentic Behavior**: Uses TensorRT-LLM's natural max_concurrency logic  
✅ **Real Memory Impact**: Measures actual speculative decoding overhead  
✅ **Production Code Path**: Same warmup logic as model_engine.warmup()  
✅ **NGram Compatible**: Uses reliable speculative decoding method

## Key Metrics

The script measures:
1. **Memory overhead**: Extra memory used when capturing additional CUDA graphs
2. **Number of additional CUDA graphs**: How many extra graphs are created
3. **Memory per graph**: Average memory cost per additional CUDA graph
4. **Percentage increase**: Relative memory overhead

## Interpreting Results

- **Low overhead** (< 500MB): Might not be worth optimizing
- **High overhead** (> 2GB): Strong case for optimization  
- **Memory per graph**: Should be ~200MB per graph as documented in TensorRT-LLM

## Results Storage

Results are saved to the `memory_profile/results/` directory as JSON files with timestamps:
`results/results_ModelName_YYYYMMDD_HHMMSS.json`

## Troubleshooting

### Container Setup Issues:
1. **ModuleNotFoundError**: Run `./setup_container.sh` first to install dependencies
2. **Missing xgrammar**: The setup script installs all required dependencies including xgrammar

### Runtime Issues:
1. **CUDA out of memory**: Reduce `--max_batch_size` or use smaller model
2. **Model not found**: Check `--model_path` points to valid model directory
3. **Import errors**: Make sure you're running from within Docker container

### One-time Setup:
Each time you enter a new container, run the setup script once:
```bash
cd memory_profile
./setup_container.sh
```

### Debug Mode:
Add more verbose output by modifying the script to include:
```python
print(f"CUDA graphs structure: {model_engine._cuda_graphs}")
```

## Model Recommendations

For testing, use models like:
- Llama 3 8B (smaller, faster to test)
- Llama 3 70B (production-like size) 
- DeepSeek models (have good MTP support)

Avoid very small models as CUDA graph overhead might be proportionally different.