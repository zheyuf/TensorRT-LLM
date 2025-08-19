#!/bin/bash

# Simple runner script for CUDA graph memory profiling
# Usage: ./run_memory_profile.sh /path/to/model

MODEL_PATH=${1:-"./models/llama3_70b"}
MODEL_NAME=${2:-"llama3_70b"}

echo "Running CUDA graph memory profiling..."
echo "Model path: $MODEL_PATH"
echo "Model name: $MODEL_NAME"

# Make sure we're in the right directory
cd "$(dirname "$0")"

# Run the profiling script
python profile_cuda_graph_memory.py \
    --model_path "$MODEL_PATH" \
    --model_name "$MODEL_NAME" \
    --max_batch_size 32 \
    --max_draft_len 4 \
    --max_concurrency_test 16 \
    --output_file "results/results_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).json"

echo "Profiling complete!"
