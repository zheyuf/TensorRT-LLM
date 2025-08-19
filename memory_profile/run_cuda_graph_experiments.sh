#!/usr/bin/env bash
set -euo pipefail

# Single-run CUDA graph memory experiments matrix
#
# Usage:
#   ./memory_profile/run_cuda_graph_experiments.sh \
#     /models/DeepSeek-R1-Distill-Llama-8B \
#     DeepSeek-R1-Distill-Llama-8B \
#     "16 32 64" \
#     "4 8" \
#     16
#
# Args:
#   $1 = MODEL_PATH (required)
#   $2 = MODEL_NAME (required)
#   $3 = BATCH_SIZES (optional, default: "16 32 64")
#   $4 = DRAFT_LENS  (optional, default: "4 8")
#   $5 = MC_ON       (optional, default: 16)

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 MODEL_PATH MODEL_NAME [BATCH_SIZES] [DRAFT_LENS] [MC_ON]" >&2
  exit 1
fi

MODEL_PATH="$1"
MODEL_NAME="$2"
BATCH_SIZES_STR="${3:-"16 32 64"}"
DRAFT_LENS_STR="${4:-"4 8"}"
MC_ON="${5:-16}"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="$ROOT_DIR/memory_profile/results"
mkdir -p "$RESULTS_DIR"

# Create model-specific directory and set permissions for Docker container access
MODEL_RESULTS_DIR="$RESULTS_DIR/${MODEL_NAME}"
mkdir -p "$MODEL_RESULTS_DIR"
chmod 777 "$MODEL_RESULTS_DIR"

echo "Root dir: $ROOT_DIR"
echo "Results dir: $RESULTS_DIR"
echo "Model: $MODEL_NAME @ $MODEL_PATH"
echo "Batch sizes: $BATCH_SIZES_STR"
echo "Draft lens:  $DRAFT_LENS_STR"
echo "Graphs-ON max_concurrency: $MC_ON"

cd "$ROOT_DIR"

timestamp() { date '+%Y%m%d_%H%M%S'; }

run_single() {
  local bs="$1"; local draft="$2"; local mc="$3"; local sepPoolsFlag="$4"
  local sepTag="sepPools0"; local sepArg=""
  if [[ "$sepPoolsFlag" == "1" ]]; then
    sepTag="sepPools1"; sepArg="--force_separate_pools"
  fi
  local ts="$(timestamp)"
  local out="$RESULTS_DIR/${MODEL_NAME}/bs${bs}_draft${draft}_mc${mc}_${sepTag}_${ts}.json"

  echo "\n=== Running: bs=$bs draft=$draft mc=$mc $sepTag ==="
  python memory_profile/profile_cuda_graph_memory_single.py \
    --model_path "$MODEL_PATH" \
    --model_name "$MODEL_NAME" \
    --max_batch_size "$bs" \
    --max_draft_len "$draft" \
    --max_concurrency "$mc" \
    --output_file "$out" \
    $sepArg
}

# Iterate matrix
for bs in $BATCH_SIZES_STR; do
  for draft in $DRAFT_LENS_STR; do
    # Graphs OFF baseline (mc=None)
    run_single "$bs" "$draft" "None" "0"
    # Graphs OFF baseline (mc=None) with separate pools
    run_single "$bs" "$draft" "None" "1"
    # Graphs ON (doubles graphs)
    run_single "$bs" "$draft" "$MC_ON" "0"
    # Graphs ON (doubles graphs) with separate pools
    run_single "$bs" "$draft" "$MC_ON" "1"
  done
done

echo "\nAll experiments completed. JSON files are in: $RESULTS_DIR"


