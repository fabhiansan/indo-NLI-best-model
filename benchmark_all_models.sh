#!/bin/bash
# Comprehensive Model Benchmarking Runner
# This script runs the benchmark_models.py script to evaluate all model checkpoints

set -e  # Exit on error

# Configuration
MODELS_DIR="$1"
OUTPUT_DIR="reports/benchmark"
BATCH_SIZE=16
MAX_LENGTH=128
DATASET_NAME="afaji/indonli"
SEED=42

# Check if models directory was provided
if [ -z "$MODELS_DIR" ]; then
  echo "Error: You must provide the models directory as the first argument"
  echo "Usage: ./benchmark_all_models.sh <models_dir>"
  exit 1
fi

# Make sure the models directory exists
if [ ! -d "$MODELS_DIR" ]; then
  echo "Error: Models directory '$MODELS_DIR' does not exist"
  exit 1
fi

# Create reports directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "=== Starting comprehensive model benchmarking ==="
echo "Models directory: $MODELS_DIR"
echo "Results will be saved to: $OUTPUT_DIR"

# Run the benchmark script
python scripts/benchmark_models.py \
  --models_dir "$MODELS_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --splits validation test_lay test_expert \
  --batch_size "$BATCH_SIZE" \
  --max_length "$MAX_LENGTH" \
  --dataset_name "$DATASET_NAME" \
  --seed "$SEED"

echo "=== Benchmarking complete ==="
echo "Check the results in $OUTPUT_DIR/$(ls -t $OUTPUT_DIR | head -1)"
