#!/bin/bash
# Indo-NLI Model Evaluation Project Runner
# This script provides a complete workflow for the IndoNLI evaluation

set -e  # Exit on error

# Configuration
VENV_DIR="venv"
MODELS_DIR="models"
REPORTS_DIR="reports/comparison"
LOG_DIR="logs"
TEST_SETS=("test_lay" "test_expert")
BATCH_SIZE=16
NUM_EPOCHS=5
SEED=42

# Get the absolute path to the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse command line arguments
for arg in "$@"; do
  case $arg in
    --no-setup)
      SKIP_SETUP=true
      shift
      ;;
    --no-train)
      SKIP_TRAIN=true
      shift
      ;;
    --no-eval)
      SKIP_EVAL=true
      shift
      ;;
    --no-report)
      SKIP_REPORT=true
      shift
      ;;
    --push-to-hub)
      PUSH_TO_HUB=true
      shift
      ;;
    --batch-size=*)
      BATCH_SIZE="${arg#*=}"
      shift
      ;;
    --epochs=*)
      NUM_EPOCHS="${arg#*=}"
      shift
      ;;
    --seed=*)
      SEED="${arg#*=}"
      shift
      ;;
    --model=*)
      SPECIFIC_MODEL="${arg#*=}"
      shift
      ;;
    --help)
      echo "Usage: ./run_experiments.sh [options]"
      echo "Options:"
      echo "  --no-setup       Skip environment setup"
      echo "  --no-train       Skip model training"
      echo "  --no-eval        Skip model evaluation"
      echo "  --no-report      Skip report generation"
      echo "  --push-to-hub    Push models to Hugging Face Hub"
      echo "  --batch-size=N   Set batch size (default: 16)"
      echo "  --epochs=N       Set number of epochs (default: 5)"
      echo "  --seed=N         Set random seed (default: 42)"
      echo "  --model=NAME     Run only for a specific model"
      echo "                   (options: indo-roberta, indo-roberta-base, sentence-bert,"
      echo "                    sentence-bert-simple, sentence-bert-proper)"
      echo "  --help           Show this help message"
      exit 0
      ;;
  esac
done

# Create directories
mkdir -p $LOG_DIR

# Print configuration
echo "=== Indo-NLI Model Evaluation ==="
echo "Project Root: $PROJECT_ROOT"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "Seed: $SEED"
if [ ! -z "$SPECIFIC_MODEL" ]; then
  echo "Running only for model: $SPECIFIC_MODEL"
fi
echo "================================="

# Setup environment
if [ "$SKIP_SETUP" != "true" ]; then
  echo "Setting up environment..."
  
  # Create virtual environment if it doesn't exist
  if [ ! -d "$VENV_DIR" ]; then
    python -m venv $VENV_DIR
  fi
  
  # Activate virtual environment
  source $VENV_DIR/bin/activate
  
  # Install requirements
  pip install -r requirements.txt
  
  # Install the project in development mode
  pip install -e .
  
  echo "Environment setup complete."
fi

# Function to run Python command with proper Python path
run_python() {
  # Use PYTHONPATH to ensure 'src' is findable
  PYTHONPATH=$PROJECT_ROOT python "$@"
}

# Function to run training and evaluation for a specific model
run_model() {
  model_config=$1
  model_name=$(basename $model_config .yaml)
  
  echo "=== Processing model: $model_name ==="
  
  # Training
  if [ "$SKIP_TRAIN" != "true" ]; then
    echo "Training $model_name..."
    run_python scripts/train.py \
      --config $model_config \
      --batch_size $BATCH_SIZE \
      --num_epochs $NUM_EPOCHS \
      --seed $SEED \
      --fp16 2>&1 | tee $LOG_DIR/train_${model_name}.log
  fi
  
  # Evaluation
  if [ "$SKIP_EVAL" != "true" ]; then
    for test_set in "${TEST_SETS[@]}"; do
      echo "Evaluating $model_name on $test_set..."
      model_dir="$MODELS_DIR/$(basename $model_name | tr '_' '-')"
      best_model_path="$model_dir/best"
      
      # Check if best model exists, otherwise use final model
      if [ ! -d "$best_model_path" ]; then
        if [ -d "$model_dir/final" ]; then
          best_model_path="$model_dir/final"
        else
          # Find the latest epoch
          best_model_path=$(find "$model_dir" -name "epoch-*" -type d | sort -V | tail -n 1)
          if [ -z "$best_model_path" ]; then
            echo "No model found for $model_name. Skipping evaluation."
            continue
          fi
        fi
      fi
      
      run_python scripts/evaluate.py \
        --model_path $best_model_path \
        --model_name $model_name \
        --test_set $test_set \
        --seed $SEED 2>&1 | tee $LOG_DIR/eval_${model_name}_${test_set}.log
    done
  fi
  
  # Push to Hub
  if [ "$PUSH_TO_HUB" = "true" ]; then
    echo "Pushing $model_name to Hugging Face Hub..."
    model_dir="$MODELS_DIR/$(basename $model_name | tr '_' '-')"
    best_model_path="$model_dir/best"
    
    # Check if best model exists, otherwise use final model
    if [ ! -d "$best_model_path" ]; then
      if [ -d "$model_dir/final" ]; then
        best_model_path="$model_dir/final"
      else
        # Find the latest epoch
        best_model_path=$(find "$model_dir" -name "epoch-*" -type d | sort -V | tail -n 1)
        if [ -z "$best_model_path" ]; then
          echo "No model found for $model_name. Skipping push to Hub."
          continue
        fi
      fi
    fi
    
    # Customize your repo name and organization as needed
    repo_name="indonli-${model_name}"
    
    run_python scripts/push_to_hub.py \
      --model_path $best_model_path \
      --model_name $model_name \
      --repo_name $repo_name 2>&1 | tee $LOG_DIR/push_${model_name}.log
  fi
  
  echo "=== Finished processing model: $model_name ==="
}

# Process all models or a specific model
if [ ! -z "$SPECIFIC_MODEL" ]; then
  # Run only for the specified model
  config_file="configs/${SPECIFIC_MODEL}.yaml"
  if [ -f "$config_file" ]; then
    run_model $config_file
  else
    echo "Error: Config file not found for model $SPECIFIC_MODEL"
    exit 1
  fi
else
  # Run for all models
  for config_file in configs/*.yaml; do
    run_model $config_file
  done
fi

# Generate comprehensive report
if [ "$SKIP_REPORT" != "true" ]; then
  echo "Generating comparison report..."
  
  test_sets_arg=""
  for test_set in "${TEST_SETS[@]}"; do
    test_sets_arg="$test_sets_arg $test_set"
  done
  
  run_python scripts/generate_report.py \
    --models_dir $MODELS_DIR \
    --output_dir $REPORTS_DIR \
    --test_sets $test_sets_arg 2>&1 | tee $LOG_DIR/report_generation.log
  
  echo "Report generation complete. Results available in $REPORTS_DIR"
fi

echo "All tasks completed successfully!"
