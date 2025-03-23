#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive Model Benchmarking Script

This script:
1. Discovers all model checkpoints in the specified directory
2. Evaluates each model on validation, test_lay, and test_expert datasets
3. Collects results into a central location with structured naming
4. Creates a summary table for easy comparison

Usage:
    python benchmark_models.py --models_dir models --output_dir reports/benchmark
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import torch
import yaml
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from transformers import set_seed
import numpy as np

# Add the parent directory to the path so we can import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import get_nli_dataloader
from src.models.model_factory import ModelFactory
from src.utils.metrics import compute_metrics


# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
console = Console()


def find_model_checkpoints(
    models_dir: str, recursive: bool = True
) -> List[Tuple[str, Path]]:
    """
    Find all model checkpoints in the specified directory.
    
    Args:
        models_dir: Root directory to search for models
        recursive: Whether to search recursively
        
    Returns:
        List of tuples containing (model_name, checkpoint_path)
    """
    logger.info(f"Looking for model checkpoints in {models_dir}")
    
    # Check if models_dir exists
    if not os.path.exists(models_dir):
        logger.warning(f"Models directory {models_dir} does not exist")
        
        # Try the logs directory as fallback
        logs_dir = "logs"
        if os.path.exists(logs_dir):
            logger.info(f"Trying to find models in {logs_dir} directory instead")
            models_dir = logs_dir
        else:
            logger.error("No model checkpoints found")
            return []
    
    models_path = Path(models_dir)
    checkpoints = []
    
    if not models_path.exists():
        logger.warning("Models directory %s does not exist", models_dir)
        return []
    
    # Find all model directories
    logger.info(f"Searching for model checkpoints in: {models_path.absolute()}")
    model_dirs = []
    if recursive:
        for path in models_path.glob("**"):
            if path.is_dir() and any((path / subdir).exists() for subdir in ["best", "final", "checkpoint-*"]):
                model_dirs.append(path)
                logger.info(f"Found potential model directory: {path}")
    else:
        model_dirs = [p for p in models_path.iterdir() if p.is_dir()]
        for dir in model_dirs:
            logger.info(f"Found potential model directory: {dir}")
        
    # Find checkpoints within each model directory
    for model_dir in model_dirs:
        model_name = model_dir.name
        logger.info(f"Searching for checkpoints in model directory: {model_dir}")
        
        # Check for 'best' checkpoint
        best_dir = model_dir / "best"
        if best_dir.exists():
            logger.info(f"Found 'best' checkpoint for {model_name} at {best_dir}")
            checkpoints.append((model_name, best_dir))
        
        # Check for 'final' checkpoint
        final_dir = model_dir / "final"
        if final_dir.exists():
            logger.info(f"Found 'final' checkpoint for {model_name} at {final_dir}")
            checkpoints.append((model_name, final_dir))
            
        # Check for epoch checkpoints
        epoch_patterns = ["epoch-*", "epoch_*", "*_epoch_*", "*-epoch-*", "checkpoint-*"]
        epoch_found = False
        
        for pattern in epoch_patterns:
            for epoch_dir in model_dir.glob(pattern):
                if epoch_dir.is_dir():
                    epoch_found = True
                    # Extract epoch number for sorting
                    match = re.search(r"epoch[-_]?(\d+)", str(epoch_dir))
                    if match:
                        epoch_num = int(match.group(1))
                        checkpoint_name = f"{model_name}-epoch-{epoch_num}"
                        logger.info("Found epoch checkpoint: %s at %s", checkpoint_name, epoch_dir)
                        checkpoints.append((checkpoint_name, epoch_dir))
                    else:
                        # If no regex match, just use the directory name
                        checkpoint_name = f"{model_name}-{epoch_dir.name}"
                        logger.info("Found checkpoint with non-standard name: %s at %s", checkpoint_name, epoch_dir)
                        checkpoints.append((checkpoint_name, epoch_dir))
        
        if not epoch_found:
            logger.warning("No epoch checkpoints found in %s", model_dir)
    
    # Sort by model name
    checkpoints.sort(key=lambda x: x[0])
    
    if checkpoints:
        logger.info(f"Found {len(checkpoints)} model checkpoints:")
        for name, path in checkpoints:
            logger.info(f"  - {name}: {path}")
    else:
        logger.warning(f"No model checkpoints found in {models_dir}")
    
    return checkpoints


def normalize_model_name(model_name: str) -> str:
    """
    Normalize model name to determine its type.
    
    Args:
        model_name: Model name
        
    Returns:
        Normalized model name
    """
    # First strip any epoch or base suffixes for the initial classification
    base_model_name = re.sub(r'-epoch-\d+$', '', model_name)
    base_model_name = re.sub(r'-base(?:-epoch-\d+)?$', '', base_model_name)
    
    logger.info(f"Normalizing model name: {model_name} -> base name: {base_model_name}")
    
    if "roberta" in base_model_name.lower():
        normalized = "Indo-roBERTa"
    elif "sentence-bert" in base_model_name.lower():
        # Handle different sentence-bert variants
        if "simple" in base_model_name.lower():
            normalized = "Sentence-BERT-Simple"
        elif "proper" in base_model_name.lower():
            normalized = "Sentence-BERT-Proper"
        else:
            normalized = "Sentence-BERT"
    else:
        # Default to a simple capitalization for unknown models
        normalized = base_model_name.capitalize()
    
    logger.info(f"Normalized model name: {model_name} -> {normalized}")
    return normalized


def evaluate_model(
    model_path: Path,
    model_name: str,
    split: str,
    batch_size: int = 32,
    max_length: int = 128,
    dataset_name: str = "afaji/indonli",
    seed: int = 42,
    debug: bool = False,
) -> Dict[str, float]:
    """Evaluate a model on a specific data split."""
    logger.info("=" * 80)
    logger.info(f"EVALUATING MODEL: {model_name}")
    logger.info(f"MODEL PATH: {model_path.absolute()}")
    logger.info(f"DATASET SPLIT: {split}")
    logger.info("=" * 80)
    
    # Check if the model path exists
    if not model_path.exists():
        logger.error(f"❌ MODEL PATH DOES NOT EXIST: {model_path.absolute()}")
        return {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0, "error": "Model path does not exist"}
    
    # List all files in the model directory
    logger.info("Model directory contents:")
    for file in model_path.glob("*"):
        logger.info(f"  - {file.name} {'(directory)' if file.is_dir() else f'({file.stat().st_size} bytes)'}")
    
    try:
        # Normalize model name to determine its type
        normalized_name = normalize_model_name(model_name)
        logger.info(f"Using normalized model type: '{normalized_name}' for model '{model_name}'")
        
        # Load the model
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            # Load model with proper error handling
            logger.info(f"🔄 LOADING MODEL: {model_name} from {model_path}")
            
            try:
                # Check for config.json
                config_path = model_path / "config.json"
                if config_path.exists():
                    logger.info(f"Found config.json at {config_path}")
                    with open(config_path, "r") as f:
                        logger.info(f"Config preview: {f.read()[:200]}...")
                
                # Check for pytorch_model.bin
                model_bin = model_path / "pytorch_model.bin"
                if model_bin.exists():
                    logger.info(f"Found pytorch_model.bin at {model_bin} ({model_bin.stat().st_size} bytes)")
                
                # Check for pretrained_model_name.txt
                pretrained_name_file = model_path / "pretrained_model_name.txt"
                if pretrained_name_file.exists():
                    with open(pretrained_name_file, "r") as f:
                        pretrained_model_name = f.read().strip()
                        logger.info(f"Found pretrained_model_name.txt: {pretrained_model_name}")
            except Exception as e:
                logger.warning(f"Error checking model files: {str(e)}")
            
            model = ModelFactory.from_pretrained(str(model_path), model_name=normalized_name)
            model.to(device)
            model.eval()
            logger.info(f"✅ MODEL LOADED SUCCESSFULLY: {model.__class__.__name__}")
            
            # Get tokenizer with robust fallback
            logger.info(f"🔄 LOADING TOKENIZER for {model_name}")
            try:
                tokenizer = ModelFactory.get_tokenizer_for_model(str(model_path), model_name=normalized_name)
                logger.info(f"✅ TOKENIZER LOADED SUCCESSFULLY: {tokenizer.__class__.__name__}")
            except Exception as e:
                logger.warning(f"❌ Error loading tokenizer via ModelFactory: {str(e)}")
                
                # Direct fallbacks for specific model types
                if "RoBERTa" in normalized_name:
                    logger.info("Falling back to default RoBERTa tokenizer")
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained("cahya/roberta-base-indonesian-1.5G")
                elif "BERT" in normalized_name:
                    logger.info("Falling back to default BERT tokenizer")
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained("firqaaa/indo-sentence-bert-base")
                else:
                    # Last resort - try direct loading
                    logger.info("Attempting to load tokenizer directly from model path")
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(
                        str(model_path), use_fast=True, local_files_only=False
                    )
            
            # Load the dataset
            logger.info(f"🔄 LOADING DATASET: {split} from {dataset_name}")
            dataloader = get_nli_dataloader(
                tokenizer=tokenizer,
                split=split,
                batch_size=batch_size,
                max_length=max_length,
                dataset_name=dataset_name,
                shuffle=False,
            )
            logger.info(f"✅ DATASET LOADED: {len(dataloader)} batches")
            
            # Evaluation
            logger.info(f"🔄 STARTING EVALUATION on {split}")
            all_logits = []
            all_labels = []
            all_preds = []
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {model_name} on {split}")):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    
                    token_type_ids = batch.get("token_type_ids", None)
                    if token_type_ids is not None:
                        token_type_ids = token_type_ids.to(device)
                    
                    labels = batch["labels"].to(device)
                    
                    # Forward pass with proper error handling
                    try:
                        # If debug and first batch, log input shape
                        if debug and batch_idx == 0:
                            logger.info(f"Input shapes - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}")
                            if token_type_ids is not None:
                                logger.info(f"token_type_ids shape: {token_type_ids.shape}")
                                
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids if token_type_ids is not None else None,
                        )
                        
                        # Extract logits correctly based on model output type
                        if isinstance(outputs, dict):
                            logits = outputs.get("logits", None)
                            if logits is None:
                                # Try other common keys
                                for key in ["scores", "predictions", "output"]:
                                    if key in outputs:
                                        logits = outputs[key]
                                        break
                                        
                            # Log what we found
                            # logger.info(f"Found logits in output dict with key: {[k for k, v in outputs.items() if v is logits][0] if logits is not None else 'NOT FOUND'}")
                                
                        elif isinstance(outputs, tuple) and len(outputs) > 0:
                            logits = outputs[0]
                            logger.info("Found logits as first element in output tuple")
                        else:
                            logits = outputs
                            logger.info("Using outputs directly as logits")
                            
                        # Ensure logits is a tensor
                        if not isinstance(logits, torch.Tensor):
                            raise ValueError(f"Unexpected logits type: {type(logits)}")
                            
                        # Debug first batch logits
                        if debug and batch_idx == 0:
                            logger.info(f"Logits shape: {logits.shape}")
                            logger.info(f"Logits sample (first 2 examples):\n{logits[:2]}")
                            
                        # Get predictions
                        preds = torch.argmax(logits, dim=1)
                        
                        # Add to all results
                        all_logits.append(logits.cpu().numpy())
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                        
                    except Exception as e:
                        logger.error(f"❌ Error during model inference: {str(e)}")
                        raise
            
            # Convert logits list to numpy array
            all_logits = np.vstack(all_logits)
            all_labels = np.array(all_labels)
            all_preds = np.array(all_preds)
            
            # Debug class distribution
            label_counts = np.bincount(all_labels.astype(np.int64), minlength=3)
            pred_counts = np.bincount(all_preds.astype(np.int64), minlength=3)
            
            logger.info("=" * 40)
            logger.info("EVALUATION STATISTICS")
            logger.info(f"Total examples: {len(all_labels)}")
            logger.info(f"Label distribution: {label_counts} (entailment, neutral, contradiction)")
            logger.info(f"Prediction distribution: {pred_counts} (entailment, neutral, contradiction)")
            logger.info("=" * 40)
            
            # Check for degenerate predictions (all same class)
            unique_preds = np.unique(all_preds)
            if len(unique_preds) == 1:
                logger.warning(
                    f"⚠️ DEGENERATE PREDICTIONS DETECTED: Model {model_name} is predicting only class {unique_preds[0]}"
                )
                # Add sample of logits to understand the issue
                logger.warning(f"Sample of logits (first 5 examples):\n{all_logits[:5]}")
            
            # Show prediction examples
            logger.info("Sample predictions (first 10):")
            for i in range(min(10, len(all_preds))):
                logger.info(f"  Example {i}: True={all_labels[i]}, Pred={all_preds[i]}, Logits={all_logits[i]}")
            
            # Compute metrics with direct logits instead of dummy logits
            logger.info("🔄 Computing evaluation metrics")
            metrics = compute_metrics(all_logits, all_labels)
            
            # Log detailed metrics
            logger.info("=" * 40)
            logger.info("EVALUATION RESULTS")
            logger.info(
                "%s on %s: Accuracy=%.4f, F1=%.4f, Precision=%.4f, Recall=%.4f",
                model_name,
                split,
                metrics["accuracy"],
                metrics["f1"],
                metrics["precision"],
                metrics["recall"],
            )
            
            # For the 3 classes, print per-class metrics
            for label in ["entailment", "neutral", "contradiction"]:
                logger.info(
                    "%s on %s: Precision_%s=%.4f, Recall_%s=%.4f, F1_%s=%.4f",
                    model_name,
                    split,
                    label,
                    metrics[f"precision_{label}"],
                    label,
                    metrics[f"recall_{label}"],
                    label,
                    metrics[f"f1_{label}"],
                )
            logger.info("=" * 40)
            
            return metrics
        except Exception as e:
            logger.error(f"❌ ERROR EVALUATING {model_name} on {split}: {str(e)}")
            logger.debug("Stack trace:", exc_info=True)
            return {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0, "error": str(e)}
    except Exception as e:
        logger.error(f"❌ ERROR EVALUATING {model_name} on {split}: {str(e)}")
        logger.debug("Stack trace:", exc_info=True)
        return {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0, "error": str(e)}


def benchmark_models(
    models_dir: str,
    output_dir: str,
    splits: List[str] = ["validation", "test_lay", "test_expert"],
    batch_size: int = 16,
    max_length: int = 128,
    dataset_name: str = "afaji/indonli",
    seed: int = 42,
    recursive: bool = True,
    debug_mode: bool = False,
):
    """
    Benchmark multiple models on multiple dataset splits and collect results.
    
    Args:
        models_dir: Directory containing model checkpoints
        output_dir: Directory to save results to
        splits: Dataset splits to evaluate on
        batch_size: Batch size for evaluation
        max_length: Maximum sequence length
        dataset_name: HuggingFace dataset name
        seed: Random seed
        recursive: Whether to search for models recursively
        debug_mode: Enable debug mode for model evaluation
        
    Returns:
        Nested dictionary of results: {model_name: {split: {metric: value}}}
    """
    # Set random seed for reproducibility
    set_seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find model checkpoints
    checkpoints = find_model_checkpoints(models_dir, recursive=recursive)
    if not checkpoints:
        logger.error("No model checkpoints found in %s", models_dir)
        return {}
    
    # Results container
    results = {}
    
    # Evaluate each model on each split
    for model_name, model_path in checkpoints:
        results[model_name] = {}
        
        for split in splits:
            try:
                metrics = evaluate_model(
                    model_path=model_path,
                    model_name=model_name,
                    split=split,
                    batch_size=batch_size,
                    max_length=max_length,
                    dataset_name=dataset_name,
                    seed=seed,
                    debug=debug_mode,
                )
                
                results[model_name][split] = metrics
                
                # Save individual result
                split_dir = os.path.join(output_dir, split)
                os.makedirs(split_dir, exist_ok=True)
                
                with open(os.path.join(split_dir, f"{model_name}.json"), "w") as f:
                    json.dump(metrics, f, indent=2)
            
            except Exception as e:
                logger.error("Error evaluating %s on %s: %s", model_name, split, str(e))
                results[model_name][split] = {
                    "accuracy": 0.0,
                    "f1": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "error": str(e),
                }
    
    # Save combined results
    with open(os.path.join(output_dir, "all_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Create summary tables
    create_summary_tables(results, output_dir)
    
    return results


def create_summary_tables(
    results: Dict[str, Dict[str, Dict[str, float]]], output_dir: str
) -> None:
    """
    Create summary tables from benchmarking results.
    
    Args:
        results: Nested dictionary of benchmarking results
        output_dir: Directory to save summary tables
    """
    # Create pandas DataFrames for each metric
    metrics = ["accuracy", "f1", "precision", "recall"]
    dfs = {}
    
    for metric in metrics:
        data = []
        for model_name, model_results in results.items():
            row = {"model": model_name}
            for split, split_results in model_results.items():
                row[split] = split_results.get(metric, 0.0)
            data.append(row)
        
        if data:
            df = pd.DataFrame(data)
            df = df.sort_values("model")
            dfs[metric] = df
    
    # Save to CSV
    for metric, df in dfs.items():
        df.to_csv(os.path.join(output_dir, f"{metric}_summary.csv"), index=False)
    
    # Create a single combined CSV with the best metric per split
    combined_data = []
    for model_name, model_results in results.items():
        row = {"model": model_name}
        for split, split_results in model_results.items():
            for metric in metrics:
                row[f"{split}_{metric}"] = split_results.get(metric, 0.0)
        combined_data.append(row)
    
    if combined_data:
        combined_df = pd.DataFrame(combined_data)
        combined_df = combined_df.sort_values("model")
        combined_df.to_csv(os.path.join(output_dir, "combined_summary.csv"), index=False)
    
    # Create a rich table for each split and print to console
    logger.info("Summary of results:")
    
    splits = list(next(iter(results.values())).keys()) if results else []
    
    for split in splits:
        table = Table(title=f"Results for {split}")
        table.add_column("Model", justify="left", style="cyan")
        
        for metric in metrics:
            table.add_column(metric.capitalize(), justify="right", style="green")
        
        for model_name, model_results in sorted(results.items()):
            split_results = model_results.get(split, {})
            row = [model_name]
            
            for metric in metrics:
                value = split_results.get(metric, 0.0)
                row.append(f"{value:.4f}")
            
            table.add_row(*row)
        
        console.print(table)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark NLI models")
    
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models",
        help="Directory containing model checkpoints",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports/benchmark",
        help="Directory to save results",
    )
    
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["validation", "test_lay", "test_expert"],
        help="Dataset splits to evaluate on",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation",
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length",
    )
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="afaji/indonli",
        help="HuggingFace dataset name",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    parser.add_argument(
        "--no_recursive",
        action="store_true",
        help="Disable recursive search for models",
    )
    
    parser.add_argument(
        "--debug_discovery",
        action="store_true",
        help="Debug checkpoint discovery without evaluation",
    )
    
    parser.add_argument(
        "--debug_mode",
        action="store_true",
        help="Enable debug mode for model evaluation",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main function."""
    args = parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    
    logger.info("Starting benchmark")
    logger.info("Models directory: %s", args.models_dir)
    logger.info("Output directory: %s", output_dir)
    logger.info("Dataset splits: %s", args.splits)
    
    if args.debug_discovery:
        # Just find checkpoints without evaluating
        logger.info("Running in discovery debug mode - no evaluations will be performed")
        logger.info("Searching for model checkpoints in %s", args.models_dir)
        checkpoints = find_model_checkpoints(args.models_dir)
        
        # List directory structure for each model
        for model_name, model_path in checkpoints:
            logger.info("Found checkpoint: %s at %s", model_name, model_path)
            logger.info("Contents: %s", list(model_path.glob("*")))
            
            # Check for tokenizer files
            tokenizer_files = list(model_path.glob("tokenizer*")) + list(model_path.glob("vocab*"))
            if tokenizer_files:
                logger.info("Tokenizer files: %s", tokenizer_files)
            else:
                logger.info("No tokenizer files found")
                
            # Check for model files
            model_files = list(model_path.glob("pytorch_model*")) + list(model_path.glob("config*"))
            if model_files:
                logger.info("Model files: %s", model_files)
            else:
                logger.info("No model files found")
        
        logger.info("Discovery debug complete. Found %d checkpoints.", len(checkpoints))
        return
    
    results = benchmark_models(
        models_dir=args.models_dir,
        output_dir=output_dir,
        splits=args.splits,
        batch_size=args.batch_size,
        max_length=args.max_length,
        dataset_name=args.dataset_name,
        seed=args.seed,
        recursive=not args.no_recursive,
        debug_mode=args.debug_mode,
    )
    
    # Create a README file with summary information
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(f"# Model Benchmarking Results\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Configuration\n\n")
        f.write(f"- Models directory: `{args.models_dir}`\n")
        f.write(f"- Dataset splits: {', '.join(args.splits)}\n")
        f.write(f"- Batch size: {args.batch_size}\n")
        f.write(f"- Max sequence length: {args.max_length}\n")
        f.write(f"- Dataset: {args.dataset_name}\n")
        f.write(f"- Random seed: {args.seed}\n\n")
        
        f.write(f"## Results Summary\n\n")
        f.write(f"See CSV files in this directory for detailed results.\n\n")
        
        f.write(f"## Best Models\n\n")
        
        # Calculate best model for each split and metric
        if results:
            for split in args.splits:
                f.write(f"### {split.capitalize()}\n\n")
                
                for metric in ["accuracy", "f1", "precision", "recall"]:
                    try:
                        best_model = max(
                            results.items(),
                            key=lambda x: x[1].get(split, {}).get(metric, 0.0),
                        )
                        best_value = best_model[1].get(split, {}).get(metric, 0.0)
                        
                        f.write(f"- Best {metric}: **{best_model[0]}** ({best_value:.4f})\n")
                    except:
                        f.write(f"- Best {metric}: No valid results\n")
                
                f.write("\n")
    
    logger.info("Benchmarking complete. Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
