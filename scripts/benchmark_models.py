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
    models_path = Path(models_dir)
    checkpoints = []
    
    if not models_path.exists():
        logger.warning("Models directory %s does not exist", models_dir)
        return []
    
    # Find all model directories
    model_dirs = []
    if recursive:
        for path in models_path.glob("**"):
            if path.is_dir() and any((path / subdir).exists() for subdir in ["best", "final", "checkpoint-*"]):
                model_dirs.append(path)
    else:
        model_dirs = [p for p in models_path.iterdir() if p.is_dir()]
        
    # Find checkpoints within each model directory
    for model_dir in model_dirs:
        model_name = model_dir.name
        
        # Check for 'best' checkpoint
        best_dir = model_dir / "best"
        if best_dir.exists():
            checkpoints.append((model_name, best_dir))
        
        # Check for 'final' checkpoint
        final_dir = model_dir / "final"
        if final_dir.exists():
            checkpoints.append((model_name, final_dir))
            
        # Check for epoch checkpoints
        for epoch_dir in model_dir.glob("epoch-*"):
            if epoch_dir.is_dir():
                # Extract epoch number for sorting
                match = re.search(r"epoch-(\d+)", str(epoch_dir))
                if match:
                    epoch_num = int(match.group(1))
                    checkpoints.append((f"{model_name}-epoch-{epoch_num}", epoch_dir))
    
    # Sort by model name
    checkpoints.sort(key=lambda x: x[0])
    
    logger.info("Found %d model checkpoints", len(checkpoints))
    return checkpoints


def evaluate_model(
    model_path: Path,
    model_name: str,
    split: str,
    batch_size: int = 16,
    max_length: int = 128,
    dataset_name: str = "afaji/indonli",
    num_workers: int = 4,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate a model on a specific dataset split.
    
    Args:
        model_path: Path to the model checkpoint
        model_name: Model name (used for logging and results)
        split: Dataset split to evaluate on
        batch_size: Batch size for evaluation
        max_length: Maximum sequence length
        dataset_name: HuggingFace dataset name
        num_workers: Number of workers for data loading
        device: Device to run evaluation on
        
    Returns:
        Dictionary of metrics
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("Evaluating %s on %s...", model_name, split)
    
    # Load the model
    try:
        model = ModelFactory.from_pretrained(str(model_path), model_name=model_name)
        model.to(device)
        model.eval()
        
        # Get tokenizer with robust fallback
        try:
            tokenizer = ModelFactory.get_tokenizer_for_model(str(model_path), model_name)
        except Exception as e:
            logger.warning("Error loading tokenizer via ModelFactory: %s", str(e))
            
            # Direct fallbacks for specific model types
            if "roberta" in model_name.lower():
                logger.info("Falling back to default RoBERTa tokenizer")
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")
            elif "bert" in model_name.lower():
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
        dataloader = get_nli_dataloader(
            tokenizer=tokenizer,
            split=split,
            batch_size=batch_size,
            max_length=max_length,
            dataset_name=dataset_name,
            num_workers=num_workers,
            shuffle=False,
        )
        
        # Evaluation
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {model_name} on {split}"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch.get("token_type_ids", None)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        dummy_logits = np.zeros((len(all_preds), 3))
        for i, pred in enumerate(all_preds):
            dummy_logits[i, pred] = 1
        
        metrics = compute_metrics(dummy_logits, all_labels)
        logger.info(
            "%s on %s: Accuracy=%.4f, F1=%.4f, Precision=%.4f, Recall=%.4f",
            model_name,
            split,
            metrics["accuracy"],
            metrics["f1"],
            metrics["precision"],
            metrics["recall"],
        )
        
        return metrics
    
    except Exception as e:
        logger.error("Error evaluating %s on %s: %s", model_name, split, str(e))
        return {
            "accuracy": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "error": str(e),
        }


def benchmark_models(
    models_dir: str,
    output_dir: str,
    splits: List[str] = ["validation", "test_lay", "test_expert"],
    batch_size: int = 16,
    max_length: int = 128,
    dataset_name: str = "afaji/indonli",
    seed: int = 42,
    recursive: bool = True,
) -> Dict[str, Dict[str, Dict[str, float]]]:
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
    
    results = benchmark_models(
        models_dir=args.models_dir,
        output_dir=output_dir,
        splits=args.splits,
        batch_size=args.batch_size,
        max_length=args.max_length,
        dataset_name=args.dataset_name,
        seed=args.seed,
        recursive=not args.no_recursive,
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
