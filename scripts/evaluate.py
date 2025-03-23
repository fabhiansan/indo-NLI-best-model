#!/usr/bin/env python
"""
Evaluation script for NLI models.
"""
import argparse
import logging
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from transformers import set_seed

from src.data.dataset import get_nli_dataloader
from src.models.model_factory import ModelFactory
from src.utils.config import load_config
from src.utils.logging import setup_logging, log_system_info
from src.utils.metrics import (
    compute_metrics,
    generate_confusion_matrix,
    generate_classification_report,
    save_metrics_to_csv,
    save_predictions_to_csv,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate NLI models")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--model_name", type=str, help="Name of the model type")
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    parser.add_argument("--test_set", type=str, default="test_lay", 
                        help="Test set to use (train, validation, test_lay, test_expert)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, help="Output directory for evaluation results")
    return parser.parse_args()


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()
    
    # Set up output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.model_path, f"evaluation_{args.test_set}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"eval_{timestamp}.log")
    logger = setup_logging(log_file=log_file)
    
    # Log arguments
    logger.info(f"Arguments: {args}")
    
    # Set seed for reproducibility
    seed = args.seed
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Log system information
    log_system_info(logger)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load configuration if provided
    config = None
    if args.config:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = ModelFactory.from_pretrained(args.model_path, model_name=args.model_name, config=config)
    model.to(device)
    
    # Get tokenizer
    if config:
        tokenizer = model.get_tokenizer(config["model"]["pretrained_model_name"])
    else:
        tokenizer = model.get_tokenizer(args.model_path)
    
    # Load test dataset
    logger.info(f"Loading test dataset: {args.test_set}")
    test_dataloader = get_nli_dataloader(
        tokenizer=tokenizer,
        split=args.test_set,
        batch_size=args.batch_size,
        max_length=128,  # Default max length
        dataset_name="afaji/indonli",
        num_workers=4,
        shuffle=False,
    )
    
    # Evaluate model
    logger.info("Starting evaluation")
    model.eval()
    
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            
            # Get logits and labels
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs[1]
            labels = batch["labels"]
            
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
    
    # Concatenate all logits and labels
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_predictions = np.argmax(all_logits, axis=1)
    
    # Compute metrics
    metrics = compute_metrics(all_logits, all_labels)
    
    # Log metrics
    logger.info("Evaluation results:")
    for key, value in metrics.items():
        logger.info(f"{key}: {value}")
    
    # Generate confusion matrix
    logger.info("Generating confusion matrix")
    confusion_matrix = generate_confusion_matrix(
        predictions=all_predictions,
        labels=all_labels,
        output_dir=output_dir,
        filename="confusion_matrix.png",
    )
    
    # Generate classification report
    logger.info("Generating classification report")
    report = generate_classification_report(
        predictions=all_predictions,
        labels=all_labels,
        output_dir=output_dir,
        filename="classification_report.txt",
    )
    
    # Save metrics to CSV
    logger.info("Saving metrics to CSV")
    save_metrics_to_csv(
        metrics=metrics,
        output_dir=output_dir,
        filename="metrics.csv",
    )
    
    # Save predictions to CSV
    logger.info("Saving predictions to CSV")
    save_predictions_to_csv(
        predictions=all_predictions,
        labels=all_labels,
        output_dir=output_dir,
        filename="predictions.csv",
    )
    
    logger.info(f"Evaluation completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
