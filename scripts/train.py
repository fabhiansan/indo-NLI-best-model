#!/usr/bin/env python
"""
Training script for NLI models.
"""
import argparse
import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
from transformers import set_seed

from src.data.dataset import get_nli_dataloader
from src.models.model_factory import ModelFactory
from src.training.trainer import NLITrainer
from src.utils.config import load_config
from src.utils.logging import setup_logging, log_system_info


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train NLI models")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--output_dir", type=str, help="Output directory (overrides config)")
    parser.add_argument("--seed", type=int, help="Random seed (overrides config)")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs (overrides config)")
    parser.add_argument("--batch_size", type=int, help="Batch size (overrides config)")
    parser.add_argument("--learning_rate", type=float, help="Learning rate (overrides config)")
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override configuration with command line arguments
    if args.output_dir:
        config["output"]["output_dir"] = args.output_dir
        config["output"]["logging_dir"] = os.path.join(args.output_dir, "logs")
        config["output"]["report_dir"] = os.path.join(args.output_dir, "reports")
    
    if args.seed:
        config["training"]["seed"] = args.seed
    
    if args.fp16:
        config["training"]["fp16"] = True
    
    if args.num_epochs:
        config["training"]["num_epochs"] = args.num_epochs
    
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    
    # Create output directories
    os.makedirs(config["output"]["output_dir"], exist_ok=True)
    os.makedirs(config["output"]["logging_dir"], exist_ok=True)
    os.makedirs(config["output"]["report_dir"], exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config["output"]["logging_dir"], f"train_{timestamp}.log")
    logger = setup_logging(log_file=log_file)
    
    # Log configuration
    logger.info(f"Configuration: {config}")
    
    # Set seed for reproducibility
    seed = config["training"]["seed"]
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
    
    # Create model
    logger.info("Creating model")
    model = ModelFactory.create_model(config, num_labels=3)
    model.to(device)
    
    # Get tokenizer
    tokenizer = model.get_tokenizer(config["model"]["pretrained_model_name"])
    
    # Load datasets
    logger.info("Loading datasets")
    max_seq_length = config["model"].get("max_seq_length", 128)
    batch_size = config["training"]["batch_size"]
    num_workers = config["data"].get("num_workers", 4)
    dataset_name = config["data"].get("dataset_name", "afaji/indonli")
    
    train_dataloader = get_nli_dataloader(
        tokenizer=tokenizer,
        split=config["data"].get("train_split", "train"),
        batch_size=batch_size,
        max_length=max_seq_length,
        dataset_name=dataset_name,
        num_workers=num_workers,
        shuffle=True,
    )
    
    eval_dataloader = get_nli_dataloader(
        tokenizer=tokenizer,
        split=config["data"].get("validation_split", "validation"),
        batch_size=batch_size,
        max_length=max_seq_length,
        dataset_name=dataset_name,
        num_workers=num_workers,
        shuffle=False,
    )
    
    # Create trainer
    logger.info("Creating trainer")
    trainer = NLITrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        config=config,
        device=device,
    )
    
    # Train the model
    logger.info("Starting training")
    training_results = trainer.train()
    
    logger.info(f"Best evaluation metric: {training_results['best_eval_metric']}")
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
