#!/usr/bin/env python
"""
Direct script to train the indonesian-roberta-large model on the IndoNLI dataset.
This script includes all necessary imports and doesn't rely on the src module structure.
"""
import os
import sys
import argparse
import logging
import random
from datetime import datetime

import numpy as np
import torch
import yaml
from transformers import (
    AutoConfig, 
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    set_seed
)
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Label mapping for the IndoNLI dataset
INDONLI_LABELS = {"entailment": 0, "neutral": 1, "contradiction": 2}

class IndoNLIDataset(Dataset):
    """Dataset class for the IndoNLI dataset."""
    
    def __init__(
        self,
        tokenizer,
        split="train",
        max_length=128,
        dataset_name="afaji/indonli",
    ):
        """
        Initialize the IndoNLI dataset.
        
        Args:
            tokenizer: Tokenizer to use for encoding the text
            split: Dataset split to use (train, validation, test_lay, test_expert)
            max_length: Maximum sequence length for tokenization
            dataset_name: Name of the dataset on Hugging Face Hub
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(f"Loading IndoNLI dataset ({split} split)...")
        self.dataset = load_dataset(dataset_name, split=split)
        logger.info(f"Loaded {len(self.dataset)} examples from {split} split")
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        
        # Tokenize the premise and hypothesis
        encoding = self.tokenizer(
            example["premise"],
            example["hypothesis"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Remove batch dimension added by the tokenizer
        for key in encoding:
            encoding[key] = encoding[key].squeeze(0)
        
        # Add label - handle both string and integer labels
        if isinstance(example["label"], str):
            encoding["labels"] = INDONLI_LABELS[example["label"]]
        else:
            # If the label is already an integer, use it directly
            encoding["labels"] = example["label"]
        
        return encoding

class DataCollatorForNLI:
    """
    Data collator for NLI tasks. Handles tokenization and padding.
    """
    
    def __init__(self, tokenizer, max_length=128, padding="max_length", truncation=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
    
    def __call__(self, features):
        """Collate examples for training or evaluation."""
        batch = {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "labels": torch.tensor([f["labels"] for f in features], dtype=torch.long),
        }
        
        if "token_type_ids" in features[0]:
            batch["token_type_ids"] = torch.stack([f["token_type_ids"] for f in features])
            
        return batch

def get_nli_dataloader(
    tokenizer,
    split,
    batch_size,
    max_length=128,
    dataset_name="afaji/indonli",
    num_workers=4,
    shuffle=None,
):
    """
    Get a dataloader for the IndoNLI dataset.
    
    Args:
        tokenizer: Tokenizer to use for encoding the text
        split: Dataset split to use (train, validation, test_lay, test_expert)
        batch_size: Batch size for the dataloader
        max_length: Maximum sequence length for tokenization
        dataset_name: Name of the dataset on Hugging Face Hub
        num_workers: Number of workers for the dataloader
        shuffle: Whether to shuffle the dataset (default: True for train, False otherwise)
        
    Returns:
        DataLoader for the specified split
    """
    if shuffle is None:
        shuffle = split == "train"
    
    dataset = IndoNLIDataset(
        tokenizer=tokenizer,
        split=split,
        max_length=max_length,
        dataset_name=dataset_name,
    )
    
    data_collator = DataCollatorForNLI(
        tokenizer=tokenizer,
        max_length=max_length,
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=data_collator,
        num_workers=num_workers,
    )
    
    return dataloader

class NLITrainer:
    """Trainer for NLI models."""
    
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        config,
        device=None,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            train_dataloader: DataLoader for training data
            eval_dataloader: DataLoader for evaluation data
            config: Configuration dictionary
            device: Device to use for training
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config
        
        # Set device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.model.to(self.device)
        
        # Training parameters
        training_config = config["training"]
        self.num_epochs = training_config["num_epochs"]
        self.learning_rate = training_config["learning_rate"]
        self.weight_decay = training_config["weight_decay"]
        self.warmup_ratio = training_config["warmup_ratio"]
        self.gradient_accumulation_steps = training_config["gradient_accumulation_steps"]
        self.fp16 = training_config.get("fp16", False)
        self.disable_tqdm = training_config.get("disable_tqdm", False)
        
        # Output directories
        self.output_dir = config["output"]["output_dir"]
        
        # Initialize training history
        self.training_history = {
            "train_loss": [],
            "eval_metrics": [],
        }
        
        # Initialize best metric
        self.best_eval_metric = {
            "accuracy": 0.0,
            "epoch": 0,
            "step": 0,
        }
        
        # Initialize step counter
        self.global_step = 0
        
        # Prepare optimizer and scheduler
        self._prepare_optimizer_and_scheduler()
    
    def _prepare_optimizer_and_scheduler(self):
        """Prepare optimizer and scheduler."""
        # Prepare optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        # Ensure learning_rate is a float
        learning_rate = float(self.learning_rate)
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
        
        # Prepare scheduler
        total_steps = len(self.train_dataloader) * self.num_epochs // self.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
    
    def train(self):
        """
        Train the model.
        
        Returns:
            Dictionary containing training history and best metrics
        """
        logger.info("Starting training")
        
        self.model.train()
        
        # Setup for mixed precision training
        scaler = None
        if self.fp16:
            scaler = GradScaler()
        
        # Training loop
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch+1}/{self.num_epochs}")
            
            epoch_loss = 0.0
            epoch_steps = 0
            
            progress_bar = tqdm(self.train_dataloader, disable=self.disable_tqdm)
            
            for step, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                if self.fp16:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                else:
                    outputs = self.model(**batch)
                    loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                if self.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Update weights if we've accumulated enough gradients
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    if self.fp16:
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                
                # Update progress bar
                epoch_loss += loss.item() * self.gradient_accumulation_steps
                epoch_steps += 1
                progress_bar.set_description(f"Loss: {loss.item():.4f}")
            
            # End of epoch
            avg_loss = epoch_loss / epoch_steps
            logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
            
            # Evaluate at the end of each epoch
            metrics = self.evaluate()
            self.training_history["eval_metrics"].append({"step": self.global_step, "epoch": epoch+1, **metrics})
            
            # Save model at the end of each epoch
            self.save_model(os.path.join(self.output_dir, f"epoch-{epoch+1}"))
        
        # Save final model
        self.save_model(os.path.join(self.output_dir, "final"))
        
        logger.info("Training completed!")
        
        return {
            "training_history": self.training_history,
            "best_eval_metric": self.best_eval_metric,
        }
    
    def evaluate(self):
        """
        Evaluate the model on the evaluation dataset.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Running evaluation")
        
        self.model.eval()
        
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, disable=self.disable_tqdm):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Get labels
                labels = batch.pop("labels")
                
                # Forward pass
                outputs = self.model(**batch)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
                
                # Move to CPU
                logits = logits.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                
                all_logits.append(logits)
                all_labels.append(labels)
        
        # Concatenate all logits and labels
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Compute metrics
        metrics = compute_metrics(all_logits, all_labels)
        
        # Log metrics
        for key, value in metrics.items():
            logger.info(f"Eval {key}: {value}")
        
        # Update best metric
        if metrics["accuracy"] > self.best_eval_metric["accuracy"]:
            self.best_eval_metric = {
                "accuracy": metrics["accuracy"],
                "epoch": self.training_history["eval_metrics"][-1]["epoch"] if self.training_history["eval_metrics"] else 0,
                "step": self.global_step,
            }
            
            # Save best model
            self.save_model(os.path.join(self.output_dir, "best"))
        
        self.model.train()
        
        return metrics
    
    def save_model(self, output_dir):
        """
        Save the model to the specified directory.
        
        Args:
            output_dir: Directory to save the model to
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(output_dir)
        
        # Save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config["model"]["pretrained_model_name"])
        tokenizer.save_pretrained(output_dir)
        
        # Save training arguments
        with open(os.path.join(output_dir, "training_args.txt"), "w") as f:
            for key, value in self.config.items():
                f.write(f"{key}: {value}\n")
        
        logger.info(f"Model saved to {output_dir}")

def compute_metrics(logits, labels):
    """
    Compute evaluation metrics.
    
    Args:
        logits: Model logits
        labels: Ground truth labels
        
    Returns:
        Dictionary containing metrics
    """
    predictions = np.argmax(logits, axis=1)
    
    # Compute accuracy
    accuracy = (predictions == labels).mean()
    
    return {
        "accuracy": float(accuracy),
    }

def load_config(config_path):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config

def main():
    """Main training function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train indonesian-roberta-large on IndoNLI")
    parser.add_argument("--config", type=str, default="configs/indo_roberta_large.yaml", help="Path to the configuration file")
    parser.add_argument("--output_dir", type=str, help="Output directory (overrides config)")
    parser.add_argument("--seed", type=int, help="Random seed (overrides config)")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs (overrides config)")
    parser.add_argument("--batch_size", type=int, help="Batch size (overrides config)")
    parser.add_argument("--learning_rate", type=float, help="Learning rate (overrides config)")
    args = parser.parse_args()
    
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
    
    # Setup logging to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config["output"]["logging_dir"], f"train_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
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
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model
    logger.info("Creating model")
    model_config = AutoConfig.from_pretrained(
        config["model"]["pretrained_model_name"],
        num_labels=3,
        output_hidden_states=config["model"].get("output_hidden_states", False),
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model"]["pretrained_model_name"],
        config=model_config,
    )
    model.to(device)
    
    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["pretrained_model_name"])
    
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
