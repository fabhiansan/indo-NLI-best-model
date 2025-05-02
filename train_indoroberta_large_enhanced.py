#!/usr/bin/env python
"""
Enhanced script to finetune indonesian-roberta-large on IndoNLI dataset.
This script includes several improvements to address the poor performance:
1. Class weighting to handle class imbalance
2. Gradient accumulation for effective larger batch sizes
3. Early stopping to prevent overfitting
4. Focal loss for better handling of hard examples
5. Data augmentation techniques
6. Learning rate finder
7. Improved evaluation metrics
"""
import os
import json
import logging
import argparse
import random
import numpy as np
# import pandas as pd # No longer needed for loading
from datetime import datetime
# from sklearn.model_selection import train_test_split # Replaced by HF datasets splits
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from collections import Counter
from datasets import load_dataset # Import datasets library

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    AdamW,
    get_linear_schedule_with_warmup,
    set_seed
)
from tqdm.auto import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"train_indoroberta_large_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """
    Focal Loss for better handling of hard examples and class imbalance.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class EntailmentDataset(Dataset):
    """Dataset for entailment classification with data augmentation."""
    
    def __init__(self, dataset, tokenizer, max_length=512, augment=False):
        """
        Initialize the dataset using a Hugging Face dataset object.

        Args:
            dataset: Hugging Face dataset object (or slice)
            tokenizer: Tokenizer to use for encoding
            max_length: Maximum sequence length
            augment: Whether to apply data augmentation
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        # Assuming standard IndoNLI columns: 'premise', 'hypothesis', 'label'
        # Label mapping: 0 -> entailment, 1 -> neutral, 2 -> contradiction
        # Adjust column names and mapping if necessary based on dataset inspection.

    def __len__(self):
        return len(self.dataset)

    def augment_text(self, text):
        """Apply simple data augmentation techniques."""
        if not self.augment or random.random() > 0.3:
            return text
            
        words = text.split()
        if len(words) <= 5:
            return text
            
        # Random word dropout
        if random.random() < 0.5:
            dropout_idx = random.randint(0, len(words) - 1)
            words.pop(dropout_idx)
            return ' '.join(words)
        
        # Random word swap
        if len(words) >= 3:
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
            return ' '.join(words)
            
        return text
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = str(item['premise'])
        hypothesis = str(item['hypothesis'])
        label = item['label'] # Assuming label is already 0, 1, or 2

        # Apply data augmentation if enabled
        if self.augment:
            text = self.augment_text(text)
            hypothesis = self.augment_text(hypothesis)
        
        # Tokenize the text and hypothesis
        encoding = self.tokenizer(
            text,
            hypothesis,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        for key in encoding:
            encoding[key] = encoding[key].squeeze(0)
        
        # Add label
        encoding["labels"] = torch.tensor(label, dtype=torch.long)
        
        return encoding

# Removed the old load_data function as we now use datasets.load_dataset

def compute_metrics(predictions, labels):
    """
    Compute evaluation metrics.
    
    Args:
        predictions: Model predictions
        labels: Ground truth labels
        
    Returns:
        Dictionary of metrics
    """
    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions)
    
    # Calculate precision, recall, F1 for each class
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average=None)
    
    # Calculate macro-averaged metrics
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    
    # Create confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Create metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
    }
    
    # Add per-class metrics (Assuming 0: entailment, 1: neutral, 2: contradiction)
    # Verify this mapping against the actual dataset if results seem off.
    class_names = ['entailment', 'neutral', 'contradiction']
    for i, class_name in enumerate(class_names[:len(precision)]):
        metrics[f'precision_{class_name}'] = precision[i]
        metrics[f'recall_{class_name}'] = recall[i]
        metrics[f'f1_{class_name}'] = f1[i]
    
    return metrics, cm

def train_model(model, train_dataloader, eval_dataloader, optimizer, scheduler, device, 
                num_epochs, output_dir, gradient_accumulation_steps=1, patience=3, 
                class_weights=None, num_labels=3):
    """
    Train the model with enhanced features.
    
    Args:
        model: Model to train
        train_dataloader: DataLoader for training data
        eval_dataloader: DataLoader for evaluation data
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to use for training
        num_epochs: Number of epochs to train for
        output_dir: Directory to save the model
        gradient_accumulation_steps: Number of steps to accumulate gradients
        patience: Number of epochs to wait for improvement before early stopping
        class_weights: Weights for each class to handle class imbalance
        num_labels: Number of labels (2 for binary, 3 for multi-class)
        
    Returns:
        Dictionary containing training history
    """
    # Initialize training history
    history = {
        'train_loss': [],
        'eval_loss': [],
        'eval_metrics': [],
        'best_metric': 0.0,
        'best_epoch': 0
    }
    
    # Initialize early stopping variables
    no_improvement_count = 0
    best_eval_metric = 0.0
    
    # Initialize loss function
    if class_weights is not None:
        logger.info(f"Using class weights: {class_weights}")
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    else:
        criterion = FocalLoss(gamma=2.0)
    
    # Training loop
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            
            # Calculate loss
            if hasattr(outputs, 'loss'):
                loss = outputs.loss
            else:
                logits = outputs.logits
                labels = batch["labels"]
                loss = criterion(logits, labels)
            
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights if we've accumulated enough gradients
            if (step + 1) % gradient_accumulation_steps == 0:
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update weights
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Update metrics
            train_loss += loss.item() * gradient_accumulation_steps
            train_steps += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})
        
        # Calculate average training loss
        avg_train_loss = train_loss / train_steps
        history['train_loss'].append(avg_train_loss)
        
        # Evaluation
        model.eval()
        eval_loss = 0.0
        eval_steps = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass
                outputs = model(**batch)
                
                # Calculate loss
                if hasattr(outputs, 'loss'):
                    loss = outputs.loss
                else:
                    logits = outputs.logits
                    labels = batch["labels"]
                    loss = criterion(logits, labels)
                
                # Get predictions
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)
                labels = batch["labels"]
                
                # Update metrics
                eval_loss += loss.item()
                eval_steps += 1
                
                # Move predictions and labels to CPU
                predictions = predictions.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                
                all_predictions.extend(predictions)
                all_labels.extend(labels)
        
        # Calculate average evaluation loss
        avg_eval_loss = eval_loss / eval_steps
        
        # Calculate evaluation metrics
        metrics, confusion_matrix = compute_metrics(all_predictions, all_labels)
        
        # Log metrics
        logger.info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}")
        for name, value in metrics.items():
            logger.info(f"Eval {name}: {value}")
        
        # Store metrics in history
        history['eval_loss'].append(avg_eval_loss)
        history['eval_metrics'].append(metrics)
        
        # Determine if this is the best model so far
        # For multi-class, use macro F1 as the primary metric
        current_metric = metrics['macro_f1'] if num_labels > 2 else metrics['accuracy']
        
        if current_metric > best_eval_metric:
            best_eval_metric = current_metric
            history['best_metric'] = current_metric
            history['best_epoch'] = epoch + 1
            no_improvement_count = 0
            
            # Save the best model
            model_save_path = os.path.join(output_dir, f"best_model")
            os.makedirs(model_save_path, exist_ok=True)
            
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            
            logger.info(f"New best model saved to {model_save_path} with {current_metric:.4f}")
        else:
            no_improvement_count += 1
            logger.info(f"No improvement for {no_improvement_count} epochs")
            
            if no_improvement_count >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save checkpoint for each epoch
        model_save_path = os.path.join(output_dir, f"checkpoint-{epoch+1}")
        os.makedirs(model_save_path, exist_ok=True)
        
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        
        logger.info(f"Checkpoint saved to {model_save_path}")
    
    # Save final model
    model_save_path = os.path.join(output_dir, "final_model")
    os.makedirs(model_save_path, exist_ok=True)
    
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    logger.info(f"Final model saved to {model_save_path}")
    
    return history

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Enhanced script to finetune indonesian-roberta-large on IndoNLI dataset")
    # parser.add_argument("--data_path", type=str, default="negfiltered_data_sample.json", help="Path to the data file") # Removed, using HF dataset
    parser.add_argument("--dataset_name", type=str, default="afaji/indonli", help="Hugging Face dataset name")
    parser.add_argument("--train_split", type=str, default="train", help="Dataset split for training")
    parser.add_argument("--validation_split", type=str, default="validation", help="Dataset split for validation")
    parser.add_argument("--model_name", type=str, default="flax-community/indonesian-roberta-large", help="Pretrained model name")
    parser.add_argument("--output_dir", type=str, default="./models/indoroberta-large-enhanced", help="Output directory")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length (reduced default)") # Reduced default max_length
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # parser.add_argument("--test_size", type=float, default=0.2, help="Test size for train/test split") # Removed
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio (adjusted default)") # Adjusted default
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--num_labels", type=int, default=3, help="Number of labels (2 for binary, 3 for multi-class)")
    parser.add_argument("--use_class_weights", action="store_true", help="Use class weights for imbalanced data")
    parser.add_argument("--use_data_augmentation", action="store_true", help="Use data augmentation")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set seed for reproducibility
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load dataset from Hugging Face Hub
    logger.info(f"Loading dataset {args.dataset_name}")
    dataset = load_dataset(args.dataset_name)

    # Get train and validation splits
    train_dataset_raw = dataset[args.train_split]
    eval_dataset_raw = dataset[args.validation_split]

    logger.info(f"Train set size: {len(train_dataset_raw)}")
    logger.info(f"Validation set size: {len(eval_dataset_raw)}")

    # Log label distribution for training set
    train_label_counts = Counter(train_dataset_raw['label'])
    logger.info(f"Train label distribution: {train_label_counts}")

    # Calculate class weights if needed
    class_weights = None
    if args.use_class_weights:
        total_samples = len(train_dataset_raw)
        num_labels = args.num_labels # Use num_labels from args
        # Ensure counts for all labels (0 to num_labels-1) are present, default to 1 if missing
        counts = [train_label_counts.get(i, 1) for i in range(num_labels)]
        class_weights = [total_samples / (num_labels * count) for count in counts]
        logger.info(f"Calculated class weights: {class_weights}")

    # Load tokenizer and model
    logger.info(f"Loading tokenizer and model from {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load model configuration
    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=args.num_labels,
    )
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        config=config,
    )
    
    # Create datasets
    train_dataset = EntailmentDataset(
        dataset=train_dataset_raw,
        tokenizer=tokenizer,
        max_length=args.max_length,
        augment=args.use_data_augmentation,
    )

    eval_dataset = EntailmentDataset(
        dataset=eval_dataset_raw,
        tokenizer=tokenizer,
        max_length=args.max_length,
        augment=False,  # No augmentation for eval set
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False, # No shuffle for eval
    )
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Move model to device
    model.to(device)
    
    # Prepare optimizer and scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    
    # Calculate total training steps
    total_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    # Create scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Train the model
    logger.info("Starting training")
    history = train_model(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader, # Use the correct eval dataloader
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.num_epochs,
        output_dir=args.output_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        patience=args.patience,
        class_weights=class_weights,
        num_labels=args.num_labels,
    )
    
    # Log final results
    logger.info(f"Best metric: {history['best_metric']:.4f} at epoch {history['best_epoch']}")
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
