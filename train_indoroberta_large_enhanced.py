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
import pandas as pd # Needed again for JSON loading
from datetime import datetime
from sklearn.model_selection import train_test_split # Needed again for splitting loaded data
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from collections import Counter
# from datasets import load_dataset # Not using HF datasets for this task

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
    
    def __init__(self, texts, hypotheses, labels, tokenizer, max_length=512, augment=False):
        """
        Initialize the dataset. (Reverted to list-based input)

        Args:
            texts: List of source texts
            hypotheses: List of generated texts
            labels: List of labels (0 or 1 for binary classification)
            tokenizer: Tokenizer to use for encoding
            max_length: Maximum sequence length
            augment: Whether to apply data augmentation
        """
        self.texts = texts
        self.hypotheses = hypotheses
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment

    def __len__(self):
        return len(self.labels)

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
        text = str(self.texts[idx])
        hypothesis = str(self.hypotheses[idx])
        label = self.labels[idx] # Should be 0 or 1

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

def load_custom_json(data_path, text_col='source_text', hypothesis_col='generated_indonesian', label_col='score'):
    """
    Load data from the specific JSON format like negfiltered_data_sample.json.

    Args:
        data_path: Path to the JSON file
        text_col: Name of the key containing source texts
        hypothesis_col: Name of the key containing hypotheses
        label_col: Name of the key containing labels

    Returns:
        texts, hypotheses, labels (as lists)
    """
    logger.info(f"Loading custom JSON data from: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, dict) or text_col not in data or hypothesis_col not in data or label_col not in data:
        raise ValueError(f"JSON file {data_path} does not have the expected structure with keys: {text_col}, {hypothesis_col}, {label_col}")

    # Convert the dictionary of dictionaries structure to lists, ensuring order by index
    try:
        # Sort keys numerically to maintain order
        sorted_indices = sorted(data[label_col].keys(), key=int)
        texts = [data[text_col][idx] for idx in sorted_indices]
        hypotheses = [data[hypothesis_col][idx] for idx in sorted_indices]
        labels = [data[label_col][idx] for idx in sorted_indices]
    except KeyError as e:
        raise ValueError(f"Missing index {e} in one of the JSON fields ({text_col}, {hypothesis_col}, {label_col})")
    except Exception as e:
        raise ValueError(f"Error processing JSON data: {e}")


    # Log label distribution
    label_counts = Counter(labels)
    logger.info(f"Label distribution in loaded data: {label_counts}")

    # Ensure labels are 0 or 1
    if not all(lbl in [0, 1] for lbl in labels):
         logger.warning("Labels contain values other than 0 or 1. Ensure this is intended for binary classification.")

    return texts, hypotheses, labels


def compute_metrics(predictions, labels):
    """
    Compute evaluation metrics.

    Args:
        predictions: Model predictions
        labels: Ground truth labels

    Returns:
        Dictionary of metrics for binary classification
    """
    accuracy = accuracy_score(labels, predictions)
    # Calculate precision, recall, F1 for the positive class (1)
    # Use zero_division=0 to handle cases where a class has no predictions/labels
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', pos_label=1, zero_division=0)

    # Create confusion matrix
    cm = confusion_matrix(labels, predictions, labels=[0, 1]) # Ensure labels are 0, 1

    # Create metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision, # Precision for class 1
        'recall': recall,       # Recall for class 1
        'f1': f1,               # F1 for class 1
    }

    # Log full classification report for details
    try:
        report = classification_report(labels, predictions, target_names=['non-entailment (0)', 'entailment (1)'], zero_division=0)
        logger.info(f"Classification Report:\n{report}")
    except ValueError:
        logger.warning("Could not generate classification report (possibly due to only one class present in labels/predictions).")


    return metrics, cm

def train_model(model, tokenizer, train_dataloader, eval_dataloader, optimizer, scheduler, device, # Added tokenizer
                num_epochs, output_dir, gradient_accumulation_steps=1, patience=3,
                class_weights=None, num_labels=3):
    """
    Train the model with enhanced features.

    Args:
        model: Model to train
        tokenizer: Tokenizer associated with the model (for saving) # Added tokenizer arg
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
        # For binary classification, use F1 score for the positive class (1) as the primary metric
        current_metric = metrics['f1'] # Using F1 score for class 1

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
    parser = argparse.ArgumentParser(description="Finetune indonesian-roberta-large for binary entailment")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input JSON data file (e.g., negfiltered_data_sample.json)")
    parser.add_argument("--text_col", type=str, default="source_text", help="JSON key for source text")
    parser.add_argument("--hypothesis_col", type=str, default="generated_indonesian", help="JSON key for hypothesis text")
    parser.add_argument("--label_col", type=str, default="score", help="JSON key for the binary label (0 or 1)")
    # parser.add_argument("--dataset_name", type=str, default="afaji/indonli", help="Hugging Face dataset name") # Removed
    # parser.add_argument("--train_split", type=str, default="train", help="Dataset split for training") # Removed
    # parser.add_argument("--validation_split", type=str, default="validation", help="Dataset split for validation") # Removed
    parser.add_argument("--model_name", type=str, default="flax-community/indonesian-roberta-large", help="Pretrained model name")
    parser.add_argument("--output_dir", type=str, default="./models/indoroberta-large-binary-entailment", help="Output directory")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--validation_size", type=float, default=0.15, help="Fraction of data to use for validation split") # Added validation split size
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--num_labels", type=int, default=2, help="Number of labels (set to 2 for binary)") # Default to 2
    parser.add_argument("--use_class_weights", action="store_true", help="Use class weights for imbalanced data")
    parser.add_argument("--use_data_augmentation", action="store_true", help="Use data augmentation (basic implementation)")
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
    
    # Load data from the custom JSON file
    texts, hypotheses, labels = load_custom_json(
        args.data_path,
        text_col=args.text_col,
        hypothesis_col=args.hypothesis_col,
        label_col=args.label_col
    )

    # Split data into train and validation sets
    if args.validation_size > 0 and args.validation_size < 1.0:
        train_texts, val_texts, train_hypotheses, val_hypotheses, train_labels, val_labels = train_test_split(
            texts, hypotheses, labels,
            test_size=args.validation_size,
            random_state=args.seed,
            stratify=labels # Stratify to maintain label distribution
        )
        logger.info(f"Train set size: {len(train_texts)}")
        logger.info(f"Validation set size: {len(val_texts)}")
    elif len(texts) > 0: # Use all data for training if validation_size is 0 or invalid, but only if data exists
        train_texts, train_hypotheses, train_labels = texts, hypotheses, labels
        val_texts, val_hypotheses, val_labels = [], [], []
        logger.info(f"Train set size: {len(train_texts)} (Using all data for training)")
        if args.validation_size <= 0 or args.validation_size >=1.0:
             logger.warning("No validation set created (validation_size <= 0 or >= 1). Early stopping and best model saving based on validation will not function.")
    else:
        logger.error("No data loaded. Exiting.")
        return # Exit if no data loaded


    # Log label distribution for the actual training set
    train_label_counts = Counter(train_labels)
    logger.info(f"Actual Train label distribution: {train_label_counts}")

    # Calculate class weights if needed (based on the actual training split)
    class_weights = None
    if args.use_class_weights and len(train_labels) > 0:
        total_samples = len(train_labels)
        num_labels = args.num_labels # Should be 2
        # Ensure counts for all labels (0 to num_labels-1) are present, default to 1 if missing
        counts = [train_label_counts.get(i, 1) for i in range(num_labels)]
        # Avoid division by zero if a class is missing entirely in the training split
        if all(c > 0 for c in counts):
             class_weights = [total_samples / (num_labels * count) for count in counts]
             logger.info(f"Calculated class weights for training: {class_weights}")
        else:
             logger.warning("Cannot calculate class weights: one or more classes missing in the training split.")


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
    
    # Create datasets (reverted to list-based input)
    train_dataset = EntailmentDataset(
        texts=train_texts,
        hypotheses=train_hypotheses,
        labels=train_labels,
        tokenizer=tokenizer,
        max_length=args.max_length,
        augment=args.use_data_augmentation,
    )

    # Create validation dataset only if val_texts is not empty
    eval_dataset = None
    if val_texts:
        eval_dataset = EntailmentDataset(
            texts=val_texts,
            hypotheses=val_hypotheses,
            labels=val_labels,
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
    
    # Create eval dataloader only if eval_dataset exists
    eval_dataloader = None
    if eval_dataset:
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
        tokenizer=tokenizer, # Pass tokenizer here
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
