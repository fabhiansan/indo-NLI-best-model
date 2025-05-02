#!/usr/bin/env python
"""
Script to finetune indonesian-roberta-large on custom entailment data.
This script uses source_text and generated_indonesian as input, and score (0 or 1) as the label.
"""
import os
import json
import logging
import argparse
import random
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

import torch
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
        logging.FileHandler(f"train_indoroberta_large_entailment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

class EntailmentDataset(Dataset):
    """Dataset for entailment classification."""
    
    def __init__(self, texts, hypotheses, labels, tokenizer, max_length=512):
        """
        Initialize the dataset.
        
        Args:
            texts: List of source texts
            hypotheses: List of generated texts
            labels: List of labels (0 or 1)
            tokenizer: Tokenizer to use for encoding
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.hypotheses = hypotheses
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        hypothesis = str(self.hypotheses[idx])
        label = self.labels[idx]
        
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

def load_data(data_path):
    """
    Load data from JSON file.
    
    Args:
        data_path: Path to the JSON file
        
    Returns:
        texts, hypotheses, labels
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame({
        'id': list(data['id'].values()),
        'source_text': list(data['source_text'].values()),
        'generated_indonesian': list(data['generated_indonesian'].values()),
        'score': list(data['score'].values())
    })
    
    return df['source_text'].tolist(), df['generated_indonesian'].tolist(), df['score'].tolist()

def train_model(model, train_dataloader, eval_dataloader, optimizer, scheduler, device, num_epochs, output_dir):
    """
    Train the model.
    
    Args:
        model: Model to train
        train_dataloader: DataLoader for training data
        eval_dataloader: DataLoader for evaluation data
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to use for training
        num_epochs: Number of epochs to train for
        output_dir: Directory to save the model
        
    Returns:
        Dictionary containing training history
    """
    # Initialize training history
    history = {
        'train_loss': [],
        'eval_loss': [],
        'eval_accuracy': [],
        'best_accuracy': 0.0
    }
    
    # Training loop
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Update metrics
            train_loss += loss.item()
            train_steps += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate average training loss
        avg_train_loss = train_loss / train_steps
        history['train_loss'].append(avg_train_loss)
        
        # Evaluation
        model.eval()
        eval_loss = 0.0
        eval_steps = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss
                
                # Get predictions
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)
                labels = batch["labels"]
                
                # Update metrics
                eval_loss += loss.item()
                eval_steps += 1
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
        
        # Calculate average evaluation loss and accuracy
        avg_eval_loss = eval_loss / eval_steps
        accuracy = correct_predictions / total_predictions
        
        history['eval_loss'].append(avg_eval_loss)
        history['eval_accuracy'].append(accuracy)
        
        logger.info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Save the model if it's the best so far
        if accuracy > history['best_accuracy']:
            history['best_accuracy'] = accuracy
            
            # Save the model
            model_save_path = os.path.join(output_dir, f"best_model")
            os.makedirs(model_save_path, exist_ok=True)
            
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            
            logger.info(f"New best model saved to {model_save_path} with accuracy: {accuracy:.4f}")
        
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
    parser = argparse.ArgumentParser(description="Finetune indonesian-roberta-large on custom entailment data")
    parser.add_argument("--data_path", type=str, default="negfiltered_data_sample.json", help="Path to the data file")
    parser.add_argument("--model_name", type=str, default="flax-community/indonesian-roberta-large", help="Pretrained model name")
    parser.add_argument("--output_dir", type=str, default="./models/indoroberta-large-entailment", help="Output directory")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size for train/test split")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
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
    
    # Load data
    logger.info(f"Loading data from {args.data_path}")
    texts, hypotheses, labels = load_data(args.data_path)
    
    # Split data into train and test sets
    train_texts, test_texts, train_hypotheses, test_hypotheses, train_labels, test_labels = train_test_split(
        texts, hypotheses, labels, test_size=args.test_size, random_state=args.seed
    )
    
    logger.info(f"Train set size: {len(train_texts)}")
    logger.info(f"Test set size: {len(test_texts)}")
    
    # Load tokenizer and model
    logger.info(f"Loading tokenizer and model from {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load model configuration
    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=2,  # Binary classification: 0 or 1
    )
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        config=config,
    )
    
    # Create datasets
    train_dataset = EntailmentDataset(
        texts=train_texts,
        hypotheses=train_hypotheses,
        labels=train_labels,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    
    test_dataset = EntailmentDataset(
        texts=test_texts,
        hypotheses=test_hypotheses,
        labels=test_labels,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
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
    total_steps = len(train_dataloader) * args.num_epochs
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
        eval_dataloader=test_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.num_epochs,
        output_dir=args.output_dir,
    )
    
    # Log final results
    logger.info(f"Best accuracy: {history['best_accuracy']:.4f}")
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
