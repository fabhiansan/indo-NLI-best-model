#!/usr/bin/env python
"""
Script to evaluate the finetuned indonesian-roberta-large model on entailment data.
"""
import os
import json
import logging
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
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

def evaluate_model(model, dataloader, device):
    """
    Evaluate the model.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to use for evaluation
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            
            # Get predictions
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            
            # Move to CPU
            predictions = predictions.detach().cpu().numpy()
            labels = batch["labels"].detach().cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'predictions': all_predictions,
        'labels': all_labels
    }

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Evaluate finetuned indonesian-roberta-large on entailment data")
    parser.add_argument("--data_path", type=str, default="negfiltered_data_sample.json", help="Path to the data file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the finetuned model")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json", help="Path to save evaluation results")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.data_path}")
    texts, hypotheses, labels = load_data(args.data_path)
    
    logger.info(f"Dataset size: {len(texts)}")
    
    # Load tokenizer and model
    logger.info(f"Loading tokenizer and model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    
    # Create dataset
    dataset = EntailmentDataset(
        texts=texts,
        hypotheses=hypotheses,
        labels=labels,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Move model to device
    model.to(device)
    
    # Evaluate model
    logger.info("Evaluating model")
    metrics = evaluate_model(model, dataloader, device)
    
    # Log metrics
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1']:.4f}")
    logger.info(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
    
    # Save results
    if args.output_file:
        # Convert numpy arrays to lists for JSON serialization
        metrics['confusion_matrix'] = metrics['confusion_matrix'].tolist()
        metrics['predictions'] = [int(p) for p in metrics['predictions']]
        metrics['labels'] = [int(l) for l in metrics['labels']]
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
