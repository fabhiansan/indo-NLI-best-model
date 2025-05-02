#!/usr/bin/env python
"""
Script to predict entailment using the finetuned indonesian-roberta-large model.
"""
import os
import argparse
import logging
import json
import pandas as pd

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def predict_entailment(model, tokenizer, text, hypothesis, device, max_length=512):
    """
    Predict entailment for a given text and hypothesis.
    
    Args:
        model: Model to use for prediction
        tokenizer: Tokenizer to use for encoding
        text: Source text
        hypothesis: Generated text
        device: Device to use for prediction
        max_length: Maximum sequence length
        
    Returns:
        Dictionary containing prediction and probability
    """
    # Tokenize the text and hypothesis
    inputs = tokenizer(
        text,
        hypothesis,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get prediction and probability
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    prediction = torch.argmax(logits, dim=1).item()
    probability = probabilities[0][prediction].item()
    
    return {
        'prediction': prediction,
        'probability': probability,
        'entailment_probability': probabilities[0][1].item(),
        'non_entailment_probability': probabilities[0][0].item()
    }

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Predict entailment using finetuned indonesian-roberta-large")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the finetuned model")
    parser.add_argument("--input_file", type=str, help="Path to input file (JSON or CSV)")
    parser.add_argument("--output_file", type=str, help="Path to save prediction results")
    parser.add_argument("--text", type=str, help="Source text for prediction")
    parser.add_argument("--hypothesis", type=str, help="Generated text for prediction")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    args = parser.parse_args()
    
    # Check if either input file or text/hypothesis is provided
    if not args.input_file and (not args.text or not args.hypothesis):
        parser.error("Either --input_file or both --text and --hypothesis must be provided")
    
    # Load tokenizer and model
    logger.info(f"Loading tokenizer and model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Move model to device
    model.to(device)
    
    # Process input
    if args.input_file:
        # Load input file
        logger.info(f"Loading input from {args.input_file}")
        
        if args.input_file.endswith('.json'):
            with open(args.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if the JSON has the expected structure
            if isinstance(data, dict) and 'source_text' in data and 'generated_indonesian' in data:
                # Convert to DataFrame
                df = pd.DataFrame({
                    'source_text': list(data['source_text'].values()),
                    'generated_indonesian': list(data['generated_indonesian'].values())
                })
            else:
                # Assume it's a list of dictionaries
                df = pd.DataFrame(data)
        
        elif args.input_file.endswith('.csv'):
            df = pd.read_csv(args.input_file)
        
        else:
            raise ValueError("Input file must be JSON or CSV")
        
        # Check if the DataFrame has the required columns
        required_columns = ['source_text', 'generated_indonesian']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Input file must contain columns: {required_columns}")
        
        # Make predictions
        logger.info("Making predictions")
        results = []
        
        for i, row in df.iterrows():
            text = row['source_text']
            hypothesis = row['generated_indonesian']
            
            prediction = predict_entailment(
                model=model,
                tokenizer=tokenizer,
                text=text,
                hypothesis=hypothesis,
                device=device,
                max_length=args.max_length
            )
            
            # Add to results
            results.append({
                'source_text': text,
                'generated_indonesian': hypothesis,
                'prediction': prediction['prediction'],
                'probability': prediction['probability'],
                'entailment_probability': prediction['entailment_probability'],
                'non_entailment_probability': prediction['non_entailment_probability']
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        if args.output_file:
            if args.output_file.endswith('.json'):
                results_df.to_json(args.output_file, orient='records', indent=2)
            elif args.output_file.endswith('.csv'):
                results_df.to_csv(args.output_file, index=False)
            else:
                results_df.to_csv(args.output_file, index=False)
            
            logger.info(f"Results saved to {args.output_file}")
        
        # Print summary
        logger.info(f"Total predictions: {len(results_df)}")
        logger.info(f"Entailment (1): {results_df['prediction'].sum()}")
        logger.info(f"Non-entailment (0): {len(results_df) - results_df['prediction'].sum()}")
    
    else:
        # Make prediction for a single text/hypothesis pair
        prediction = predict_entailment(
            model=model,
            tokenizer=tokenizer,
            text=args.text,
            hypothesis=args.hypothesis,
            device=device,
            max_length=args.max_length
        )
        
        # Print prediction
        logger.info(f"Source text: {args.text}")
        logger.info(f"Generated text: {args.hypothesis}")
        logger.info(f"Prediction: {'Entailment' if prediction['prediction'] == 1 else 'Non-entailment'}")
        logger.info(f"Probability: {prediction['probability']:.4f}")
        logger.info(f"Entailment probability: {prediction['entailment_probability']:.4f}")
        logger.info(f"Non-entailment probability: {prediction['non_entailment_probability']:.4f}")

if __name__ == "__main__":
    main()
