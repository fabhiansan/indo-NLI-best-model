"""
Evaluator for NLI models.
"""
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.models.base_model import BaseNLIModel
from src.utils.metrics import (
    compute_metrics,
    generate_confusion_matrix,
    generate_classification_report,
    save_metrics_to_csv,
    save_predictions_to_csv,
)

logger = logging.getLogger(__name__)


class NLIEvaluator:
    """Evaluator for NLI models."""
    
    def __init__(
        self,
        model: BaseNLIModel,
        dataloader: DataLoader,
        output_dir: str,
        device: Optional[torch.device] = None,
        disable_tqdm: bool = False,
    ):
        """
        Initialize the evaluator.
        
        Args:
            model: Model to evaluate
            dataloader: DataLoader for evaluation data
            output_dir: Directory to save evaluation results
            device: Device to use for evaluation
            disable_tqdm: Whether to disable the progress bar
        """
        self.model = model
        self.dataloader = dataloader
        self.output_dir = output_dir
        self.disable_tqdm = disable_tqdm
        
        # Set device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.model.to(self.device)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def evaluate(self) -> Dict:
        """
        Evaluate the model.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Starting evaluation")
        
        self.model.eval()
        
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.dataloader, disable=self.disable_tqdm, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                
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
        for key, value in metrics.items():
            logger.info(f"{key}: {value}")
        
        # Generate confusion matrix
        logger.info("Generating confusion matrix")
        confusion_matrix = generate_confusion_matrix(
            predictions=all_predictions,
            labels=all_labels,
            output_dir=self.output_dir,
            filename="confusion_matrix.png",
        )
        
        # Generate classification report
        logger.info("Generating classification report")
        report = generate_classification_report(
            predictions=all_predictions,
            labels=all_labels,
            output_dir=self.output_dir,
            filename="classification_report.txt",
        )
        
        # Save metrics to CSV
        logger.info("Saving metrics to CSV")
        save_metrics_to_csv(
            metrics=metrics,
            output_dir=self.output_dir,
            filename="metrics.csv",
        )
        
        # Save predictions to CSV
        logger.info("Saving predictions to CSV")
        save_predictions_to_csv(
            predictions=all_predictions,
            labels=all_labels,
            output_dir=self.output_dir,
            filename="predictions.csv",
        )
        
        logger.info(f"Evaluation completed. Results saved to {self.output_dir}")
        
        return metrics


def evaluate_on_dataset(
    model: BaseNLIModel,
    dataloader: DataLoader,
    output_dir: str,
    device: Optional[torch.device] = None,
    disable_tqdm: bool = False,
) -> Dict:
    """
    Convenience function to evaluate a model on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation data
        output_dir: Directory to save evaluation results
        device: Device to use for evaluation
        disable_tqdm: Whether to disable the progress bar
        
    Returns:
        Dictionary containing evaluation metrics
    """
    evaluator = NLIEvaluator(
        model=model,
        dataloader=dataloader,
        output_dir=output_dir,
        device=device,
        disable_tqdm=disable_tqdm,
    )
    
    return evaluator.evaluate()
