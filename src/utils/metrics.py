"""
Metrics utilities for model evaluation.
"""
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

logger = logging.getLogger(__name__)

# Define the labels for the IndoNLI dataset
INDONLI_LABEL_LIST = ["entailment", "neutral", "contradiction"]


def compute_metrics(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Compute evaluation metrics for NLI task.
    
    Args:
        logits: Logits from the model
        labels: Ground truth labels
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Get predictions from logits
    predictions = np.argmax(logits, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions)
    
    # Calculate precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    
    # Calculate precision, recall, and F1 score for each class
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0, labels=range(len(INDONLI_LABEL_LIST))
    )
    
    # Create the metrics dictionary
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    
    # Add per-class metrics
    for i, label in enumerate(INDONLI_LABEL_LIST):
        metrics[f"precision_{label}"] = precision_per_class[i]
        metrics[f"recall_{label}"] = recall_per_class[i]
        metrics[f"f1_{label}"] = f1_per_class[i]
    
    return metrics


def generate_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    output_dir: Optional[str] = None,
    filename: str = "confusion_matrix.png",
) -> np.ndarray:
    """
    Generate a confusion matrix for the predictions.
    
    Args:
        predictions: Predicted labels
        labels: Ground truth labels
        output_dir: Directory to save the confusion matrix image
        filename: Filename for the confusion matrix image
        
    Returns:
        Confusion matrix as a numpy array
    """
    # Calculate confusion matrix
    cm = confusion_matrix(labels, predictions, labels=range(len(INDONLI_LABEL_LIST)))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=INDONLI_LABEL_LIST,
        yticklabels=INDONLI_LABEL_LIST,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    
    # Save the plot if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
    
    plt.close()
    
    return cm


def generate_classification_report(
    predictions: np.ndarray,
    labels: np.ndarray,
    output_dir: Optional[str] = None,
    filename: str = "classification_report.txt",
) -> str:
    """
    Generate a classification report for the predictions.
    
    Args:
        predictions: Predicted labels
        labels: Ground truth labels
        output_dir: Directory to save the classification report
        filename: Filename for the classification report
        
    Returns:
        Classification report as a string
    """
    # Calculate classification report
    report = classification_report(
        labels,
        predictions,
        labels=range(len(INDONLI_LABEL_LIST)),
        target_names=INDONLI_LABEL_LIST,
        digits=4,
    )
    
    # Save the report if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, filename), "w") as f:
            f.write(report)
    
    return report


def save_metrics_to_csv(
    metrics: Dict[str, float],
    output_dir: str,
    filename: str = "metrics.csv",
) -> None:
    """
    Save metrics to a CSV file.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        output_dir: Directory to save the CSV file
        filename: Filename for the CSV file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert metrics to DataFrame
    df = pd.DataFrame([metrics])
    
    # Save DataFrame to CSV
    df.to_csv(os.path.join(output_dir, filename), index=False)
    
    logger.info(f"Metrics saved to {os.path.join(output_dir, filename)}")


def save_predictions_to_csv(
    predictions: np.ndarray,
    labels: np.ndarray,
    output_dir: str,
    filename: str = "predictions.csv",
) -> None:
    """
    Save predictions to a CSV file.
    
    Args:
        predictions: Predicted labels
        labels: Ground truth labels
        output_dir: Directory to save the CSV file
        filename: Filename for the CSV file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert predictions and labels to DataFrame
    df = pd.DataFrame({
        "true_label": labels,
        "predicted_label": predictions,
        "true_label_name": [INDONLI_LABEL_LIST[label] for label in labels],
        "predicted_label_name": [INDONLI_LABEL_LIST[pred] for pred in predictions],
    })
    
    # Save DataFrame to CSV
    df.to_csv(os.path.join(output_dir, filename), index=False)
    
    logger.info(f"Predictions saved to {os.path.join(output_dir, filename)}")
