"""
Trainer for NLI models.
"""
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from src.models.base_model import BaseNLIModel
from src.utils.metrics import compute_metrics

logger = logging.getLogger(__name__)


class NLITrainer:
    """Trainer for NLI models."""
    
    def __init__(
        self,
        model: BaseNLIModel,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        config: Dict,
        device: Optional[torch.device] = None,
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
        
        # Set training parameters
        self.learning_rate = config["training"]["learning_rate"]
        self.num_epochs = config["training"]["num_epochs"]
        self.warmup_ratio = config["training"].get("warmup_ratio", 0.1)
        self.weight_decay = config["training"].get("weight_decay", 0.01)
        self.gradient_accumulation_steps = config["training"].get("gradient_accumulation_steps", 1)
        self.save_steps = config["training"].get("save_steps", 500)
        self.eval_steps = config["training"].get("eval_steps", 500)
        self.logging_steps = config["training"].get("logging_steps", 100)
        self.disable_tqdm = config["training"].get("disable_tqdm", False)
        self.fp16 = config["training"].get("fp16", False)
        
        # Set output directories
        self.output_dir = config["output"]["output_dir"]
        self.logging_dir = config["output"]["logging_dir"]
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logging_dir, exist_ok=True)
        
        # Initialize optimizer and scheduler
        self._init_optimizer_and_scheduler()
        
        # Initialize tracking variables
        self.global_step = 0
        self.best_eval_metric = float("-inf")
        self.training_history = {"loss": [], "eval_metrics": []}
    
    def _init_optimizer_and_scheduler(self):
        """Initialize optimizer and learning rate scheduler."""
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
        
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        
        # Prepare scheduler
        total_steps = len(self.train_dataloader) * self.num_epochs // self.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
    
    def train(self) -> Dict:
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
            from torch.cuda.amp import GradScaler
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
                
                epoch_loss += loss.item() * self.gradient_accumulation_steps
                epoch_steps += 1
                
                # Update weights
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
                    progress_bar.set_description(f"Loss: {loss.item():.4f}")
                    
                    # Logging
                    if self.global_step % self.logging_steps == 0:
                        self.training_history["loss"].append({"step": self.global_step, "loss": epoch_loss / epoch_steps})
                        logger.info(f"Step {self.global_step}: loss = {epoch_loss / epoch_steps:.4f}")
                    
                    # Evaluation
                    if self.global_step % self.eval_steps == 0:
                        metrics = self.evaluate()
                        self.training_history["eval_metrics"].append({"step": self.global_step, **metrics})
                        
                        # Save best model
                        if metrics["accuracy"] > self.best_eval_metric:
                            logger.info(f"New best model! Accuracy: {metrics['accuracy']:.4f}")
                            self.best_eval_metric = metrics["accuracy"]
                            self.save_model(os.path.join(self.output_dir, "best"))
                    
                    # Save checkpoint
                    if self.global_step % self.save_steps == 0:
                        self.save_model(os.path.join(self.output_dir, f"checkpoint-{self.global_step}"))
            
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
    
    def evaluate(self) -> Dict:
        """
        Evaluate the model.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Running evaluation")
        
        self.model.eval()
        
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, disable=self.disable_tqdm, desc="Evaluating"):
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
        
        # Compute metrics
        metrics = compute_metrics(all_logits, all_labels)
        
        # Log metrics
        for key, value in metrics.items():
            logger.info(f"Eval {key}: {value}")
        
        self.model.train()
        
        return metrics
    
    def save_model(self, output_dir: str):
        """
        Save the model to the specified directory.
        
        Args:
            output_dir: Directory to save the model to
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(output_dir)
        
        # Save training arguments
        with open(os.path.join(output_dir, "training_args.txt"), "w") as f:
            for key, value in self.config.items():
                f.write(f"{key}: {value}\n")
        
        logger.info(f"Model saved to {output_dir}")
