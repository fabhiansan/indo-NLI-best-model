"""
RoBERTa-based model for NLI task.
"""
import logging
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from src.models.base_model import BaseNLIModel

logger = logging.getLogger(__name__)


class RoBERTaForNLI(BaseNLIModel):
    """RoBERTa model for NLI task."""
    
    def __init__(self, config: Dict, num_labels: int = 3):
        """
        Initialize the RoBERTa model for NLI.
        
        Args:
            config: Configuration dictionary
            num_labels: Number of labels for classification
        """
        super().__init__(config, num_labels)
        
        self.pretrained_model_name = config["model"]["pretrained_model_name"]
        logger.info("Loading RoBERTa model from %s", self.pretrained_model_name)
        
        # Load model configuration
        model_config = AutoConfig.from_pretrained(
            self.pretrained_model_name,
            num_labels=num_labels,
            output_hidden_states=config["model"].get("output_hidden_states", False),
        )
        
        # Load pre-trained model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_model_name,
            config=model_config,
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs (not used in RoBERTa)
            labels: Ground truth labels
            
        Returns:
            Dictionary containing loss (if labels are provided) and logits
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        return outputs
    
    def save_pretrained(self, output_dir: str):
        """
        Save the model to the specified directory.
        
        Args:
            output_dir: Directory to save the model
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the original pretrained model name
        with open(os.path.join(output_dir, "pretrained_model_name.txt"), "w") as f:
            f.write(self.pretrained_model_name)
        
        # Save the model
        self.model.save_pretrained(output_dir)
        
        # Save the tokenizer
        try:
            tokenizer = self.get_tokenizer(self.pretrained_model_name)
            tokenizer.save_pretrained(output_dir)
            logger.info("Tokenizer saved successfully to %s", output_dir)
        except Exception as e:
            logger.warning("Failed to save tokenizer: %s", str(e))
    
    @classmethod
    def from_pretrained(cls, model_path: str, config: Optional[Dict] = None, **kwargs):
        """
        Load a model from the specified directory.
        
        Args:
            model_path: Path to the saved model
            config: Optional configuration dictionary
            
        Returns:
            Loaded model
        """
        logger.info(f"Loading model of type {cls.__name__} from {model_path}")
        
        # Check if we have a stored pretrained model name
        pretrained_model_name_path = os.path.join(model_path, "pretrained_model_name.txt")
        pretrained_model_name = None
        
        if os.path.exists(pretrained_model_name_path):
            with open(pretrained_model_name_path, "r") as f:
                pretrained_model_name = f.read().strip()
                logger.info(f"Found stored pretrained model name: {pretrained_model_name}")
        else:
            raise ValueError(f"No pretrained_model_name.txt found at {pretrained_model_name_path}. Cannot load model.")

        # Check if model has fine-tuned weights
        pytorch_model_path = os.path.join(model_path, "pytorch_model.bin")
        safetensors_path = os.path.join(model_path, "model.safetensors")
        has_fine_tuned_weights = os.path.exists(pytorch_model_path) or os.path.exists(safetensors_path)
        
        if not has_fine_tuned_weights:
            raise ValueError(f"No fine-tuned weights found at {pytorch_model_path} or {safetensors_path}. Cannot load model.")
        
        if os.path.exists(pytorch_model_path):
            logger.info(f"Found fine-tuned model weights at {pytorch_model_path}")
        if os.path.exists(safetensors_path):
            logger.info(f"Found fine-tuned model weights in safetensors format at {safetensors_path}")
        
        # Create minimal config for initial model instance
        if config is None:
            config = {
                "model": {
                    "pretrained_model_name": pretrained_model_name,
                    "output_hidden_states": False,
                }
            }
        
        # Create instance first
        instance = cls(config, **kwargs)
        
        # Now load the fine-tuned weights from the checkpoint
        try:
            # Load model configuration
            model_config = AutoConfig.from_pretrained(model_path)
            
            # Load the fine-tuned model directly from checkpoint
            instance.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                config=model_config,
            )
            
            logger.info(f"Successfully loaded fine-tuned model weights from {model_path}")
            return instance
        except Exception as e:
            logger.error(f"Error loading fine-tuned model: {str(e)}")
            raise ValueError(f"Failed to load fine-tuned model weights: {str(e)}")

    @staticmethod
    def get_tokenizer(pretrained_model_name: str, **kwargs) -> AutoTokenizer:
        """
        Get the tokenizer for the model.
        
        Args:
            pretrained_model_name: Path to model directory or name of pretrained model
            
        Returns:
            Tokenizer for the model
        """
        # If this is a local path, check for stored pretrained model name
        pretrained_model_name_path = os.path.join(pretrained_model_name, "pretrained_model_name.txt")
        if os.path.exists(pretrained_model_name_path):
            with open(pretrained_model_name_path, "r") as f:
                original_pretrained_name = f.read().strip()
                logger.info("Loading tokenizer from original pretrained model: %s", original_pretrained_name)
                return AutoTokenizer.from_pretrained(original_pretrained_name, **kwargs)
        
        # Otherwise try to load directly (might be a Hugging Face model name)
        logger.info("Loading tokenizer from: %s", pretrained_model_name)
        try:
            return AutoTokenizer.from_pretrained(pretrained_model_name, **kwargs)
        except OSError:
            # Fall back to default model if loading fails
            logger.warning("Failed to load tokenizer from %s, falling back to default", pretrained_model_name)
            return AutoTokenizer.from_pretrained("indolem/indobert-base-uncased", **kwargs)
