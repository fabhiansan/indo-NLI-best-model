"""
Base model class for NLI models.
"""
import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class BaseNLIModel(nn.Module):
    """Base class for all NLI models."""
    
    def __init__(self, config: Dict, num_labels: int = 3):
        """
        Initialize the base NLI model.
        
        Args:
            config: Configuration dictionary
            num_labels: Number of labels for classification
        """
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs (for BERT-based models)
            labels: Ground truth labels
            
        Returns:
            Tuple containing loss (if labels are provided) and logits
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def save_pretrained(self, output_dir: str):
        """
        Save the model to the specified directory.
        
        Args:
            output_dir: Directory to save the model to
        """
        raise NotImplementedError("Subclasses must implement this method")
    
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
        raise NotImplementedError("Subclasses must implement this method")
    
    @staticmethod
    def get_tokenizer(pretrained_model_name: str, **kwargs) -> PreTrainedTokenizer:
        """
        Get the tokenizer for the model.
        
        Args:
            pretrained_model_name: Name of the pretrained model
            
        Returns:
            Tokenizer for the model
        """
        raise NotImplementedError("Subclasses must implement this method")
