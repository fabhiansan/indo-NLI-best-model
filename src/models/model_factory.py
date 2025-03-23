"""
Model factory for NLI models.
"""
import logging
from typing import Dict, Optional, Type

from src.models.base_model import BaseNLIModel
from src.models.roberta_model import RoBERTaForNLI
from src.models.sentence_bert_model import (
    SentenceBERTForNLI,
    SentenceBERTWithSimpleClassifier,
    SentenceBERTWithProperClassifier,
)

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating NLI models."""
    
    _models = {
        "Indo-roBERTa": RoBERTaForNLI,
        "Indo-roBERTa-base": RoBERTaForNLI,
        "Sentence-BERT": SentenceBERTForNLI,
        "Sentence-BERT-Simple": SentenceBERTWithSimpleClassifier,
        "Sentence-BERT-Proper": SentenceBERTWithProperClassifier,
    }
    
    # Additional mappings for common alternative formats
    _model_name_aliases = {
        "indo_roberta": "Indo-roBERTa",
        "indo_roberta_base": "Indo-roBERTa-base",
        "sentence_bert": "Sentence-BERT",
        "sentence_bert_simple": "Sentence-BERT-Simple",
        "sentence_bert_proper": "Sentence-BERT-Proper",
    }
    
    @classmethod
    def _normalize_model_name(cls, model_name: str) -> str:
        """
        Normalize model name to match the keys in _models dictionary.
        
        Args:
            model_name: Input model name in any format
            
        Returns:
            Normalized model name
        """
        # Check if it's already a key in _models
        if model_name in cls._models:
            return model_name
        
        # Check if it's in our aliases
        if model_name in cls._model_name_aliases:
            return cls._model_name_aliases[model_name]
        
        # Try to normalize by converting underscores to hyphens and capitalizing
        normalized_name = model_name.replace('_', '-').title()
        
        # Special handling for common patterns
        normalized_name = normalized_name.replace('Roberta', 'roBERTa')
        normalized_name = normalized_name.replace('Bert', 'BERT')
        
        # Log the normalization for debugging
        if normalized_name != model_name:
            logger.info("Normalized model name from '%s' to '%s'", model_name, normalized_name)
        
        return normalized_name
    
    @classmethod
    def create_model(cls, config: Dict, num_labels: int = 3, **kwargs) -> BaseNLIModel:
        """
        Create a model based on the configuration.
        
        Args:
            config: Configuration dictionary
            num_labels: Number of labels for classification
            
        Returns:
            Instance of a model
        """
        model_name = config["model"]["name"]
        normalized_name = cls._normalize_model_name(model_name)
        
        if normalized_name not in cls._models:
            raise ValueError(f"Unknown model type: {model_name} (normalized: {normalized_name})")
        
        logger.info("Creating model of type %s", normalized_name)
        model_class = cls._models[normalized_name]
        
        return model_class(config, num_labels=num_labels, **kwargs)
    
    @classmethod
    def get_model_class(cls, model_name: str) -> Type[BaseNLIModel]:
        """
        Get the model class for a given model name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model class
        """
        normalized_name = cls._normalize_model_name(model_name)
        
        if normalized_name not in cls._models:
            raise ValueError(f"Unknown model type: {model_name} (normalized: {normalized_name})")
        
        return cls._models[normalized_name]
    
    @classmethod
    def from_pretrained(cls, model_path: str, model_name: str, config: Optional[Dict] = None, **kwargs) -> BaseNLIModel:
        """
        Load a pretrained model.
        
        Args:
            model_path: Path to the pretrained model
            model_name: Name of the model
            config: Configuration dictionary
            
        Returns:
            Loaded model
        """
        normalized_name = cls._normalize_model_name(model_name)
        
        if normalized_name not in cls._models:
            raise ValueError(f"Unknown model type: {model_name}")
        
        logger.info("Loading model of type %s from %s", normalized_name, model_path)
        model_class = cls._models[normalized_name]
        
        return model_class.from_pretrained(model_path, config, **kwargs)
