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
        if model_name not in cls._models:
            raise ValueError(f"Unknown model type: {model_name}")
        
        logger.info(f"Creating model of type {model_name}")
        model_class = cls._models[model_name]
        
        return model_class(config, num_labels=num_labels, **kwargs)
    
    @classmethod
    def get_model_class(cls, model_name: str) -> Type[BaseNLIModel]:
        """
        Get the model class by name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model class
        """
        if model_name not in cls._models:
            raise ValueError(f"Unknown model type: {model_name}")
        
        return cls._models[model_name]
    
    @classmethod
    def from_pretrained(cls, model_path: str, model_name: Optional[str] = None, config: Optional[Dict] = None, **kwargs) -> BaseNLIModel:
        """
        Load a model from the specified directory.
        
        Args:
            model_path: Path to the saved model
            model_name: Name of the model (if not provided, will be inferred)
            config: Optional configuration dictionary
            
        Returns:
            Loaded model
        """
        if model_name is None:
            # Try to infer model name from the path
            for name in cls._models:
                if name.lower() in model_path.lower():
                    model_name = name
                    break
            
            if model_name is None:
                raise ValueError(f"Could not infer model type from path: {model_path}")
        
        if model_name not in cls._models:
            raise ValueError(f"Unknown model type: {model_name}")
        
        logger.info(f"Loading model of type {model_name} from {model_path}")
        model_class = cls._models[model_name]
        
        return model_class.from_pretrained(model_path, config, **kwargs)
