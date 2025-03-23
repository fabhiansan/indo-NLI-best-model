"""
Sentence-BERT-based models for NLI task with different classifier architectures.
"""
import logging
import os
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from src.models.base_model import BaseNLIModel

logger = logging.getLogger(__name__)


class SentenceBERTForNLI(BaseNLIModel):
    """Base Sentence-BERT model for NLI task."""
    
    def __init__(self, config: Dict, num_labels: int = 3):
        """
        Initialize the Sentence-BERT model for NLI.
        
        Args:
            config: Configuration dictionary
            num_labels: Number of labels for classification
        """
        super().__init__(config, num_labels)
        
        self.pretrained_model_name = config["model"]["pretrained_model_name"]
        logger.info("Loading Sentence-BERT model from %s", self.pretrained_model_name)
        
        # Load model configuration
        model_config = AutoConfig.from_pretrained(
            self.pretrained_model_name,
            output_hidden_states=config["model"].get("output_hidden_states", True),
        )
        
        # Load pre-trained model
        self.bert = AutoModel.from_pretrained(
            self.pretrained_model_name,
            config=model_config,
        )
        
        # Default classifier is just a linear layer
        self.dropout = nn.Dropout(config["model"].get("hidden_dropout_prob", 0.1))
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
        # Pooling strategy (mean pooling by default)
        self.pooling_strategy = config["model"].get("pooling_strategy", "mean")
    
    def _apply_pooling(self, sequence_output, attention_mask):
        """
        Apply pooling to the sequence output.
        
        Args:
            sequence_output: Output sequence from the model
            attention_mask: Attention mask
            
        Returns:
            Pooled representation
        """
        if self.pooling_strategy == "cls":
            # Use [CLS] token embedding
            return sequence_output[:, 0]
        elif self.pooling_strategy == "mean":
            # Mean pooling: take average of all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
            sum_embeddings = torch.sum(sequence_output * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask
        else:
            raise ValueError(f"Unsupported pooling strategy: {self.pooling_strategy}")
    
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
            token_type_ids: Token type IDs
            labels: Ground truth labels
            
        Returns:
            Dictionary containing loss (if labels are provided) and logits
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        sequence_output = outputs.last_hidden_state
        pooled_output = self._apply_pooling(sequence_output, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {"loss": loss, "logits": logits}
    
    def save_pretrained(self, output_dir: str):
        """
        Save the model to the specified directory.
        
        Args:
            output_dir: Directory to save the model
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save original pretrained model name
        with open(os.path.join(output_dir, "pretrained_model_name.txt"), "w") as f:
            f.write(self.pretrained_model_name)
        
        # Save the model
        self.bert.save_pretrained(output_dir)
        
        # Save classifier weights
        torch.save(
            {
                "classifier": self.classifier.state_dict(),
                "pooling_strategy": self.pooling_strategy,
                "num_labels": self.num_labels,
            },
            os.path.join(output_dir, "classifier.pt"),
        )
    
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
        # Check if we have a stored pretrained model name
        pretrained_model_name_path = os.path.join(model_path, "pretrained_model_name.txt")
        pretrained_model_name = None
        
        if os.path.exists(pretrained_model_name_path):
            with open(pretrained_model_name_path, "r") as f:
                pretrained_model_name = f.read().strip()
                logger.info("Found stored pretrained model name: %s", pretrained_model_name)
        
        # If not found, and no config provided, use a default pretrained model
        if pretrained_model_name is None and config is None:
            pretrained_model_name = "firqaaa/indo-sentence-bert-base"
            logger.warning("No pretrained model name found, using default: %s", pretrained_model_name)
        
        if config is None:
            # Create a default config if none is provided
            config = {
                "model": {
                    "pretrained_model_name": pretrained_model_name or "firqaaa/indo-sentence-bert-base",
                    "output_hidden_states": True,
                }
            }
        
        model = cls(config, **kwargs)
        
        # Load classifier weights if they exist
        classifier_path = os.path.join(model_path, "classifier.pt")
        if os.path.exists(classifier_path):
            classifier_dict = torch.load(classifier_path)
            model.classifier.load_state_dict(classifier_dict["classifier"])
            model.pooling_strategy = classifier_dict.get("pooling_strategy", "mean")
            model.num_labels = classifier_dict.get("num_labels", 3)
        
        return model
    
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
            return AutoTokenizer.from_pretrained("firqaaa/indo-sentence-bert-base", **kwargs)


class SentenceBERTWithSimpleClassifier(SentenceBERTForNLI):
    """Sentence-BERT model with a simple classifier."""
    
    def __init__(self, config: Dict, num_labels: int = 3):
        """
        Initialize the Sentence-BERT model with a simple classifier.
        
        Args:
            config: Configuration dictionary
            num_labels: Number of labels for classification
        """
        super().__init__(config, num_labels)
        
        # Override the default classifier with a simple MLP
        hidden_dropout_prob = config["model"].get("classifier_dropout", 0.1)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(self.bert.config.hidden_size, num_labels),
        )


class SentenceBERTWithProperClassifier(SentenceBERTForNLI):
    """Sentence-BERT model with a proper classifier."""
    
    def __init__(self, config: Dict, num_labels: int = 3):
        """
        Initialize the Sentence-BERT model with a proper classifier.
        
        Args:
            config: Configuration dictionary
            num_labels: Number of labels for classification
        """
        super().__init__(config, num_labels)
        
        # Override the default classifier with a more complex MLP
        hidden_dropout_prob = config["model"].get("classifier_dropout", 0.1)
        hidden_dim = config["model"].get("classifier_hidden_dim", 768)
        num_layers = config["model"].get("classifier_num_layers", 2)
        
        # Create a multi-layer classifier
        layers = []
        input_dim = self.bert.config.hidden_size
        
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(hidden_dropout_prob))
            input_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(hidden_dim, num_labels))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model with proper classifier.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            labels: Ground truth labels
            
        Returns:
            Dictionary containing loss (if labels are provided) and logits
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        
        # Use both last hidden state and pooled output for classification
        sequence_output = outputs.last_hidden_state
        pooled_output = self._apply_pooling(sequence_output, attention_mask)
        
        # Additional feature: concatenate with the [CLS] token embedding if available
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        
        # Run through classifier
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {"loss": loss, "logits": logits}
