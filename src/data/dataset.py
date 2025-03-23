"""
Dataset utilities for loading and processing the IndoNLI dataset.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Union

import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

# Label mapping for the IndoNLI dataset
INDONLI_LABELS = {"entailment": 0, "neutral": 1, "contradiction": 2}


@dataclass
class DataCollatorForNLI:
    """
    Data collator for NLI tasks. Handles tokenization and padding.
    """
    
    tokenizer: PreTrainedTokenizer
    max_length: int = 128
    padding: Union[bool, str] = "max_length"
    truncation: bool = True
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """Collate examples for training or evaluation."""
        batch = {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "labels": torch.tensor([f["labels"] for f in features], dtype=torch.long),
        }
        
        if "token_type_ids" in features[0]:
            batch["token_type_ids"] = torch.stack([f["token_type_ids"] for f in features])
            
        return batch


class IndoNLIDataset(Dataset):
    """Dataset class for the IndoNLI dataset."""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        max_length: int = 128,
        dataset_name: str = "afaji/indonli",
    ):
        """
        Initialize the IndoNLI dataset.
        
        Args:
            tokenizer: Tokenizer to use for encoding the text
            split: Dataset split to use (train, validation, test_lay, test_expert)
            max_length: Maximum sequence length for tokenization
            dataset_name: Name of the dataset on Hugging Face Hub
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(f"Loading IndoNLI dataset ({split} split)...")
        self.dataset = datasets.load_dataset(dataset_name, split=split)
        logger.info(f"Loaded {len(self.dataset)} examples from {split} split")
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        
        # Tokenize the premise and hypothesis
        encoding = self.tokenizer(
            example["premise"],
            example["hypothesis"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Remove batch dimension added by the tokenizer
        for key in encoding:
            encoding[key] = encoding[key].squeeze(0)
        
        # Add label - handle both string and integer labels
        if isinstance(example["label"], str):
            encoding["labels"] = INDONLI_LABELS[example["label"]]
        else:
            # If the label is already an integer, use it directly
            encoding["labels"] = example["label"]
        
        return encoding


def get_nli_dataloader(
    tokenizer: PreTrainedTokenizer,
    split: str,
    batch_size: int,
    max_length: int = 128,
    dataset_name: str = "afaji/indonli",
    num_workers: int = 4,
    shuffle: bool = None,
) -> DataLoader:
    """
    Get a dataloader for the IndoNLI dataset.
    
    Args:
        tokenizer: Tokenizer to use for encoding the text
        split: Dataset split to use (train, validation, test_lay, test_expert)
        batch_size: Batch size for the dataloader
        max_length: Maximum sequence length for tokenization
        dataset_name: Name of the dataset on Hugging Face Hub
        num_workers: Number of workers for the dataloader
        shuffle: Whether to shuffle the dataset (default: True for train, False otherwise)
        
    Returns:
        DataLoader for the specified split
    """
    if shuffle is None:
        shuffle = split == "train"
    
    dataset = IndoNLIDataset(
        tokenizer=tokenizer,
        split=split,
        max_length=max_length,
        dataset_name=dataset_name,
    )
    
    data_collator = DataCollatorForNLI(
        tokenizer=tokenizer,
        max_length=max_length,
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=data_collator,
        num_workers=num_workers,
    )
    
    return dataloader
