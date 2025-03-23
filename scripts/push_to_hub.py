#!/usr/bin/env python
"""
Script to push trained models to the Hugging Face Hub.
"""
import argparse
import logging
import os
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Add the project root to the path
project_root = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, project_root)

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

from src.models.model_factory import ModelFactory
from src.utils.logging import setup_logging
from src.utils.config import load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Push model to Hugging Face Hub")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--model_name", type=str, help="Name of the model type")
    parser.add_argument("--repo_name", type=str, required=True, help="Repository name on Hugging Face Hub")
    parser.add_argument("--organization", type=str, help="Organization name on Hugging Face Hub")
    parser.add_argument("--private", action="store_true", help="Whether to make the repository private")
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    parser.add_argument("--metrics_file", type=str, help="Path to metrics CSV file")
    return parser.parse_args()


def create_model_card(args, model_info, metrics=None):
    """
    Create model card markdown content.
    
    Args:
        args: Command line arguments
        model_info: Dictionary containing model information
        metrics: Optional dictionary containing evaluation metrics
        
    Returns:
        Model card content as a string
    """
    model_name = model_info.get("model_name", "Unknown")
    dataset_name = model_info.get("dataset_name", "afaji/indonli")
    
    model_card = []
    model_card.append(f"---")
    model_card.append(f"language: id")
    model_card.append(f"license: mit")
    model_card.append(f"tags:")
    model_card.append(f"- indonesian")
    model_card.append(f"- nli")
    model_card.append(f"- natural-language-inference")
    model_card.append(f"- text-classification")
    
    if model_info.get("pretrained_model_name"):
        model_card.append(f"- {model_info['pretrained_model_name']}")
    
    datasets_tag = dataset_name.replace("/", "--")
    model_card.append(f"- {datasets_tag}")
    model_card.append(f"datasets:")
    model_card.append(f"- {dataset_name}")
    model_card.append(f"---")
    
    # Model card content
    model_card.append(f"# {model_name} for Indonesian Natural Language Inference")
    model_card.append("")
    model_card.append(f"This model is fine-tuned on the [IndoNLI dataset]({dataset_name}) for natural language inference in Indonesian.")
    model_card.append("")
    
    # Model information
    model_card.append("## Model Description")
    model_card.append("")
    model_card.append(f"- **Model Type:** {model_name}")
    
    if model_info.get("pretrained_model_name"):
        model_card.append(f"- **Base Model:** [{model_info['pretrained_model_name']}](https://huggingface.co/{model_info['pretrained_model_name']})")
    
    model_card.append(f"- **Task:** Natural Language Inference (Textual Entailment)")
    model_card.append(f"- **Language:** Indonesian")
    model_card.append(f"- **License:** MIT")
    model_card.append("")
    
    # Training information
    model_card.append("## Training Procedure")
    model_card.append("")
    
    if model_info.get("training"):
        training_info = model_info["training"]
        model_card.append("### Training Hyperparameters")
        model_card.append("")
        model_card.append("The model was trained with the following hyperparameters:")
        model_card.append("")
        model_card.append("- Learning Rate: " + str(training_info.get("learning_rate", "N/A")))
        model_card.append("- Batch Size: " + str(training_info.get("batch_size", "N/A")))
        model_card.append("- Number of Epochs: " + str(training_info.get("num_epochs", "N/A")))
        model_card.append("- Warmup Ratio: " + str(training_info.get("warmup_ratio", "N/A")))
        model_card.append("- Weight Decay: " + str(training_info.get("weight_decay", "N/A")))
        model_card.append("")
    
    # Dataset information
    model_card.append("## Dataset")
    model_card.append("")
    model_card.append(f"This model was trained on the [IndoNLI dataset]({dataset_name}), which contains 10k sentence pairs as a benchmark for natural language inference (NLI) in Indonesian.")
    model_card.append("")
    model_card.append("The dataset is split into:")
    model_card.append("- Training set: 10k pairs")
    model_card.append("- Validation set: 2.5k pairs")
    model_card.append("- Test set (lay): 2.5k pairs")
    model_card.append("- Test set (expert): 2.5k pairs")
    model_card.append("")
    
    # Evaluation results
    if metrics:
        model_card.append("## Evaluation Results")
        model_card.append("")
        model_card.append("The model was evaluated on the test set with the following results:")
        model_card.append("")
        model_card.append("| Metric | Score |")
        model_card.append("|--------|-------|")
        
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                model_card.append(f"| {metric} | {value:.4f} |")
        
        model_card.append("")
    
    # Usage information
    model_card.append("## Usage")
    model_card.append("")
    model_card.append("```python")
    model_card.append("from transformers import AutoTokenizer, AutoModelForSequenceClassification")
    model_card.append("")
    repo_id = f"{args.organization + '/' if args.organization else ''}{args.repo_name}"
    model_card.append(f'tokenizer = AutoTokenizer.from_pretrained("{repo_id}")')
    model_card.append(f'model = AutoModelForSequenceClassification.from_pretrained("{repo_id}")')
    model_card.append("")
    model_card.append("# Prepare the input")
    model_card.append('premise = "Seorang wanita sedang makan di restoran."')
    model_card.append('hypothesis = "Seorang wanita sedang berada di luar ruangan."')
    model_card.append("")
    model_card.append("# Tokenize the input")
    model_card.append('inputs = tokenizer(premise, hypothesis, return_tensors="pt")')
    model_card.append("")
    model_card.append("# Get the prediction")
    model_card.append("outputs = model(**inputs)")
    model_card.append("predictions = outputs.logits.argmax(dim=1)")
    model_card.append("")
    model_card.append("# Map predictions to labels")
    model_card.append('id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}')
    model_card.append("predicted_label = id2label[predictions.item()]")
    model_card.append('print(f"Predicted label: {predicted_label}")')
    model_card.append("```")
    model_card.append("")
    
    # Citation information
    model_card.append("## Citation")
    model_card.append("")
    model_card.append("If you use this model, please cite the IndoNLI paper:")
    model_card.append("")
    model_card.append("```bibtex")
    model_card.append("@inproceedings{mahendra-etal-2021-indonli,")
    model_card.append("    title = {IndoNLI: A Natural Language Inference Dataset for Indonesian},")
    model_card.append("    author = {Mahendra, Rahmad and Aji, Alham Fikri and Louvan, Samuel and Rahman, Fahrurrozi and Vania, Clara},")
    model_card.append("    booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},")
    model_card.append("    year = {2021},")
    model_card.append("    publisher = {Association for Computational Linguistics},")
    model_card.append("}")
    model_card.append("```")
    
    return "\n".join(model_card)


def prepare_model_for_hub(args, model_path, temp_dir):
    """
    Prepare model for pushing to the Hub.
    
    Args:
        args: Command line arguments
        model_path: Path to the model directory
        temp_dir: Temporary directory for preparing the model
        
    Returns:
        Dictionary containing model information
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Preparing model for Hub from {model_path}")
    
    # Load model configuration
    config_path = args.config if args.config else os.path.join(model_path, "training_args.txt")
    model_info = {}
    
    if os.path.exists(config_path):
        if config_path.endswith(".yaml") or config_path.endswith(".yml"):
            config = load_config(config_path)
            model_info = config
        else:
            # Parse training args file
            with open(config_path, "r") as f:
                lines = f.readlines()
            
            config = {}
            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    config[key.strip()] = value.strip()
            
            if "model" in config and isinstance(config["model"], dict):
                model_info = config
            else:
                # Try to reconstruct config structure
                model_info = {
                    "model": {
                        "name": args.model_name or os.path.basename(model_path),
                    },
                    "training": {},
                    "data": {
                        "dataset_name": "afaji/indonli",
                    }
                }
                
                for key, value in config.items():
                    if key.startswith("model."):
                        model_info["model"][key.replace("model.", "")] = value
                    elif key.startswith("training."):
                        model_info["training"][key.replace("training.", "")] = value
                    elif key.startswith("data."):
                        model_info["data"][key.replace("data.", "")] = value
    else:
        logger.warning(f"No configuration file found at {config_path}")
        model_info = {
            "model": {
                "name": args.model_name or os.path.basename(model_path),
            },
            "data": {
                "dataset_name": "afaji/indonli",
            }
        }
    
    # Load metrics if available
    metrics = None
    metrics_file = args.metrics_file
    if not metrics_file:
        # Try to find metrics file in evaluation directories
        for test_set in ["test_lay", "test_expert"]:
            eval_dir = os.path.join(os.path.dirname(model_path), f"evaluation_{test_set}")
            potential_metrics_file = os.path.join(eval_dir, "metrics.csv")
            if os.path.exists(potential_metrics_file):
                metrics_file = potential_metrics_file
                break
    
    if metrics_file and os.path.exists(metrics_file):
        import pandas as pd
        metrics_df = pd.read_csv(metrics_file)
        if not metrics_df.empty:
            metrics = metrics_df.iloc[0].to_dict()
    
    # Create model card
    model_card_content = create_model_card(args, model_info, metrics)
    with open(os.path.join(temp_dir, "README.md"), "w") as f:
        f.write(model_card_content)
    
    # Prepare config.json
    if os.path.exists(os.path.join(model_path, "config.json")):
        shutil.copy(os.path.join(model_path, "config.json"), os.path.join(temp_dir, "config.json"))
    else:
        # Create a minimal config
        label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
        id2label = {str(i): label for i, label in label_map.items()}
        label2id = {label: str(i) for i, label in label_map.items()}
        
        config_dict = {
            "model_type": "roberta" if "roberta" in model_info.get("model", {}).get("name", "").lower() else "bert",
            "num_labels": 3,
            "id2label": id2label,
            "label2id": label2id,
        }
        
        with open(os.path.join(temp_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
    
    # Copy model files
    for filename in os.listdir(model_path):
        src_path = os.path.join(model_path, filename)
        dst_path = os.path.join(temp_dir, filename)
        
        if os.path.isfile(src_path) and filename not in ["README.md", "training_args.txt"]:
            shutil.copy(src_path, dst_path)
    
    return model_info


def push_to_hub(args):
    """
    Push model to Hugging Face Hub.
    
    Args:
        args: Command line arguments
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Pushing model from {args.model_path} to Hugging Face Hub")
    
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        logger.error("huggingface_hub package not installed. Please install it with 'pip install huggingface_hub'.")
        return
    
    # Create a temporary directory for preparing the model
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        # Prepare model for Hub
        model_info = prepare_model_for_hub(args, args.model_path, temp_dir)
        
        # Determine repository ID
        repo_id = f"{args.organization + '/' if args.organization else ''}{args.repo_name}"
        logger.info(f"Repository ID: {repo_id}")
        
        # Create repo if it doesn't exist
        try:
            api = HfApi()
            create_repo(repo_id, private=args.private, exist_ok=True)
            logger.info(f"Repository created/verified: {repo_id}")
            
            # Upload model to Hub
            logger.info(f"Uploading model files to {repo_id}")
            api.upload_folder(
                folder_path=temp_dir,
                repo_id=repo_id,
                commit_message=f"Upload {model_info.get('model', {}).get('name', 'model')} for Indonesian NLI",
            )
            logger.info(f"Model uploaded successfully to {repo_id}")
            
            # Log model URL
            model_url = f"https://huggingface.co/{repo_id}"
            logger.info(f"Model is now available at: {model_url}")
            
        except Exception as e:
            logger.error(f"Error pushing model to Hub: {e}")


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    log_file = f"push_to_hub_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logging(log_file=log_file)
    
    logger.info(f"Arguments: {args}")
    
    # Check for Hugging Face token
    if "HUGGINGFACE_TOKEN" not in os.environ:
        logger.warning("HUGGINGFACE_TOKEN environment variable not set.")
        logger.warning("You might need to login using `huggingface-cli login` or set the token.")
    
    # Push model to Hub
    push_to_hub(args)


if __name__ == "__main__":
    main()
