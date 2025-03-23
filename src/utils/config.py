"""
Configuration utilities for the project.
"""
import logging
import os
from typing import Dict

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    logger.info(f"Loading configuration from {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Ensure all required sections are present
    required_sections = ["model", "training", "data", "output"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section '{section}' in configuration")
    
    # Ensure output directories are absolute paths
    for key in ["output_dir", "logging_dir", "report_dir"]:
        if key in config["output"]:
            path = config["output"][key]
            if not os.path.isabs(path):
                config["output"][key] = os.path.abspath(path)
    
    return config


def save_config(config: Dict, output_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save the configuration file
    """
    logger.info(f"Saving configuration to {output_path}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
