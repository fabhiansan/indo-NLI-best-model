"""
Logging utilities for the project.
"""
import logging
import os
import sys
from typing import Optional

import torch


def setup_logging(
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
    console_log: bool = True,
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to the log file
        log_level: Log level
        console_log: Whether to log to console
        
    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Add file handler if log_file is provided
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Add console handler if console_log is True
    if console_log:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger


def log_system_info(logger: logging.Logger) -> None:
    """
    Log system information.
    
    Args:
        logger: Logger instance
    """
    logger.info("System information:")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA available: Yes")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        logger.info(f"Current GPU: {torch.cuda.current_device()}")
        logger.info(f"GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        logger.info("CUDA available: No")
