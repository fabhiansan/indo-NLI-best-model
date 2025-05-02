#!/usr/bin/env python
"""
Script to train the indonesian-roberta-large model on the IndoNLI dataset.
"""
import os
import sys
import subprocess

def main():
    """Run the training script with the indo_roberta_large configuration."""
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set up the command
    cmd = [
        "python", "scripts/train.py",
        "--config", "configs/indo_roberta_large.yaml",
        "--fp16"  # Enable mixed precision training
    ]
    
    # Run the command
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
