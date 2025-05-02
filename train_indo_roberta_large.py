#!/usr/bin/env python
"""
Script to train the indonesian-roberta-large model on the IndoNLI dataset.
"""
import os
import sys
import subprocess

def main():
    """Run the training script with the indo_roberta_large configuration."""
    # Get the current directory (project root)
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Create a modified environment with PYTHONPATH set to include the project root
    env = os.environ.copy()
    python_path = env.get('PYTHONPATH', '')
    if python_path:
        env['PYTHONPATH'] = f"{project_root}:{python_path}"
    else:
        env['PYTHONPATH'] = project_root

    # Set up the command
    cmd = [
        "python", "scripts/train.py",
        "--config", "configs/indo_roberta_large.yaml",
        "--fp16"  # Enable mixed precision training
    ]

    # Run the command with the modified environment
    print(f"Running command: {' '.join(cmd)}")
    print(f"PYTHONPATH: {env['PYTHONPATH']}")
    subprocess.run(cmd, check=True, env=env)

if __name__ == "__main__":
    main()
