#!/usr/bin/env python
"""
Script to run the full evaluation pipeline on all models.
"""
import argparse
import logging
import os
import yaml
import glob
import time
from datetime import datetime

from src.utils.config import load_config
from src.utils.logging import setup_logging, log_system_info


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--configs_dir", type=str, default="./configs", help="Directory containing model configurations")
    parser.add_argument("--output_dir", type=str, default="./models", help="Output directory for models")
    parser.add_argument("--num_epochs", type=int, help="Override number of epochs for all models")
    parser.add_argument("--batch_size", type=int, help="Override batch size for all models")
    parser.add_argument("--seed", type=int, help="Override random seed for all models")
    parser.add_argument("--skip_training", action="store_true", help="Skip training and only run evaluation")
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation after training")
    parser.add_argument("--test_sets", type=str, nargs="+", default=["test_lay", "test_expert"],
                        help="Test sets to evaluate on")
    parser.add_argument("--generate_report", action="store_true", help="Generate comparison report after all experiments")
    return parser.parse_args()


def run_experiment(config_path, args):
    """
    Run a single experiment.
    
    Args:
        config_path: Path to the model configuration file
        args: Command line arguments
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running experiment with config: {config_path}")
    
    # Extract model name from config
    config = load_config(config_path)
    model_name = config["model"]["name"]
    model_output_dir = os.path.join(args.output_dir, model_name.lower().replace(" ", "-"))
    
    # Build command for training
    if not args.skip_training:
        train_cmd = f"python scripts/train.py --config {config_path} --output_dir {model_output_dir}"
        
        if args.num_epochs:
            train_cmd += f" --num_epochs {args.num_epochs}"
        
        if args.batch_size:
            train_cmd += f" --batch_size {args.batch_size}"
        
        if args.seed:
            train_cmd += f" --seed {args.seed}"
        
        logger.info(f"Running training: {train_cmd}")
        os.system(train_cmd)
    
    # Run evaluation on all test sets
    if not args.skip_eval:
        for test_set in args.test_sets:
            eval_output_dir = os.path.join(model_output_dir, f"evaluation_{test_set}")
            
            # Get the best model path
            best_model_path = os.path.join(model_output_dir, "best")
            if not os.path.exists(best_model_path):
                # Try to find the final model
                final_model_path = os.path.join(model_output_dir, "final")
                if os.path.exists(final_model_path):
                    best_model_path = final_model_path
                else:
                    # Find the latest epoch model
                    epoch_models = glob.glob(os.path.join(model_output_dir, "epoch-*"))
                    if epoch_models:
                        best_model_path = max(epoch_models, key=os.path.getctime)
                    else:
                        logger.warning(f"No trained model found for {model_name}")
                        continue
            
            eval_cmd = f"python scripts/evaluate.py --model_path {best_model_path} --model_name {model_name} --test_set {test_set} --output_dir {eval_output_dir}"
            
            if args.seed:
                eval_cmd += f" --seed {args.seed}"
            
            logger.info(f"Running evaluation: {eval_cmd}")
            os.system(eval_cmd)


def generate_comparison_report(args):
    """
    Generate comparison report for all models.
    
    Args:
        args: Command line arguments
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating comparison report")
    
    report_cmd = f"python scripts/generate_report.py --models_dir {args.output_dir} --output_dir ./reports/comparison --test_sets {' '.join(args.test_sets)}"
    
    logger.info(f"Running report generation: {report_cmd}")
    os.system(report_cmd)


def main():
    """Main function to run all experiments."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"run_all_experiments_{timestamp}.log"
    logger = setup_logging(log_file=log_file)
    
    logger.info(f"Arguments: {args}")
    log_system_info(logger)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all configuration files
    config_files = glob.glob(os.path.join(args.configs_dir, "*.yaml"))
    logger.info(f"Found {len(config_files)} configuration files")
    
    # Run experiments for all models
    start_time = time.time()
    
    for config_file in config_files:
        run_experiment(config_file, args)
    
    total_time = time.time() - start_time
    logger.info(f"All experiments completed in {total_time:.2f} seconds")
    
    # Generate comparison report if requested
    if args.generate_report:
        generate_comparison_report(args)
        logger.info("Comparison report generated")
    
    logger.info("All experiments and evaluations completed")


if __name__ == "__main__":
    main()
