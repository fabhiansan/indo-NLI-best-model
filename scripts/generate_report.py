#!/usr/bin/env python
"""
Script to generate a comprehensive comparison report of all models.
"""
import argparse
import logging
import os
import glob
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logging import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate model comparison report")
    parser.add_argument("--models_dir", type=str, required=True, 
                        help="Directory containing model evaluation results")
    parser.add_argument("--output_dir", type=str, default="./reports/comparison",
                        help="Output directory for the comparison report")
    parser.add_argument("--test_sets", type=str, nargs="+", default=["test_lay", "test_expert"],
                        help="Test sets to include in the comparison")
    return parser.parse_args()


def collect_model_metrics(models_dir, test_sets):
    """
    Collect metrics from all models and test sets.
    
    Args:
        models_dir: Directory containing model directories
        test_sets: List of test sets to include
        
    Returns:
        DataFrame containing metrics for all models
    """
    results = []
    
    # Get all model directories
    model_dirs = [d for d in os.listdir(models_dir) 
                 if os.path.isdir(os.path.join(models_dir, d))]
    
    for model_name in model_dirs:
        model_path = os.path.join(models_dir, model_name)
        
        for test_set in test_sets:
            eval_dir = os.path.join(model_path, f"evaluation_{test_set}")
            
            if not os.path.exists(eval_dir):
                logging.warning(f"Evaluation directory not found for {model_name} on {test_set}")
                continue
            
            # Find metrics file
            metrics_file = os.path.join(eval_dir, "metrics.csv")
            if not os.path.exists(metrics_file):
                logging.warning(f"Metrics file not found for {model_name} on {test_set}")
                continue
            
            # Load metrics
            metrics_df = pd.read_csv(metrics_file)
            
            # Add model name and test set
            metrics_df["model_name"] = model_name
            metrics_df["test_set"] = test_set
            
            results.append(metrics_df)
    
    if not results:
        raise ValueError("No evaluation results found")
    
    # Combine all results
    return pd.concat(results, ignore_index=True)


def generate_comparison_plots(metrics_df, output_dir):
    """
    Generate comparison plots for all models.
    
    Args:
        metrics_df: DataFrame containing metrics for all models
        output_dir: Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define metrics to plot
    main_metrics = ["accuracy", "precision", "recall", "f1"]
    
    # Generate bar plots for each test set
    for test_set in metrics_df["test_set"].unique():
        test_df = metrics_df[metrics_df["test_set"] == test_set]
        
        # Main metrics comparison
        plt.figure(figsize=(12, 8))
        df_plot = test_df[["model_name"] + main_metrics].melt(
            id_vars=["model_name"], 
            var_name="Metric", 
            value_name="Value"
        )
        sns.barplot(x="model_name", y="Value", hue="Metric", data=df_plot)
        plt.title(f"Model Comparison on {test_set}")
        plt.xlabel("Model")
        plt.ylabel("Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"comparison_{test_set}_main_metrics.png"))
        plt.close()
        
        # Accuracy comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(x="model_name", y="accuracy", data=test_df)
        plt.title(f"Accuracy Comparison on {test_set}")
        plt.xlabel("Model")
        plt.ylabel("Accuracy")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"comparison_{test_set}_accuracy.png"))
        plt.close()
        
        # Per-class F1 comparison
        per_class_cols = [col for col in test_df.columns if col.startswith("f1_")]
        if per_class_cols:
            plt.figure(figsize=(12, 8))
            df_plot = test_df[["model_name"] + per_class_cols].melt(
                id_vars=["model_name"], 
                var_name="Class", 
                value_name="F1 Score"
            )
            # Clean up class names
            df_plot["Class"] = df_plot["Class"].str.replace("f1_", "")
            sns.barplot(x="model_name", y="F1 Score", hue="Class", data=df_plot)
            plt.title(f"Per-class F1 Score Comparison on {test_set}")
            plt.xlabel("Model")
            plt.ylabel("F1 Score")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"comparison_{test_set}_f1_per_class.png"))
            plt.close()
    
    # Generate test sets comparison for each model
    if len(metrics_df["test_set"].unique()) > 1:
        for model_name in metrics_df["model_name"].unique():
            model_df = metrics_df[metrics_df["model_name"] == model_name]
            
            plt.figure(figsize=(12, 8))
            df_plot = model_df[["test_set"] + main_metrics].melt(
                id_vars=["test_set"], 
                var_name="Metric", 
                value_name="Value"
            )
            sns.barplot(x="test_set", y="Value", hue="Metric", data=df_plot)
            plt.title(f"{model_name} Performance Across Test Sets")
            plt.xlabel("Test Set")
            plt.ylabel("Score")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"comparison_{model_name}_test_sets.png"))
            plt.close()


def generate_comparison_tables(metrics_df, output_dir):
    """
    Generate comparison tables for all models.
    
    Args:
        metrics_df: DataFrame containing metrics for all models
        output_dir: Output directory for tables
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define metrics to include in the table
    main_metrics = ["accuracy", "precision", "recall", "f1"]
    
    # Generate tables for each test set
    for test_set in metrics_df["test_set"].unique():
        test_df = metrics_df[metrics_df["test_set"] == test_set]
        
        # Main metrics table
        table_df = test_df[["model_name"] + main_metrics].copy()
        table_df.set_index("model_name", inplace=True)
        
        # Round to 4 decimal places
        table_df = table_df.round(4)
        
        # Highlight the best model for each metric
        styled_df = table_df.style.highlight_max(axis=0)
        
        # Save to CSV
        table_df.to_csv(os.path.join(output_dir, f"comparison_{test_set}_metrics.csv"))
        
        # Save to HTML with highlighting
        styled_df.to_html(os.path.join(output_dir, f"comparison_{test_set}_metrics.html"))
    
    # Generate overall ranking
    ranking_df = pd.DataFrame()
    for metric in main_metrics:
        temp_df = metrics_df.groupby("model_name")[metric].mean().sort_values(ascending=False)
        ranking_df[f"{metric}_rank"] = range(1, len(temp_df) + 1)
        ranking_df.index = temp_df.index
    
    # Calculate overall rank (average of all ranks)
    ranking_df["overall_rank"] = ranking_df.mean(axis=1)
    ranking_df = ranking_df.sort_values("overall_rank")
    
    # Save overall ranking
    ranking_df.to_csv(os.path.join(output_dir, "overall_model_ranking.csv"))


def generate_report(metrics_df, output_dir):
    """
    Generate a comprehensive comparison report.
    
    Args:
        metrics_df: DataFrame containing metrics for all models
        output_dir: Output directory for the report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots
    plots_dir = os.path.join(output_dir, "plots")
    generate_comparison_plots(metrics_df, plots_dir)
    
    # Generate tables
    tables_dir = os.path.join(output_dir, "tables")
    generate_comparison_tables(metrics_df, tables_dir)
    
    # Generate summary report (HTML)
    summary_html = []
    summary_html.append("<html><head><title>Model Comparison Report</title>")
    summary_html.append("<style>")
    summary_html.append("body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }")
    summary_html.append("table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }")
    summary_html.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
    summary_html.append("th { background-color: #f2f2f2; }")
    summary_html.append("tr:nth-child(even) { background-color: #f9f9f9; }")
    summary_html.append("h1, h2, h3 { color: #333; }")
    summary_html.append("img { max-width: 100%; height: auto; }")
    summary_html.append(".highlight { background-color: #ffffcc; }")
    summary_html.append("</style>")
    summary_html.append("</head><body>")
    
    # Add title and date
    summary_html.append(f"<h1>IndoNLI Model Comparison Report</h1>")
    summary_html.append(f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
    
    # Add model list
    summary_html.append("<h2>Models Evaluated</h2>")
    summary_html.append("<ul>")
    for model_name in sorted(metrics_df["model_name"].unique()):
        summary_html.append(f"<li>{model_name}</li>")
    summary_html.append("</ul>")
    
    # Add test sets list
    summary_html.append("<h2>Test Sets Evaluated</h2>")
    summary_html.append("<ul>")
    for test_set in sorted(metrics_df["test_set"].unique()):
        summary_html.append(f"<li>{test_set}</li>")
    summary_html.append("</ul>")
    
    # Add overall ranking
    ranking_file = os.path.join(tables_dir, "overall_model_ranking.csv")
    if os.path.exists(ranking_file):
        ranking_df = pd.read_csv(ranking_file)
        ranking_df.set_index("model_name", inplace=True)
        
        summary_html.append("<h2>Overall Model Ranking</h2>")
        summary_html.append("<p>Based on average rank across all metrics and test sets:</p>")
        summary_html.append("<ol>")
        for model_name, row in ranking_df.sort_values("overall_rank").iterrows():
            summary_html.append(f"<li>{model_name} (Average Rank: {row['overall_rank']:.2f})</li>")
        summary_html.append("</ol>")
    
    # Add metrics tables for each test set
    summary_html.append("<h2>Detailed Metrics by Test Set</h2>")
    for test_set in sorted(metrics_df["test_set"].unique()):
        summary_html.append(f"<h3>{test_set}</h3>")
        
        # Include metrics table
        table_file = os.path.join(tables_dir, f"comparison_{test_set}_metrics.html")
        if os.path.exists(table_file):
            with open(table_file, "r") as f:
                table_html = f.read()
            summary_html.append(table_html)
        
        # Include plots
        summary_html.append("<h4>Visualizations</h4>")
        summary_html.append("<div style='display: flex; flex-wrap: wrap; justify-content: space-between;'>")
        
        # Main metrics plot
        plot_file = f"plots/comparison_{test_set}_main_metrics.png"
        if os.path.exists(os.path.join(output_dir, plot_file)):
            summary_html.append(f"<div style='flex: 1; min-width: 45%; margin: 10px;'>")
            summary_html.append(f"<img src='{plot_file}' alt='Main Metrics Comparison' />")
            summary_html.append("</div>")
        
        # Per-class F1 plot
        plot_file = f"plots/comparison_{test_set}_f1_per_class.png"
        if os.path.exists(os.path.join(output_dir, plot_file)):
            summary_html.append(f"<div style='flex: 1; min-width: 45%; margin: 10px;'>")
            summary_html.append(f"<img src='{plot_file}' alt='F1 Score Per Class' />")
            summary_html.append("</div>")
        
        summary_html.append("</div>")
    
    # Add comparison across test sets
    if len(metrics_df["test_set"].unique()) > 1:
        summary_html.append("<h2>Model Performance Across Test Sets</h2>")
        summary_html.append("<div style='display: flex; flex-wrap: wrap; justify-content: space-between;'>")
        
        for model_name in sorted(metrics_df["model_name"].unique()):
            plot_file = f"plots/comparison_{model_name}_test_sets.png"
            if os.path.exists(os.path.join(output_dir, plot_file)):
                summary_html.append(f"<div style='flex: 1; min-width: 45%; margin: 10px;'>")
                summary_html.append(f"<h3>{model_name}</h3>")
                summary_html.append(f"<img src='{plot_file}' alt='{model_name} Performance' />")
                summary_html.append("</div>")
        
        summary_html.append("</div>")
    
    # Add conclusion
    summary_html.append("<h2>Conclusion</h2>")
    
    # Find best model for accuracy
    best_models = {}
    for test_set in metrics_df["test_set"].unique():
        test_df = metrics_df[metrics_df["test_set"] == test_set]
        best_model = test_df.loc[test_df["accuracy"].idxmax()]
        best_models[test_set] = {
            "model": best_model["model_name"],
            "accuracy": best_model["accuracy"],
            "f1": best_model["f1"]
        }
    
    summary_html.append("<p>Based on the evaluation results:</p>")
    summary_html.append("<ul>")
    for test_set, best in best_models.items():
        summary_html.append(f"<li>For <strong>{test_set}</strong>, the best performing model is <strong>{best['model']}</strong> with an accuracy of {best['accuracy']:.4f} and F1 score of {best['f1']:.4f}.</li>")
    summary_html.append("</ul>")
    
    summary_html.append("</body></html>")
    
    # Write the summary report
    with open(os.path.join(output_dir, "comparison_report.html"), "w") as f:
        f.write("\n".join(summary_html))
    
    # Also save a simplified text version
    with open(os.path.join(output_dir, "comparison_report.txt"), "w") as f:
        f.write("IndoNLI Model Comparison Report\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Models Evaluated:\n")
        for model_name in sorted(metrics_df["model_name"].unique()):
            f.write(f"- {model_name}\n")
        f.write("\n")
        
        f.write("Test Sets Evaluated:\n")
        for test_set in sorted(metrics_df["test_set"].unique()):
            f.write(f"- {test_set}\n")
        f.write("\n")
        
        f.write("Best Performing Models:\n")
        for test_set, best in best_models.items():
            f.write(f"- For {test_set}, the best model is {best['model']} with accuracy {best['accuracy']:.4f}\n")
        f.write("\n")
        
        if os.path.exists(ranking_file):
            ranking_df = pd.read_csv(ranking_file)
            ranking_df.set_index("model_name", inplace=True)
            
            f.write("Overall Model Ranking:\n")
            for model_name, row in ranking_df.sort_values("overall_rank").iterrows():
                f.write(f"{model_name}: Average Rank {row['overall_rank']:.2f}\n")


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "generate_report.log")
    logger = setup_logging(log_file=log_file)
    
    logger.info(f"Arguments: {args}")
    
    # Collect metrics from all models and test sets
    logger.info("Collecting metrics from all models")
    metrics_df = collect_model_metrics(args.models_dir, args.test_sets)
    
    # Generate report
    logger.info("Generating comparison report")
    generate_report(metrics_df, args.output_dir)
    
    logger.info(f"Report generation complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
