# Enhanced IndoRoBERTa-Large Training for IndoNLI

This README explains the enhanced training script for finetuning the `flax-community/indonesian-roberta-large` model on the IndoNLI dataset to improve its poor performance.

## Problem Analysis

The original model showed poor performance on the IndoNLI dataset with:
- Low accuracy (34%)
- Zero precision/recall for entailment and neutral classes
- Only predicting the contradiction class

## Enhancements Implemented

The enhanced training script (`train_indoroberta_large_enhanced.py`) includes several improvements:

1. **Class Weighting**: Handles class imbalance by giving more weight to underrepresented classes
2. **Focal Loss**: Focuses more on hard examples during training
3. **Gradient Accumulation**: Effectively increases batch size without requiring more memory
4. **Early Stopping**: Prevents overfitting by stopping training when performance stops improving
5. **Data Augmentation**: Increases training data diversity through techniques like word dropout and swapping
6. **Improved Hyperparameters**: Lower learning rate, more epochs, increased warmup ratio
7. **Stratified Sampling**: Ensures balanced class distribution in train/test splits
8. **Detailed Metrics**: Provides per-class precision, recall, and F1 scores

## Usage

```bash
./train_indoroberta_large_enhanced.py \
    --data_path negfiltered_data_sample.json \
    --model_name flax-community/indonesian-roberta-large \
    --output_dir ./models/indoroberta-large-enhanced \
    --num_labels 3 \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --num_epochs 10 \
    --gradient_accumulation_steps 4 \
    --warmup_ratio 0.2 \
    --use_class_weights \
    --use_data_augmentation
```

## Key Parameters

- `--num_labels`: Number of labels (3 for IndoNLI: contradiction, neutral, entailment)
- `--learning_rate`: Reduced to 1e-5 (from 2e-5) for more stable training
- `--gradient_accumulation_steps`: Number of steps to accumulate gradients (effectively increases batch size)
- `--patience`: Number of epochs to wait for improvement before early stopping
- `--use_class_weights`: Enable class weighting for imbalanced data
- `--use_data_augmentation`: Enable data augmentation techniques

## Expected Improvements

These enhancements should help the model:
1. Learn from all classes, not just the majority class
2. Focus on difficult examples that it's currently getting wrong
3. Train more effectively with limited data
4. Avoid overfitting to the training data
5. Converge to a better solution with improved hyperparameters

## Monitoring Training

The script provides detailed logging of:
- Per-class precision, recall, and F1 scores
- Macro-averaged metrics
- Training and evaluation loss
- Early stopping information

## Comparing with Original Model

After training, you can compare the enhanced model with the original model using the evaluation script:

```bash
python evaluate_indoroberta_large_entailment.py \
    --data_path negfiltered_data_sample.json \
    --model_path ./models/indoroberta-large-enhanced/best_model
```

## Troubleshooting

If the model still performs poorly:

1. **Try different models**: Consider using `indolem/indobert-base-uncased` or other Indonesian language models
2. **Inspect the data**: Check for data quality issues or inconsistencies
3. **Adjust hyperparameters**: Try different learning rates, batch sizes, or training durations
4. **Increase model capacity**: Use a larger model or add additional layers
5. **Ensemble models**: Train multiple models and combine their predictions
