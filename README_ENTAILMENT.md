# Finetuning indonesian-roberta-large for Entailment Classification

This repository contains scripts to finetune the `flax-community/indonesian-roberta-large` model on a custom entailment dataset. The model is trained to determine whether a generated text entails (is consistent with) a source text.

## Dataset Format

The dataset should be in JSON format with the following structure:

```json
{
    "id": {
        "0": "item_1",
        "1": "item_2",
        ...
    },
    "source_text": {
        "0": "Source text 1",
        "1": "Source text 2",
        ...
    },
    "generated_indonesian": {
        "0": "Generated text 1",
        "1": "Generated text 2",
        ...
    },
    "score": {
        "0": 0,
        "1": 1,
        ...
    }
}
```

Where:
- `source_text`: The original text
- `generated_indonesian`: The generated text to check for entailment
- `score`: The entailment label (0 for non-entailment, 1 for entailment)

## Scripts

### 1. Training Script

The `train_indoroberta_large_entailment.py` script finetunes the model on the custom dataset.

```bash
python train_indoroberta_large_entailment.py \
    --data_path negfiltered_data_sample.json \
    --model_name flax-community/indonesian-roberta-large \
    --output_dir ./models/indoroberta-large-entailment \
    --max_length 512 \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --num_epochs 5 \
    --seed 42 \
    --test_size 0.2 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01
```

### 2. Evaluation Script

The `evaluate_indoroberta_large_entailment.py` script evaluates the finetuned model on a dataset.

```bash
python evaluate_indoroberta_large_entailment.py \
    --data_path negfiltered_data_sample.json \
    --model_path ./models/indoroberta-large-entailment/best_model \
    --output_file evaluation_results.json \
    --max_length 512 \
    --batch_size 8
```

### 3. Prediction Script

The `predict_entailment.py` script uses the finetuned model to predict entailment for new data.

#### For a single text/hypothesis pair:

```bash
python predict_entailment.py \
    --model_path ./models/indoroberta-large-entailment/best_model \
    --text "Source text" \
    --hypothesis "Generated text" \
    --max_length 512
```

#### For a file containing multiple pairs:

```bash
python predict_entailment.py \
    --model_path ./models/indoroberta-large-entailment/best_model \
    --input_file new_data.json \
    --output_file predictions.json \
    --max_length 512
```

## Requirements

- Python 3.6+
- PyTorch
- Transformers
- Pandas
- NumPy
- scikit-learn
- tqdm

## Installation

```bash
pip install torch transformers pandas numpy scikit-learn tqdm
```

## Model Output

The finetuned model will be saved in the specified output directory:

```
./models/indoroberta-large-entailment/
├── best_model/         # Best model based on validation accuracy
├── checkpoint-1/       # Checkpoint after epoch 1
├── checkpoint-2/       # Checkpoint after epoch 2
...
└── final_model/        # Final model after all epochs
```

## Prediction Output

The prediction script outputs:
- Prediction: 0 (non-entailment) or 1 (entailment)
- Probability: Confidence score for the prediction
- Entailment probability: Probability of entailment (class 1)
- Non-entailment probability: Probability of non-entailment (class 0)

## Example Usage

1. Train the model:
```bash
./train_indoroberta_large_entailment.py --data_path negfiltered_data_sample.json
```

2. Evaluate the model:
```bash
./evaluate_indoroberta_large_entailment.py --data_path negfiltered_data_sample.json --model_path ./models/indoroberta-large-entailment/best_model
```

3. Make predictions:
```bash
./predict_entailment.py --model_path ./models/indoroberta-large-entailment/best_model --text "Source text" --hypothesis "Generated text"
```
