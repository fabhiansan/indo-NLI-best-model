# Training indonesian-roberta-large on IndoNLI Dataset

This README explains how to train the `flax-community/indonesian-roberta-large` model on the IndoNLI dataset.

## Prerequisites

- Python 3.6+
- PyTorch
- Transformers
- Datasets
- Other dependencies as specified in the project

## Configuration

The configuration for the model is defined in `configs/indo_roberta_large.yaml`. You can modify this file to adjust hyperparameters such as:

- Batch size
- Learning rate
- Number of epochs
- Sequence length
- etc.

## Training

To train the model, simply run:

```bash
python train_indo_roberta_large.py
```

Or make it executable and run:

```bash
chmod +x train_indo_roberta_large.py
./train_indo_roberta_large.py
```

## Model Output

The trained model will be saved in the directory specified in the configuration file:

```
./models/indo-roberta-large
```

Logs will be saved in:

```
./logs/indo-roberta-large
```

Reports will be saved in:

```
./reports/indo-roberta-large
```

## Evaluation

The model will be evaluated on the validation set during training. The best model will be saved based on the evaluation metrics.

## Using the Trained Model

After training, you can use the model for inference by loading it from the saved directory:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_path = "./models/indo-roberta-large/final"  # or use a specific epoch
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Use the model for inference
inputs = tokenizer("premise", "hypothesis", return_tensors="pt")
outputs = model(**inputs)
```
