import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import os


INPUT_DF_PATH = "indonesia_amr_perturber/outputs/train_output_3-checkpoint-4_main_corrected.json" 

MODEL_PATH = "indo-NLI-best-model/models/indo-roberta-base/epoch-4" 
# TODO: Replace with the desired path for the output DataFrame
OUTPUT_DF_PATH = "indonesia_amr_perturber/outputs/test_indoroberta.csv" 

# Base model name used for tokenizer and model architecture
BASE_MODEL_NAME = "indolem/indobert-base-uncased" 
MAX_LENGTH = 128
BATCH_SIZE = 2 # Adjust based on your system's memory

# --- Helper Functions ---

def mean_pooling(model_output, attention_mask):
    """
    Mean pooling to get sentence embeddings.
    """
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def compute_similarity(embedding1, embedding2):
    """
    Compute cosine similarity between two embeddings.
    """
    return F.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0)).item()

# --- Main Script ---

def main():
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model
    print(f"Loading tokenizer from {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print(f"Loading model architecture from {MODEL_PATH}")
    model = AutoModel.from_pretrained(MODEL_PATH)

    # Check if model path exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model directory not found at {MODEL_PATH}")
        print("Please update MODEL_PATH in the script with the correct path to the saved model directory.")
        return

    # Model architecture and weights are loaded by AutoModel.from_pretrained
    # if MODEL_PATH is a directory containing saved model files (e.g., pytorch_model.bin or model.safetensors)

    model.to(device)
    model.eval()

    # Load DataFrame
    if not os.path.exists(INPUT_DF_PATH):
        print(f"Error: Input DataFrame not found at {INPUT_DF_PATH}")
        print("Please update INPUT_DF_PATH in the script with the correct path.")
        return

    print(f"Loading DataFrame from {INPUT_DF_PATH}")
    df = pd.read_json(INPUT_DF_PATH)

    # Check if required columns exist
    required_columns = ["target_summary", "text_from_amr"]
    if not all(col in df.columns for col in required_columns):
        print(f"Error: Input DataFrame must contain columns: {required_columns}")
        return

    # Initialize list to store scores
    similarity_scores = []

    print("Processing DataFrame and computing similarity scores...")
    # Process in batches
    for i in range(0, 10, BATCH_SIZE):
        batch_df = df.iloc[i : i + BATCH_SIZE]
        
        # Get texts from batch
        texts1 = batch_df["target_summary"].tolist()
        texts2 = batch_df["text_from_amr"].tolist()

        # Tokenize texts
        encoded_input1 = tokenizer(texts1, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors='pt').to(device)
        encoded_input2 = tokenizer(texts2, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors='pt').to(device)

        # Compute embeddings
        with torch.no_grad():
            model_output1 = model(**encoded_input1)
            model_output2 = model(**encoded_input2)

        # Perform mean pooling
        sentence_embeddings1 = mean_pooling(model_output1, encoded_input1['attention_mask'])
        sentence_embeddings2 = mean_pooling(model_output2, encoded_input2['attention_mask'])

        # Compute similarity for each pair in the batch
        for j in range(len(batch_df)):
            score = compute_similarity(sentence_embeddings1[j], sentence_embeddings2[j])
            similarity_scores.append(score)

        print(f"Processed batch {i // BATCH_SIZE + 1}/{(len(df) + BATCH_SIZE - 1) // BATCH_SIZE}")


    # Add scores to DataFrame
    df["similarity_score"] = similarity_scores

    # Save the updated DataFrame
    print(f"Saving updated DataFrame to {OUTPUT_DF_PATH}")
    df.to_csv(OUTPUT_DF_PATH, index=False)

    print("Script finished. Similarity scores added to the output DataFrame.")