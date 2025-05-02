# Model Explanations

This document provides brief explanations for the models benchmarked using the `scripts/benchmark_models.py` script. The model types are identified based on the normalization logic within the script.

## 1. Indo-roBERTa Models

This section describes the Indo-roBERTa models evaluated by the benchmark script. The script (`scripts/benchmark_models.py`) distinguishes between "base" and potentially "fine-tuned" models primarily based on the **naming convention of the directories** where the model checkpoints are discovered (e.g., `logs/indo-roberta-base` vs. `logs/indo-roberta`).

### 1.1 Indo-roBERTa (Base) - *Script Interpretation*

*   **Identification by Script:** Checkpoints found within a directory whose name ends with `-base` (e.g., `indo-roberta-base`) are treated by the script as representing the "base" model evaluation.
*   **Intended Meaning:** This typically corresponds to evaluating the foundational, pre-trained Indonesian RoBERTa model (`cahya/roberta-base-indonesian-1.5G`) directly on the NLI task, possibly with only a classification head added and *without* specific fine-tuning on the IndoNLI dataset itself. It serves as a baseline.
*   **Underlying Model:** Assumed to be `cahya/roberta-base-indonesian-1.5G`.
*   **Architecture:** Standard RoBERTa architecture pre-trained on a large Indonesian corpus.
*   **Use Case in Benchmark:** Represents the model's general language understanding capabilities applied to NLI before task-specific adaptation.

### 1.2 Indo-roBERTa (Fine-tuned) - *Script Interpretation*

*   **Identification by Script:** Checkpoints found within a directory whose name does *not* end with `-base` (e.g., `indo-roberta`) are treated by the script as representing the potentially fine-tuned model.
*   **Intended Meaning:** This typically corresponds to evaluating the Indo-roBERTa base model *after* it has undergone additional training (fine-tuning) specifically on the Indonesian NLI (IndoNLI) dataset. This process adapts the model's parameters to better perform the NLI task.
*   **Underlying Model:** Assumed to start from `cahya/roberta-base-indonesian-1.5G` and then be fine-tuned on IndoNLI.
*   **Architecture:** Same RoBERTa architecture, but weights are updated during fine-tuning.
*   **Use Case in Benchmark:** Represents the model optimized specifically for the IndoNLI task, expected to show improved performance compared to the base model evaluation.

*   **Hugging Face Reference (Pre-trained Base):** [`cahya/roberta-base-indonesian-1.5G`](https://huggingface.co/cahya/roberta-base-indonesian-1.5G)
*   **Details on Base Model:** The referenced Hugging Face model card for the base model lacks detailed information about its specific training data or methodology beyond the standard RoBERTa approach.

## 2. Sentence-BERT (Generic)

*   **Description:** Sentence-BERT (SBERT) is a modification of standard pre-trained Transformer models (like BERT, RoBERTa) specifically designed to generate semantically meaningful, fixed-size embeddings for entire sentences. It addresses a key limitation of standard BERT: finding the most similar sentence pairs requires feeding all possible pairs into the network, which is computationally very expensive (quadratic complexity). SBERT significantly reduces this cost by generating independent sentence embeddings that can be efficiently compared using cosine similarity (linear complexity).
*   **Architecture:** SBERT typically employs a **Siamese** or **Triplet** network structure during fine-tuning. In a Siamese setup, two identical, weight-sharing BERT models process two sentences independently. The outputs (token embeddings) are then passed through a **pooling layer** (commonly mean pooling, but max or CLS pooling are also options) to obtain fixed-size sentence embeddings. These embeddings are then used in the training objective.
*   **Training:** SBERT is fine-tuned on sentence-pair datasets using objectives tailored for semantic similarity. Common objectives include:
    *   **Classification Objective:** Training a classifier on top of the combined sentence embeddings (e.g., concatenation, difference, product) for tasks like NLI.
    *   **Regression Objective:** Training the model to predict the similarity score (e.g., from STS benchmarks) between sentence embeddings using cosine similarity.
    *   **Triplet Objective:** Training the model to ensure that an "anchor" sentence embedding is closer to a "positive" (similar) sentence embedding than to a "negative" (dissimilar) sentence embedding.
    *   **Contrastive Objectives (like MultipleNegativesRankingLoss):** Training the model to distinguish positive pairs from many negative pairs within a batch, pushing positive pairs closer and negative pairs further apart in the embedding space.
*   **Use Case:** Highly effective for large-scale semantic similarity comparison, semantic search, clustering, paraphrase mining, and information retrieval tasks.
*   **Example Implementation (Indonesian):** The benchmarking script references `firqaaa/indo-sentence-bert-base` ([HF Link](https://huggingface.co/firqaaa/indo-sentence-bert-base)), which is an SBERT model for Indonesian using a BERT base, mean pooling, and trained with Multiple Negatives Ranking Loss.

## 3. Sentence-BERT-Simple

*   **Description (Based on `src/models/sentence_bert_model.py`):** This variant uses the standard Sentence-BERT embedding generation (e.g., `firqaaa/indo-sentence-bert-base` with mean or CLS pooling) but employs a specific **"Simple" classifier** on top of the resulting sentence embedding.
    *   **Classifier Architecture:** The classifier is a two-layer MLP: `Linear -> ReLU -> Dropout -> Linear`.
    *   **Input:** It takes the pooled sentence embedding (mean or CLS) as input.
*   **Use Case:** Represents a baseline approach where a relatively simple classifier is used on top of pre-computed or fine-tuned sentence embeddings for the NLI task.

## 4. Sentence-BERT-Proper

*   **Description (Based on `src/models/sentence_bert_model.py`):** This variant uses the standard Sentence-BERT embedding generation but employs a more complex, configurable **"Proper" classifier** on top.
    *   **Classifier Architecture:** The classifier is a configurable multi-layer MLP, with each layer consisting of `Linear -> LayerNorm -> ReLU -> Dropout`, followed by a final `Linear` layer for classification. The number of layers and hidden dimensions are configurable.
    *   **Input:** It attempts to use the standard BERT `pooler_output` (typically derived from the `[CLS]` token after further processing) as input if available. If not, it falls back to using the pooled sentence embedding (mean or CLS) like the "Simple" variant.
*   **Use Case:** Represents a more sophisticated classification approach on top of sentence embeddings, potentially capturing more complex relationships in the NLI task compared to the "Simple" classifier.

*Note: The primary distinction between "Sentence-BERT-Simple" and "Sentence-BERT-Proper" **within this project's code** lies in the architecture of the classification head applied *after* the sentence embedding is generated, not necessarily in the SBERT fine-tuning process itself.*