# Day 3: Trust Pipeline Wrap-up & Project Handover

## Overview
This document serves as the final handover report for Person A's contributions to the **FairAnnot** project. Over the past three days, we have successfully ingested the raw annotation datasets, cleaned the data, run an Expectation-Maximization algorithm (Dawid-Skene) to score annotator reliability, and built a final set of "soft labels" to account for subjective disagreement.

The output of this pipeline is a single, clean `training_dataset.csv` that is ready for model consumption.

---

## Required Files (Inputs)
To run this pipeline from scratch, you need the following raw datasets placed in the `data/raw/` directory:
1. `hate_speech_dataset.csv` (Measuring Hate Speech)
2. `HateXplain.json` (HateXplain)

---

## Execution Order
If you ever need to reproduce the dataset from scratch, you can run the entire process in a single step using the wrapper script:

```bash
python run_full_pipeline.py
```

This script sequentially executes:
1. **Preprocessing**: Cleans the text (removes URLs/handles/punctuation), handles missing values, and standardizes labels.
2. **Trust Pipeline**:
   - `diadem_integration.py`: Reshapes the datasets into `(item, annotator, label)` triplets.
   - `dawid_skene.py`: Fits the EM model to calculate reliability weights and soft labels.
   - `annotation_quality.py`: Computes agreement metrics (like Cohen's Kappa, majority agreement).
   - `dataset_assembly.py`: Merges everything into the final format.

---

## Generated Outputs
The pipeline automatically generates the following files:
- `data/processed/processed_hate_speech.csv`: Cleaned Measuring Hate Speech dataset.
- `data/processed/processed_hatexplain.csv`: Cleaned HateXplain dataset.
- `trust_module/results/annotator_weights.csv`: Contains the `reliability_score` for every annotator.
- `trust_module/results/soft_labels.csv`: Contains the inferred probability distribution (`p_normal`, `p_offensive`, `p_hate`) for every item.
- `data/processed/training_dataset.csv`: The final, combined dataset.

---

## How Person B Uses These Files
Person B (Model Training & Fairness Integration) can **ignore all intermediate files** and load `training_dataset.csv` directly into their PyTorch `Dataset`.

### Key Columns for Person B:
- `clean_text`: Use this as the input to the HuggingFace Tokenizer (e.g., ModernBERT).
- `confidence`: Use this to filter out extremely low-confidence samples if needed, or to weight the loss function during training.
- `p_normal`, `p_offensive`, `p_hate`: Use these as the target variables if you are doing **soft-label training** (e.g., Cross-Entropy with probabilistic targets).
- `predicted_label`: The hard Dawid-Skene MAP (Maximum A Posteriori) label. Use this if you are doing traditional hard-label training.

Person B has already verified the forward pass using this format, so no structural changes are needed!
