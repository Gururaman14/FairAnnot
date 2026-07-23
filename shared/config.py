import os
from pathlib import Path

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FINAL_DATA_DIR = DATA_DIR / "final"

# Trust Module Directories
TRUST_MODULE_DIR = BASE_DIR / "trust_module"
TRUST_RESULTS_DIR = TRUST_MODULE_DIR / "results"
TRUST_PLOTS_DIR = TRUST_RESULTS_DIR / "plots"

# Create Directories
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
TRUST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TRUST_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Paths
HATE_SPEECH_DATASET_PATH = RAW_DATA_DIR / "hate_speech_dataset.csv"
HATEXPLAIN_DATASET_PATH = RAW_DATA_DIR / "HateXplain.json"

# Processed Dataset Paths
PROCESSED_HATE_SPEECH_PATH = PROCESSED_DATA_DIR / "processed_hate_speech.csv"
PROCESSED_HATEXPLAIN_PATH = PROCESSED_DATA_DIR / "processed_hatexplain.csv"
TRAINING_DATASET_PATH = PROCESSED_DATA_DIR / "training_dataset.csv"
FINAL_TRAINING_DATASET_PATH = FINAL_DATA_DIR / "training_dataset.csv"
PROCESSED_ANNOTATOR_WEIGHTS_PATH = PROCESSED_DATA_DIR / "annotator_weights.csv"
PROCESSED_SOFT_LABELS_PATH = PROCESSED_DATA_DIR / "soft_labels.csv"

# Trust Module Output Paths
ANNOTATOR_WEIGHTS_PATH = TRUST_RESULTS_DIR / "annotator_weights.csv"
SOFT_LABELS_PATH = TRUST_RESULTS_DIR / "soft_labels.csv"

# Explainability and Evaluation Output Paths
EXPLAINABILITY_DIR = BASE_DIR / "explainability"
EXPLAINABILITY_RESULTS_DIR = EXPLAINABILITY_DIR / "results"
EVALUATION_DIR = BASE_DIR / "evaluation"
EVALUATION_RESULTS_DIR = EVALUATION_DIR / "results"

EXPLAINABILITY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Model Configuration
MODEL_NAME = "answerdotai/ModernBERT-base"

# Training Configuration
BATCH_SIZE = 16
MAX_LENGTH = 128
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3

# Common Project Constants
RANDOM_SEED = 42
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
