import os
from pathlib import Path

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ensure processed directory exists
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Paths
HATE_SPEECH_DATASET_PATH = RAW_DATA_DIR / "hate_speech_dataset.csv"
HATEXPLAIN_DATASET_PATH = RAW_DATA_DIR / "HateXplain.json"

# Output Paths
PROCESSED_HATE_SPEECH_PATH = PROCESSED_DATA_DIR / "processed_hate_speech.csv"
PROCESSED_HATEXPLAIN_PATH = PROCESSED_DATA_DIR / "processed_hatexplain.csv"

# Common Project Constants
RANDOM_SEED = 42
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
