import json
import pandas as pd
from . import config

def load_hate_speech_dataset():
    """Load the raw hate speech dataset."""
    return pd.read_csv(config.HATE_SPEECH_DATASET_PATH)

def load_hatexplain_dataset():
    """Load the raw HateXplain dataset."""
    with open(config.HATEXPLAIN_DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    records = []
    for post_id, post_data in data.items():
        text = " ".join(post_data.get('post_tokens', []))
        
        # Determine the majority label from annotators
        annotators = post_data.get('annotators', [])
        labels = tuple(ann['label'] for ann in annotators)
        
        records.append({
            'post_id': post_id,
            'text': text,
            'labels': labels
        })
    return pd.DataFrame(records)

def load_processed_dataset(dataset_name):
    """Load a processed dataset by name ('hate_speech' or 'hatexplain')."""
    if dataset_name == 'hate_speech':
        return pd.read_csv(config.PROCESSED_HATE_SPEECH_PATH)
    elif dataset_name == 'hatexplain':
        return pd.read_csv(config.PROCESSED_HATEXPLAIN_PATH)
    else:
        raise ValueError("Invalid dataset name. Choose 'hate_speech' or 'hatexplain'.")
