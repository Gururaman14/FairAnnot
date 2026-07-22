# cspell:disable
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from shared import config


def assemble_training_dataset(hate_speech_df, hatexplain_df, hs_soft_labels, hx_soft_labels):
    hs = hate_speech_df[['comment_id', 'text', 'clean_text', 'text_length', 'word_count', 'label']].copy()
    hs = hs.rename(columns={'comment_id': 'item_id', 'label': 'majority_label'})
    hs['item_id'] = hs['item_id'].astype(str)
    hs['dataset_source'] = 'hate_speech'

    hs = hs.drop_duplicates(subset='item_id').reset_index(drop=True)

    hs_merged = hs.merge(hs_soft_labels, on='item_id', how='left')
    if 'p_offensive' not in hs_merged.columns:
        hs_merged['p_offensive'] = float('nan')

    hx = hatexplain_df[['post_id', 'text', 'clean_text', 'text_length', 'word_count', 'label']].copy()
    hx = hx.rename(columns={'post_id': 'item_id', 'label': 'majority_label'})
    hx['dataset_source'] = 'hatexplain'

    hx_merged = hx.merge(hx_soft_labels, on='item_id', how='left')

    COLS = ['item_id', 'dataset_source', 'text', 'clean_text',
            'text_length', 'word_count', 'majority_label',
            'predicted_label', 'confidence',
            'p_normal', 'p_offensive', 'p_hate']

    for col in COLS:
        if col not in hs_merged.columns:
            hs_merged[col] = float('nan')
        if col not in hx_merged.columns:
            hx_merged[col] = float('nan')

    training_df = pd.concat([hs_merged[COLS], hx_merged[COLS]], ignore_index=True)

    return training_df


def save_training_dataset(training_df):
    training_df.to_csv(config.TRAINING_DATASET_PATH, index=False)
    print(f"\n  Training dataset saved: {config.TRAINING_DATASET_PATH}")
    print(f"  Shape: {training_df.shape[0]:,} rows × {training_df.shape[1]} cols")
    print(f"  Source breakdown:\n{training_df['dataset_source'].value_counts().to_string()}")
    print(f"  Label breakdown:\n{training_df['predicted_label'].value_counts().to_string()}")
