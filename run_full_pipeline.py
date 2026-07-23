import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from shared import config
from shared.dataset_loader import load_hate_speech_dataset, load_hatexplain_dataset
from shared.preprocessing import (
    clean_text, remove_duplicates, standardize_labels_hate_speech,
    standardize_labels_hatexplain, calculate_text_length,
    calculate_word_count, handle_missing_text
)
from trust_module.run_trust_pipeline import run_pipeline

def main():
    print("="*60)
    print("  RUNNING FULL PIPELINE (PREPROCESSING + TRUST)")
    print("="*60)
    
    print("\n[1] Loading raw datasets...")
    hate_df = load_hate_speech_dataset()
    hatexplain_df = load_hatexplain_dataset()

    print("\n[2] Preprocessing Measuring Hate Speech...")
    hate_df, _   = remove_duplicates(hate_df)
    hate_df, _ = handle_missing_text(hate_df, 'text')
    hate_df             = standardize_labels_hate_speech(hate_df)
    hate_df['clean_text'] = hate_df['text'].apply(clean_text)
    hate_df             = calculate_text_length(hate_df, 'clean_text')
    hate_df             = calculate_word_count(hate_df, 'clean_text')

    print("\n[3] Preprocessing HateXplain...")
    hatexplain_df, _   = remove_duplicates(hatexplain_df)
    hatexplain_df, _ = handle_missing_text(hatexplain_df, 'text')
    hatexplain_df             = standardize_labels_hatexplain(hatexplain_df)
    hatexplain_df['clean_text'] = hatexplain_df['text'].apply(clean_text)
    hatexplain_df             = calculate_text_length(hatexplain_df, 'clean_text')
    hatexplain_df             = calculate_word_count(hatexplain_df, 'clean_text')

    print("\n[4] Saving processed datasets...")
    hate_df.to_csv(config.PROCESSED_HATE_SPEECH_PATH, index=False)
    hatexplain_df.to_csv(config.PROCESSED_HATEXPLAIN_PATH, index=False)

    print("\n[5] Calling Trust Pipeline...")
    run_pipeline(ds_n_iter=30, hx_n_iter=50, verbose=True)

if __name__ == '__main__':
    main()
