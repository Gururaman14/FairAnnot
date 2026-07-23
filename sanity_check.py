import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(__file__))
from shared import config

def write_line(f, line=""):
    print(line)
    f.write(line + "\n")

def check_dataset(name, path, f):
    write_line(f, f"### Validation: {name}")
    if not os.path.exists(path):
        write_line(f, f"- [MISSING] File missing: {path}")
        return None
    
    df = pd.read_csv(path)
    write_line(f, f"- **Shape**: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    # Missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) == 0:
        write_line(f, "- **Missing Values**: 0")
    else:
        write_line(f, "- **Missing Values**: > 0")
        for col, val in missing_cols.items():
            write_line(f, f"  - `{col}`: {val} missing")
            
    # Duplicates
    if 'item_id' in df.columns:
        dupes = df.duplicated(subset=['item_id']).sum()
        write_line(f, f"- **Duplicates (by item_id)**: {dupes}")
    else:
        dupes = df.duplicated().sum()
        write_line(f, f"- **Duplicate rows**: {dupes}")
        
    return df

def main():
    report_path = "day_3_summary_stats.md"
    with open(report_path, "w", encoding="utf-8") as f:
        write_line(f, "# Day 3 Sanity Check & Summary Statistics\n")
        
        # 1. Output files verification
        write_line(f, "## 1. Dataset Validations\n")
        hate_df = check_dataset('Processed Hate Speech', config.PROCESSED_HATE_SPEECH_PATH, f)
        hatexplain_df = check_dataset('Processed HateXplain', config.PROCESSED_HATEXPLAIN_PATH, f)
        weights_df = check_dataset('Annotator Weights', config.ANNOTATOR_WEIGHTS_PATH, f)
        soft_df = check_dataset('Soft Labels', config.SOFT_LABELS_PATH, f)
        train_df = check_dataset('Training Dataset', config.TRAINING_DATASET_PATH, f)
        
        write_line(f, "\n## 2. Deep Integrity Checks\n")
        
        # Soft Labels validation
        if soft_df is not None:
            write_line(f, "### Soft Labels bounds")
            conf_out_of_bounds = soft_df[(soft_df['confidence'] < 0.0) | (soft_df['confidence'] > 1.0)]
            if len(conf_out_of_bounds) == 0:
                write_line(f, "- [PASS] Confidence scores strictly between 0 and 1.")
            else:
                write_line(f, f"- [FAIL] Confidence scores out of bounds: {len(conf_out_of_bounds)}")
                
            write_line(f, "- Probability columns check:")
            # p_normal, p_hate, p_offensive
            for col in ['p_normal', 'p_hate', 'p_offensive']:
                valid = soft_df[soft_df[col].notna()]
                out_of_bounds = valid[(valid[col] < 0.0) | (valid[col] > 1.0)]
                if len(out_of_bounds) == 0:
                    write_line(f, f"  - [PASS] `{col}` bounds correct [0, 1]")
                else:
                    write_line(f, f"  - [FAIL] `{col}` out of bounds: {len(out_of_bounds)}")
        
        # Training dataset validation
        if train_df is not None:
            write_line(f, "\n### Training Dataset Distributions")
            write_line(f, "**Source Distribution**:")
            write_line(f, "```text")
            write_line(f, train_df['dataset_source'].value_counts().to_string())
            write_line(f, "```")
            
            write_line(f, "**Label Distribution (Predicted MAP Label)**:")
            write_line(f, "```text")
            write_line(f, train_df['predicted_label'].value_counts().to_string())
            write_line(f, "```")
        
        write_line(f, "\n## 3. Global Summary Statistics\n")
        if train_df is not None and weights_df is not None:
            write_line(f, f"- **Total Samples (Unique Items)**: {train_df.shape[0]:,}")
            write_line(f, f"- **Total Annotators**: {weights_df['annotator_id'].nunique():,}")
            write_line(f, f"- **Average Annotator Reliability**: {weights_df['reliability_score'].mean():.4f}")
            write_line(f, f"- **Average Prediction Confidence**: {train_df['confidence'].mean():.4f}")

if __name__ == '__main__':
    main()
