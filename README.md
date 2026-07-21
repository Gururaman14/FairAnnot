# Trust Module

This module analyses **annotator-level trust and reliability** in hate speech datasets.
It builds on the preprocessed outputs from Day 1 and will eventually produce per-annotator trust scores, reliability metrics, and fairness diagnostics.

## Day 1 - Completed 

### What was done

 Task | Status 
 Load `hate_speech_dataset.csv` (135,556 annotation rows, 143 cols) 
 Load `HateXplain.json` (~20 K posts, 3 annotators each)
 Data exploration: shape, dtypes, missing values, duplicates 
 Label distribution analysis + class imbalance identified
 Text length and word count distributions examined
 Duplicate removal (none found)
 Missing value handling (none found; guard in place)
 Label standardisation - Hate Speech: `hatespeech` column → `'hate'` / `'normal'` 
 Label standardisation - HateXplain: majority vote over 3 annotators → `'hate'` / `'offensive'` / `'normal'` 
 Text cleaning: lowercase, remove URLs/mentions/hashtags/punctuation
 Feature engineering: `clean_text`, `text_length`, `word_count`
 Visualisations saved to `trust_module/results/plots/`
 Processed datasets saved to `data/processed/`
 Reusable functions extracted to `shared/preprocessing.py`
 Reusable loaders extracted to `shared/dataset_loader.py`

### Key Findings

- Class imbalance: Both datasets are skewed towards 'normal' labels. This must be addressed in modelling (weighted loss, stratified splits, F1 reporting).
- No missing data: Both datasets are clean exports; no imputation required.
- No duplicates: Zero exact duplicate rows in either dataset.
- Text length difference: HateXplain posts are short (social media), Hate Speech texts vary more widely (Twitter + Reddit mix). Cross-dataset generalisation needs careful consideration.

### Output Files

 File & Description
 `data/processed/processed_hate_speech.csv` - Cleaned Measuring Hate Speech dataset
 `data/processed/processed_hatexplain.csv` - Cleaned HateXplain dataset
 `trust_module/results/plots/label_distribution.png` - Label distribution per dataset
 `trust_module/results/plots/missing_values.png` - Missing value chart
 `trust_module/results/plots/text_length_distribution.png` - Text length histogram

