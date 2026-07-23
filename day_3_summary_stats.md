# Day 3 Sanity Check & Summary Statistics

## 1. Dataset Validations

### Validation: Processed Hate Speech
- **Shape**: 135,556 rows, 147 columns
- **Missing Values**: > 0
  - `annotator_educ`: 17 missing
  - `annotator_income`: 103 missing
  - `annotator_ideology`: 27 missing
  - `annotator_age`: 105 missing
  - `clean_text`: 208 missing
- **Duplicate rows**: 0
### Validation: Processed HateXplain
- **Shape**: 20,148 rows, 7 columns
- **Missing Values**: 0
- **Duplicate rows**: 0
### Validation: Annotator Weights
- **Shape**: 8,165 rows, 4 columns
- **Missing Values**: 0
- **Duplicate rows**: 0
### Validation: Soft Labels
- **Shape**: 59,713 rows, 7 columns
- **Missing Values**: > 0
  - `p_offensive`: 39565 missing
- **Duplicates (by item_id)**: 0
### Validation: Training Dataset
- **Shape**: 59,713 rows, 12 columns
- **Missing Values**: > 0
  - `clean_text`: 89 missing
  - `p_offensive`: 39565 missing
- **Duplicates (by item_id)**: 0

## 2. Deep Integrity Checks

### Soft Labels bounds
- [PASS] Confidence scores strictly between 0 and 1.
- Probability columns check:
  - [PASS] `p_normal` bounds correct [0, 1]
  - [PASS] `p_hate` bounds correct [0, 1]
  - [PASS] `p_offensive` bounds correct [0, 1]

### Training Dataset Distributions
**Source Distribution**:
```text
dataset_source
hate_speech    39565
hatexplain     20148
```
**Label Distribution (Predicted MAP Label)**:
```text
predicted_label
normal       37908
hate         18414
offensive     3391
```

## 3. Global Summary Statistics

- **Total Samples (Unique Items)**: 59,713
- **Total Annotators**: 8,165
- **Average Annotator Reliability**: 0.8460
- **Average Prediction Confidence**: 0.9212
