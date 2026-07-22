# Day 2: Advanced Data Engineering & Fairness Constraints

## What we're doing today
Today is all about taking the cleaned, trusted data we prepared on Day 1 and making it ready for the actual models. Yesterday, we figured out which annotators to trust and built our soft labels. Today, we're building the infrastructure to properly load this data, tokenize it, and set up the fairness constraints so our models don't just learn biases.

## Key Objectives
1. **PyTorch Data Loaders:** We need a solid `Dataset` class to handle both the Measuring Hate Speech and HateXplain data efficiently.
2. **Tokenization:** Getting the text properly tokenized so the BERT/RoBERTa models can actually process it. We'll be using HuggingFace's tokenizers for this.
3. **Fairness Setup:** We need to define the protected attributes (like race, gender, religion) so we can penalize the model later if it starts behaving unfairly towards specific groups.
4. **Data Splitting:** Properly dividing our merged dataset into train, validation, and test splits while making sure the distributions stay balanced.

## File Structure for Today
We'll mostly be working inside the `data_engineering` module and starting on some new notebooks.

```text
FairAnnot/
├── data_engineering/
│   ├── dataloader.py        # PyTorch Dataset & DataLoader logic
│   ├── tokenization.py      # HuggingFace tokenizers setup
│   └── fairness_splits.py   # Code to handle protected attributes and splits
├── notebooks/
│   └── 02_data_loaders.ipynb # Testing our loaders and tokenization
```

## Notes
- Let's keep the code clean and functional, just like Day 1.
- Remember to save any new intermediate datasets into `data/processed/`.
- If we hit any memory issues with the tokenizers on the larger Measuring Hate Speech dataset, we might need to adjust the max sequence lengths.

Let's get into it!
