import pandas as pd
import re

# Text Cleaning 

def clean_text(text):
    """Clean input text: lowercase, strip URLs, mentions, hashtags, and punctuation."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

#  Duplicate Handling 

def remove_duplicates(df, subset=None):
    """Remove duplicate rows. Returns cleaned DataFrame and count of removed rows."""
    initial = df.shape[0]
    df = df.drop_duplicates(subset=subset).reset_index(drop=True)
    removed = initial - df.shape[0]
    print(f"  Duplicates removed: {removed}")
    return df, removed

# Label Standardisation 

def standardize_labels_hate_speech(df):
    """
    Standardise labels for the Measuring Hate Speech dataset.

    The dataset contains a continuous 'hate_speech_score' column and a binary
    'hatespeech' column (1 = hate speech, 0 = not).  We use 'hatespeech'
    directly as it is the authoritative per-annotation binary label.
    """
    df = df.copy()
    df['label'] = df['hatespeech'].apply(lambda x: 'hate' if float(x) >= 1 else 'normal')
    return df

def standardize_labels_hatexplain(df):
    """
    Standardise labels for the HateXplain dataset.

    Each post has up to 3 annotator labels stored as a tuple.  We derive the
    final label via majority vote and map raw annotator strings to canonical
    values: 'hate', 'offensive', 'normal'.
    """
    label_mapping = {
        'hatespeech': 'hate',
        'normal': 'normal',
        'offensive': 'offensive'
    }

    def majority_vote(labels_tuple):
        if not isinstance(labels_tuple, (list, tuple)) or len(labels_tuple) == 0:
            return 'unknown'
        return max(set(labels_tuple), key=labels_tuple.count)

    df = df.copy()
    df['label'] = df['labels'].apply(majority_vote)
    df['label'] = df['label'].str.lower().map(label_mapping).fillna('unknown')
    return df

#  Feature Engineering 

def calculate_text_length(df, text_column='clean_text'):
    """Add a 'text_length' column (character count)."""
    df = df.copy()
    df['text_length'] = df[text_column].astype(str).apply(len)
    return df

def calculate_word_count(df, text_column='clean_text'):
    """Add a 'word_count' column (whitespace-split token count)."""
    df = df.copy()
    df['word_count'] = df[text_column].astype(str).apply(lambda x: len(x.split()))
    return df

#  Missing Value Handling 

def handle_missing_text(df, text_column='text'):
    """Fill NaN values in the text column with an empty string."""
    missing_count = df[text_column].isna().sum()
    df = df.copy()
    df[text_column] = df[text_column].fillna('')
    print(f"  Missing values filled in '{text_column}': {missing_count}")
    return df, int(missing_count)

#  Preprocessing Summary 

def print_summary(name, initial_rows, duplicates_removed, missing_handled,
                  final_df, output_path):
    """Print a concise preprocessing summary for one dataset."""
    print(f"\n{'='*55}")
    print(f"  Preprocessing Summary — {name}")
    print(f"{'='*55}")
    print(f"  Initial samples        : {initial_rows:,}")
    print(f"  Duplicates removed     : {duplicates_removed:,}")
    print(f"  Missing values filled  : {missing_handled:,}")
    print(f"  Final dataset size     : {final_df.shape[0]:,} rows × {final_df.shape[1]} cols")
    print(f"  Label distribution     :\n{final_df['label'].value_counts().to_string()}")
    print(f"  Saved to               : {output_path}")
    print(f"{'='*55}\n")
