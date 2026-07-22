# cspell:disable
import json
import ast
import pandas as pd

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from shared import config


_HATE_SPEECH_LABEL_MAP = {
    0.0: 'normal',
    1.0: 'hate',
    0:   'normal',
    1:   'hate',
}

_HATEXPLAIN_LABEL_MAP = {
    'hatespeech': 'hate',
    'normal':     'normal',
    'offensive':  'offensive',
}


def prepare_hate_speech_annotations(df):
    hs_numeric = pd.to_numeric(df['hatespeech'], errors='coerce')
    ann = pd.DataFrame({
        'item_id':      df['comment_id'].astype(str),
        'annotator_id': df['annotator_id'].astype(str),
        'label':        hs_numeric.apply(
                            lambda x: 'hate' if pd.notna(x) and x >= 1 else (
                                      'normal' if pd.notna(x) and x == 0 else None)
                        ),
    })

    before = len(ann)
    ann = ann.dropna(subset=['label']).reset_index(drop=True)
    dropped = before - len(ann)
    if dropped:
        print(f"  [hate_speech] Dropped {dropped} rows with unmapped labels.")

    print(f"  [hate_speech] Annotation triplets: {len(ann):,}")
    return ann


def prepare_hatexplain_annotations():
    with open(config.HATEXPLAIN_DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    records = []
    for post_id, post_data in data.items():
        for ann in post_data.get('annotators', []):
            raw_label = ann['label'].lower().strip()
            label = _HATEXPLAIN_LABEL_MAP.get(raw_label)
            if label is None:
                continue
            records.append({
                'item_id':      post_id,
                'annotator_id': f"hx_{ann['annotator_id']}",
                'label':        label,
            })

    ann_df = pd.DataFrame(records)
    print(f"  [hatexplain] Annotation triplets: {len(ann_df):,}")
    return ann_df
