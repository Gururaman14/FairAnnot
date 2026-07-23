# cspell:disable
import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared import config
from trust_module.dawid_skene import DawidSkene
from trust_module.annotation_quality import (
    compute_quality_report,
    print_quality_summary,
)
from trust_module.diadem_integration import (
    prepare_hate_speech_annotations,
    prepare_hatexplain_annotations,
)
from trust_module.dataset_assembly import assemble_training_dataset, save_training_dataset


def run_pipeline(ds_n_iter=30, hx_n_iter=50, verbose=True):

    print("\n" + "="*60)
    print("  TRUST PIPELINE - Dawid-Skene Annotator Reliability")
    print("="*60)

    print("\n[1/6] Loading processed datasets...")
    hate_df = pd.read_csv(config.PROCESSED_HATE_SPEECH_PATH)
    hatexplain_df = pd.read_csv(config.PROCESSED_HATEXPLAIN_PATH)
    print(f"  Hate Speech : {hate_df.shape[0]:,} rows")
    print(f"  HateXplain  : {hatexplain_df.shape[0]:,} rows")

    print("\n[2/6] Preparing annotation triplets...")
    hs_ann = prepare_hate_speech_annotations(hate_df)
    hx_ann = prepare_hatexplain_annotations()

    print(f"\n[3/6] Fitting Dawid-Skene — Measuring Hate Speech (max {ds_n_iter} iter)...")
    ds_hate = DawidSkene(n_iter=ds_n_iter, verbose=verbose)
    ds_hate.fit(hs_ann)

    print(f"\n[3/6] Fitting Dawid-Skene — HateXplain (max {hx_n_iter} iter)...")
    ds_hx = DawidSkene(n_iter=hx_n_iter, verbose=verbose)
    ds_hx.fit(hx_ann)

    print("\n[4/6] Computing annotation quality metrics...")
    hs_weights = ds_hate.get_annotator_weights()
    hx_weights = ds_hx.get_annotator_weights()

    hs_quality = compute_quality_report(hs_ann, hs_weights, 'hate_speech')
    hx_quality = compute_quality_report(hx_ann, hx_weights, 'hatexplain')

    print_quality_summary(hs_quality, 'Measuring Hate Speech')
    print_quality_summary(hx_quality, 'HateXplain')

    print("\n[5/6] Saving annotator weights and soft labels...")

    hs_weights['dataset'] = 'hate_speech'
    hx_weights['dataset'] = 'hatexplain'
    all_weights = pd.concat([hs_weights, hx_weights], ignore_index=True)
    all_weights.to_csv(config.ANNOTATOR_WEIGHTS_PATH, index=False)
    all_weights.to_csv(config.PROCESSED_ANNOTATOR_WEIGHTS_PATH, index=False)
    print(f"  Saved: {config.ANNOTATOR_WEIGHTS_PATH}")

    hs_soft = ds_hate.get_soft_labels()
    hs_soft['dataset'] = 'hate_speech'
    if 'p_offensive' not in hs_soft.columns:
        hs_soft['p_offensive'] = float('nan')

    hx_soft = ds_hx.get_soft_labels()
    hx_soft['dataset'] = 'hatexplain'
    if 'p_offensive' not in hx_soft.columns:
        hx_soft['p_offensive'] = float('nan')

    SOFT_COLS = ['item_id', 'dataset', 'p_normal', 'p_offensive', 'p_hate',
                 'predicted_label', 'confidence']

    for df in [hs_soft, hx_soft]:
        for col in SOFT_COLS:
            if col not in df.columns:
                df[col] = float('nan')

    all_soft = pd.concat([hs_soft[SOFT_COLS], hx_soft[SOFT_COLS]], ignore_index=True)
    all_soft.to_csv(config.SOFT_LABELS_PATH, index=False)
    all_soft.to_csv(config.PROCESSED_SOFT_LABELS_PATH, index=False)
    print(f"  Saved: {config.SOFT_LABELS_PATH}")

    print("\n[6/6] Assembling training_dataset.csv...")
    training_df = assemble_training_dataset(
        hate_df, hatexplain_df, hs_soft, hx_soft
    )
    save_training_dataset(training_df)

    print("\n" + "="*60)
    print("  Trust Pipeline complete.")
    print("="*60 + "\n")

    return {
        'hate_speech_model': ds_hate,
        'hatexplain_model':  ds_hx,
        'annotator_weights': all_weights,
        'soft_labels':       all_soft,
        'training_dataset':  training_df,
    }


if __name__ == '__main__':
    run_pipeline()
