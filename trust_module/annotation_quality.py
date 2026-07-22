# cspell:disable
import numpy as np
import pandas as pd


def cohens_kappa(labels_a, labels_b):
    labels_a = np.array(labels_a)
    labels_b = np.array(labels_b)
    classes = np.unique(np.concatenate([labels_a, labels_b]))

    n = len(labels_a)
    if n == 0:
        return float('nan')

    p_o = (labels_a == labels_b).mean()

    p_e = sum(
        ((labels_a == c).mean() * (labels_b == c).mean())
        for c in classes
    )

    if p_e == 1.0:
        return 1.0

    return (p_o - p_e) / (1.0 - p_e)


def percent_agreement(labels_a, labels_b):
    labels_a = np.array(labels_a)
    labels_b = np.array(labels_b)
    if len(labels_a) == 0:
        return float('nan')
    return float((labels_a == labels_b).mean())


def annotator_vs_majority(annotations_df):
    majority = (
        annotations_df.groupby('item_id')['label']
        .agg(lambda x: x.value_counts().idxmax())
        .rename('majority_label')
    )

    merged = annotations_df.merge(majority, on='item_id')
    merged['agrees'] = merged['label'] == merged['majority_label']

    stats = (
        merged.groupby('annotator_id')
        .agg(
            agreement_with_majority=('agrees', 'mean'),
            num_annotations=('agrees', 'count'),
        )
        .reset_index()
    )
    return stats


def compute_quality_report(annotations_df, annotator_weights_df, dataset_name):
    agreement_stats = annotator_vs_majority(annotations_df)

    report = annotator_weights_df.merge(agreement_stats, on='annotator_id', how='left')
    report['dataset'] = dataset_name
    report = report.sort_values('reliability_score', ascending=False).reset_index(drop=True)

    return (report[['dataset', 'annotator_id', 'reliability_score', 
                    'agreement_with_majority', 'num_annotations_x']]
            .rename(columns={'num_annotations_x': 'num_annotations'}))


def print_quality_summary(report, dataset_name):
    print(f"\n{'='*55}")
    print(f"  Annotation Quality - {dataset_name}")
    print(f"{'='*55}")
    print(f"  Annotators               : {len(report):,}")
    print(f"  Avg reliability score    : {report['reliability_score'].mean():.4f}")
    print(f"  Avg agreement w/ majority: {report['agreement_with_majority'].mean():.4f}")
    print(f"  High-reliability (>=0.8) : {(report['reliability_score'] >= 0.8).sum():,}")
    print(f"  Low-reliability  (<0.5)  : {(report['reliability_score'] < 0.5).sum():,}")
    print(f"{'='*55}")
