from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from shared import config


RESULTS_DIR = config.BASE_DIR / "explainability_module" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_feature_importance(feature_importance, feature_col="feature", score_col="importance", top_n=20):
    data = feature_importance.sort_values(score_col, ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=data, x=score_col, y=feature_col, ax=ax)
    ax.set_title("Global Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(confusion_values, labels=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(confusion_values, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


def plot_class_distribution(labels, title="Class Distribution"):
    counts = pd.Series(labels).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def plot_fairness_metrics(metrics, group_col="group", metric_col="score"):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=metrics, x=group_col, y=metric_col, ax=ax)
    ax.set_title("Fairness Metrics")
    ax.set_xlabel("Group")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def save_plot(fig, filename):
    output_path = RESULTS_DIR / Path(filename).name
    fig.savefig(output_path, bbox_inches="tight")
    return output_path
