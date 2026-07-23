import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from shared import config


def calculate_metrics(y_true, y_pred, average="weighted"):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def class_wise_performance(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    rows = []
    for label, values in report.items():
        if isinstance(values, dict):
            rows.append({"class": label, **values})
    return pd.DataFrame(rows)


def create_confusion_matrix(y_true, y_pred, labels=None):
    return confusion_matrix(y_true, y_pred, labels=labels)


def save_metrics(metrics, filename="metrics.csv"):
    output_path = config.EVALUATION_RESULTS_DIR / filename
    pd.DataFrame([metrics]).to_csv(output_path, index=False)
    return output_path


def save_predictions(dataframe, y_true, y_pred, filename="predictions.csv"):
    output = dataframe.copy()
    output["true_label"] = y_true
    output["predicted_label"] = y_pred
    output_path = config.EVALUATION_RESULTS_DIR / filename
    output.to_csv(output_path, index=False)
    return output_path


def save_classification_report(y_true, y_pred, filename="classification_report.csv"):
    report_df = class_wise_performance(y_true, y_pred)
    output_path = config.EVALUATION_RESULTS_DIR / filename
    report_df.to_csv(output_path, index=False)
    return output_path


def save_confusion_matrix_plot(matrix, labels, filename="confusion_matrix.png"):
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_path = config.EVALUATION_RESULTS_DIR / filename
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def evaluate_predictions(dataframe, true_col, pred_col):
    y_true = dataframe[true_col]
    y_pred = dataframe[pred_col]
    labels = sorted(pd.Series(y_true).dropna().unique())
    metrics = calculate_metrics(y_true, y_pred)
    report = class_wise_performance(y_true, y_pred)
    matrix = create_confusion_matrix(y_true, y_pred, labels=labels)
    return metrics, report, matrix
