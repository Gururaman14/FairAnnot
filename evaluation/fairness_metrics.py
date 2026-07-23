import pandas as pd

from shared import config


PROTECTED_ATTRIBUTE_CANDIDATES = [
    "gender",
    "race",
    "religion",
    "age",
    "ethnicity",
    "identity",
    "target_group",
    "protected_attribute",
]


def find_protected_columns(dataframe):
    return [col for col in PROTECTED_ATTRIBUTE_CANDIDATES if col in dataframe.columns]


def prediction_distribution(dataframe, pred_col, group_col=None):
    if group_col is None:
        return dataframe[pred_col].value_counts(normalize=True).rename("rate").reset_index()

    grouped = dataframe.groupby(group_col)[pred_col].value_counts(normalize=True)
    return grouped.rename("rate").reset_index()


def group_performance(dataframe, true_col, pred_col, group_col):
    rows = []
    for group, group_df in dataframe.groupby(group_col):
        rows.append({
            "group_column": group_col,
            "group": group,
            "sample_count": len(group_df),
            "accuracy": (group_df[true_col] == group_df[pred_col]).mean(),
        })
    return pd.DataFrame(rows)


def demographic_parity_difference(dataframe, pred_col, group_col, positive_label):
    rates = (
        dataframe.assign(is_positive=dataframe[pred_col] == positive_label)
        .groupby(group_col)["is_positive"]
        .mean()
    )
    return rates.max() - rates.min()


def evaluate_fairness(dataframe, true_col, pred_col, protected_cols=None, positive_label=None):
    protected_cols = protected_cols or find_protected_columns(dataframe)
    if not protected_cols:
        return pd.DataFrame([{
            "status": "No protected attribute columns available for fairness evaluation."
        }])

    positive_label = positive_label or dataframe[pred_col].value_counts().idxmax()
    results = []
    for col in protected_cols:
        performance = group_performance(dataframe, true_col, pred_col, col)
        performance["positive_label"] = positive_label
        performance["demographic_parity_difference"] = demographic_parity_difference(
            dataframe, pred_col, col, positive_label
        )
        results.append(performance)

    return pd.concat(results, ignore_index=True)


def save_fairness_results(results, filename="fairness_results.csv"):
    output_path = config.EVALUATION_RESULTS_DIR / filename
    results.to_csv(output_path, index=False)
    return output_path


def save_prediction_distribution(distribution, filename="prediction_distribution.csv"):
    output_path = config.EVALUATION_RESULTS_DIR / filename
    distribution.to_csv(output_path, index=False)
    return output_path
