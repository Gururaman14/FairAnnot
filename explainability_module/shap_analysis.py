import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from shared import config


def load_model(model_path=None):
    path = model_path or getattr(config, "TRAINED_MODEL_PATH", None)
    if path is not None:
        path = Path(path)
        if not path.exists() and getattr(config, "ROOT_TRAINED_MODEL_PATH", None):
            root_path = Path(config.ROOT_TRAINED_MODEL_PATH)
            if root_path.exists():
                path = root_path
    if path is None:
        raise FileNotFoundError("Trained model path is not configured yet.")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trained model not found: {path}")

    if path.suffix in {".pt", ".pth"}:
        import torch

        return torch.load(path, map_location="cpu")

    with open(path, "rb") as model_file:
        return pickle.load(model_file)


def load_dataset():
    if config.FINAL_TRAINING_DATASET_PATH.exists():
        return pd.read_csv(config.FINAL_TRAINING_DATASET_PATH)

    datasets = []
    if config.PROCESSED_HATE_SPEECH_PATH.exists():
        datasets.append(pd.read_csv(config.PROCESSED_HATE_SPEECH_PATH))
    if config.PROCESSED_HATEXPLAIN_PATH.exists():
        datasets.append(pd.read_csv(config.PROCESSED_HATEXPLAIN_PATH))

    if not datasets:
        raise FileNotFoundError("No processed or final training dataset is available.")

    return pd.concat(datasets, ignore_index=True)


def initialize_shap(model, background_data, explainer_class=None):
    import shap

    explainer = explainer_class or shap.Explainer
    return explainer(model, background_data)


def generate_shap_values(explainer, data):
    return explainer(data)


def generate_global_feature_importance(shap_values, feature_names=None):
    values = np.asarray(getattr(shap_values, "values", shap_values))
    importance = np.abs(values).mean(axis=0)
    if importance.ndim > 1:
        importance = importance.mean(axis=-1)

    return pd.DataFrame({
        "feature": feature_names or list(range(len(importance))),
        "importance": importance,
    }).sort_values("importance", ascending=False)


def generate_local_explanation(shap_values, row_index):
    values = getattr(shap_values, "values", shap_values)
    return values[row_index]
