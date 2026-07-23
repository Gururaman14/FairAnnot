import pickle
from pathlib import Path

import pandas as pd

from shared import config


def load_model(model_path=None):
    path = model_path or getattr(config, "MODEL_PATH", None)
    if path is None:
        raise FileNotFoundError("Trained model path is not configured yet.")

    path = Path(path)
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


def integrated_gradients(model, inputs, target=None, baselines=None):
    from captum.attr import IntegratedGradients

    attribution = IntegratedGradients(model)
    return attribution.attribute(inputs, baselines=baselines, target=target)


def saliency_map(model, inputs, target=None):
    from captum.attr import Saliency

    attribution = Saliency(model)
    return attribution.attribute(inputs, target=target)


def feature_attribution(model, inputs, method="integrated_gradients", target=None, baselines=None):
    if method == "integrated_gradients":
        return integrated_gradients(model, inputs, target=target, baselines=baselines)
    if method == "saliency":
        return saliency_map(model, inputs, target=target)
    raise ValueError("method must be 'integrated_gradients' or 'saliency'.")
