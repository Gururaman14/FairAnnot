import json
import pandas as pd

from shared import config


def load_trained_model(model_path=None):
    import torch

    path = model_path or config.TRAINED_MODEL_PATH
    if not path.exists() and config.ROOT_TRAINED_MODEL_PATH.exists():
        path = config.ROOT_TRAINED_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Trained model not found: {path}")

    return torch.load(path, map_location="cpu")


def load_label_mapping(mapping_path=None):
    path = mapping_path or config.LABEL_MAPPING_PATH
    if not path.exists() and config.ROOT_LABEL_MAPPING_PATH.exists():
        path = config.ROOT_LABEL_MAPPING_PATH
    if not path.exists():
        raise FileNotFoundError(f"Label mapping not found: {path}")

    with open(path, "r", encoding="utf-8") as mapping_file:
        return json.load(mapping_file)


def load_dataset():
    if config.FINAL_TRAINING_DATASET_PATH.exists():
        return pd.read_csv(config.FINAL_TRAINING_DATASET_PATH)
    return pd.read_csv(config.TRAINING_DATASET_PATH)


def initialize_text_explainer(prediction_function, masker=None):
    import shap

    masker = masker or shap.maskers.Text()
    return shap.Explainer(prediction_function, masker)


def generate_shap_values(explainer, texts):
    return explainer(texts)


def save_feature_importance(shap_values, filename="shap_feature_importance.png"):
    import matplotlib.pyplot as plt
    import shap

    output_path = config.EXPLAINABILITY_RESULTS_DIR / filename
    shap.plots.bar(shap_values, show=False)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return output_path


def save_local_explanation(shap_values, sample_index, filename=None):
    import matplotlib.pyplot as plt
    import shap

    filename = filename or f"shap_local_sample_{sample_index}.png"
    output_path = config.EXPLAINABILITY_RESULTS_DIR / filename
    shap.plots.waterfall(shap_values[sample_index], show=False)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return output_path


def split_correct_and_misclassified(predictions, true_col="true_label", pred_col="predicted_label"):
    correct = predictions[predictions[true_col] == predictions[pred_col]]
    incorrect = predictions[predictions[true_col] != predictions[pred_col]]
    return correct, incorrect
