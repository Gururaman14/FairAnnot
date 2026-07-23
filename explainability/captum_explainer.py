from shared import config


def load_trained_model(model_path=None):
    import torch

    path = model_path or config.TRAINED_MODEL_PATH
    if not path.exists() and config.ROOT_TRAINED_MODEL_PATH.exists():
        path = config.ROOT_TRAINED_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Trained model not found: {path}")

    return torch.load(path, map_location="cpu")


def integrated_gradients(model, inputs, target=None, baselines=None):
    from captum.attr import IntegratedGradients

    model.eval()
    attribution = IntegratedGradients(model)
    return attribution.attribute(inputs, baselines=baselines, target=target)


def saliency(model, inputs, target=None):
    from captum.attr import Saliency

    model.eval()
    attribution = Saliency(model)
    return attribution.attribute(inputs, target=target)


def summarize_token_attributions(tokens, attributions):
    values = attributions.detach().cpu()
    if values.ndim > 1:
        values = values.sum(dim=-1)
    return list(zip(tokens, values.tolist()))


def save_token_attributions(token_scores, filename="captum_token_attributions.csv"):
    import pandas as pd

    output_path = config.EXPLAINABILITY_RESULTS_DIR / filename
    pd.DataFrame(token_scores, columns=["token", "attribution"]).to_csv(output_path, index=False)
    return output_path


def plot_token_attributions(token_scores, filename="captum_token_attributions.png", top_n=20):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    output_path = config.EXPLAINABILITY_RESULTS_DIR / filename
    data = pd.DataFrame(token_scores, columns=["token", "attribution"])
    data["absolute_attribution"] = data["attribution"].abs()
    data = data.sort_values("absolute_attribution", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=data, x="attribution", y="token", ax=ax)
    ax.set_title("Captum Token Attributions")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def compare_with_shap(captum_scores, shap_scores):
    return {
        "captum_items": len(captum_scores),
        "shap_items": len(shap_scores),
        "comparison_note": "Compare highest-attribution tokens/features from both methods.",
    }
