# Explainability Module

## Purpose

This module provides reusable utilities for model explainability, evaluation, and plotting. It is prepared for Person B's trained model, but it does not train a model or run SHAP or Captum analysis on Day 1.

## Files Completed

- `shap_analysis.py` - reusable SHAP setup and explanation functions.
- `captum_analysis.py` - reusable Captum attribution functions.
- `visualization.py` - reusable plotting and plot-saving functions.
- `evaluation.py` - reusable classification metric functions.
- `results/` - output folder for future explainability plots.

## Inputs

- Final training dataset from `shared/config.py` when available.
- Processed fallback datasets from `shared/config.py`:
  - `processed_hate_speech.csv`
  - `processed_hatexplain.csv`
- Trained model path after Person B completes model training.

## Outputs

- Evaluation metric values returned from `evaluation.py`.
- SHAP values and feature importance returned from `shap_analysis.py`.
- Captum attributions returned from `captum_analysis.py`.
- Saved plots inside `explainability_module/results/`.

## Day 2 Plan

- Load Person B's trained model.
- Run SHAP global and local explanations.
- Run Captum attribution methods on model inputs.
- Generate confusion matrix, class distribution, fairness metric, and feature importance plots.
- Save all explainability outputs inside `explainability_module/results/`.
