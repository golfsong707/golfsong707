"""Visual evaluation: ROC curves, confusion matrix and feature importance."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve

from .train import OUT_DIR

MODEL_KEYS = [
    "baseline_majority",
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
]


def _load() -> tuple[pd.DataFrame, pd.DataFrame | None]:
    proba = pd.read_csv(OUT_DIR / "test_predictions.csv", parse_dates=["date"])
    imp_path = OUT_DIR / "feature_importance.csv"
    importance = pd.read_csv(imp_path) if imp_path.exists() else None
    return proba, importance


def make_plots() -> list[Path]:
    proba, importance = _load()
    y_true = proba["target"]

    # 1) ROC curves
    fig, ax = plt.subplots(figsize=(7, 6))
    for key in MODEL_KEYS:
        fpr, tpr, _ = roc_curve(y_true, proba[f"proba_{key}"])
        ax.plot(fpr, tpr, label=key.replace("_", " "))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC — next-day price direction")
    ax.legend(loc="lower right")
    fig.tight_layout()
    roc_path = OUT_DIR / "roc_curves.png"
    fig.savefig(roc_path, dpi=130)
    plt.close(fig)

    # 2) Confusion matrix for the best non-baseline model (random forest)
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_true, (proba["proba_random_forest"] >= 0.5).astype(int))
    ConfusionMatrixDisplay(cm, display_labels=["Down", "Up"]).plot(ax=ax, colorbar=False)
    ax.set_title("Random forest — confusion matrix")
    fig.tight_layout()
    cm_path = OUT_DIR / "confusion_matrix_rf.png"
    fig.savefig(cm_path, dpi=130)
    plt.close(fig)

    # 3) Feature importance
    paths = [roc_path, cm_path]
    if importance is not None:
        fig, ax = plt.subplots(figsize=(7, 6))
        imp = importance.head(13).iloc[::-1]
        ax.barh(imp["feature"], imp["importance"])
        ax.set_xlabel("Mean decrease in impurity")
        ax.set_title("Random forest — feature importance")
        fig.tight_layout()
        imp_path = OUT_DIR / "feature_importance.png"
        fig.savefig(imp_path, dpi=130)
        plt.close(fig)
        paths.append(imp_path)

    return paths


if __name__ == "__main__":
    for p in make_plots():
        print(f"[evaluate] wrote {p}")
