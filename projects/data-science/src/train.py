"""Train and evaluate models that predict next-day price direction."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES, split_time

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def build_models() -> dict[str, object]:
    return {
        "baseline_majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=707)
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250, max_depth=8, random_state=707, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=707
        ),
    }


def train_and_evaluate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, test = split_time(df)
    print(f"[train] train={len(train)} rows, test={len(test)} rows")
    print(f"[train] test period: {test['date'].min().date()} -> {test['date'].max().date()}")

    X_train, y_train = train[FEATURES], train["target"]
    X_test, y_test = test[FEATURES], test["target"]

    metrics_rows = []
    proba_frame = test[["date", "ticker", "sector", "target"]].reset_index(drop=True)
    importance = None

    models = build_models()
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else pred.astype(float)
        )

        metrics_rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred),
                "recall": recall_score(y_test, pred),
                "f1": f1_score(y_test, pred),
                "roc_auc": roc_auc_score(y_test, proba),
            }
        )
        proba_frame[f"proba_{name}"] = proba

        if name == "random_forest":
            importance = pd.DataFrame(
                {
                    "feature": FEATURES,
                    "importance": model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODELS_DIR / f"{name}.pkl", "wb") as fh:
            pickle.dump(model, fh)

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(OUT_DIR / "metrics.csv", index=False)
    proba_frame.to_csv(OUT_DIR / "test_predictions.csv", index=False)
    if importance is not None:
        importance.to_csv(OUT_DIR / "feature_importance.csv", index=False)

    return metrics, proba_frame, importance


if __name__ == "__main__":
    from .make_dataset import load_market_data
    from .features import build_features

    data = load_market_data()
    feat = build_features(data)
    metrics, _, _ = train_and_evaluate(feat)
    print(metrics.to_string(index=False))
