"""Завантаження збереженої моделі та прогноз для нових клієнтів."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from churn import config
from churn.explain import ChurnExplainer


@dataclass
class ChurnModel:
    pipeline: Pipeline
    threshold: float
    background: pd.DataFrame
    metadata: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = config.MODEL_PATH) -> ChurnModel:
        return cls(**joblib.load(path))

    def save(self, path: Path = config.MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "threshold": self.threshold,
                "background": self.background,
                "metadata": self.metadata,
            },
            path,
        )

    @cached_property
    def explainer(self) -> ChurnExplainer:
        return ChurnExplainer(self.pipeline, self.background)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        X = X[config.RAW_FEATURES]
        return pd.Series(self.pipeline.predict_proba(X)[:, 1], index=X.index)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return (self.predict_proba(X) >= self.threshold).astype(int)


def risk_level(proba: float, threshold: float) -> str:
    """high — модель радить діяти (≥ порогу); medium — «список спостереження»."""
    if proba >= threshold:
        return "high"
    if proba >= 0.6 * threshold:
        return "medium"
    return "low"
