"""Кандидати-моделі. Кожна — повний пайплайн від сирих ознак до ймовірності."""

from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churn.config import RANDOM_STATE
from churn.features import ChurnFeatureEngineer, build_preprocessor


def make_pipeline(model, scale_numeric: bool = True) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", ChurnFeatureEngineer()),
            ("preprocess", build_preprocessor(scale_numeric=scale_numeric)),
            ("model", model),
        ]
    )


def get_candidates() -> dict[str, Pipeline]:
    """Від найпростішої до найскладнішої.

    * ``baseline`` завжди прогнозує частку класу. Будь-яка модель має бути кращою.
    * ``logreg`` — сильний інтерпретований бейзлайн.
    * ``random_forest``, ``xgboost``, ``lightgbm`` — ансамблі дерев, які ловлять
      нелінійності та взаємодії ознак.

    ``class_weight="balanced"`` / ``scale_pos_weight`` компенсують дисбаланс класів
    (≈26.5% відтоку). Гіперпараметри помірно консервативні: датасет малий (7k рядків),
    тож глибокі дерева швидко перенавчаються.
    """
    pos_weight = 2.77  # ≈ (1 - 0.265) / 0.265
    return {
        "baseline": make_pipeline(DummyClassifier(strategy="prior")),
        "logreg": make_pipeline(LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000)),
        "random_forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=400,
                max_depth=8,
                min_samples_leaf=10,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            scale_numeric=False,
        ),
        "xgboost": make_pipeline(
            XGBClassifier(
                n_estimators=400,
                learning_rate=0.03,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_lambda=1.0,
                scale_pos_weight=pos_weight,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            scale_numeric=False,
        ),
        "lightgbm": make_pipeline(
            LGBMClassifier(
                n_estimators=400,
                learning_rate=0.03,
                num_leaves=15,
                min_child_samples=30,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                verbose=-1,
            ),
            scale_numeric=False,
        ),
    }
