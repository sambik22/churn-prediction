"""Tests do not need the internet: they use a small synthetic dataset in the
IBM Telco format and a quick model trained right inside the fixture."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from churn import config
from churn.models import make_pipeline
from churn.predict import ChurnModel

ADDON_CHOICES = ["Yes", "No", "No internet service"]


def make_raw_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    internet = rng.choice(["DSL", "Fiber optic", "No"], n)

    def addon():
        vals = rng.choice(["Yes", "No"], n)
        return np.where(internet == "No", "No internet service", vals)

    phone = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])
    tenure = rng.integers(0, 72, n)
    monthly = rng.uniform(18, 120, n).round(2)
    total = (monthly * tenure).round(2).astype(str)
    total[tenure == 0] = " "
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n)
    churn_logit = -1 + 1.5 * (contract == "Month-to-month") - 0.04 * tenure
    churn = rng.random(n) < 1 / (1 + np.exp(-churn_logit))
    return pd.DataFrame(
        {
            "customerID": [f"ID-{i}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], n),
            "SeniorCitizen": rng.choice([0, 1], n),
            "Partner": rng.choice(["Yes", "No"], n),
            "Dependents": rng.choice(["Yes", "No"], n),
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": np.where(
                phone == "No", "No phone service", rng.choice(["Yes", "No"], n)
            ),
            "InternetService": internet,
            "OnlineSecurity": addon(),
            "OnlineBackup": addon(),
            "DeviceProtection": addon(),
            "TechSupport": addon(),
            "StreamingTV": addon(),
            "StreamingMovies": addon(),
            "Contract": contract,
            "PaperlessBilling": rng.choice(["Yes", "No"], n),
            "PaymentMethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                n,
            ),
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Churn": np.where(churn, "Yes", "No"),
        }
    )


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return make_raw_df()


@pytest.fixture(scope="session")
def trained_model(raw_df, tmp_path_factory) -> tuple[ChurnModel, str]:
    from churn.data import clean

    df = clean(raw_df)
    X, y = df[config.RAW_FEATURES], df[config.TARGET]
    pipe = make_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    model = ChurnModel(
        pipeline=pipe,
        threshold=0.5,
        background=X.sample(50, random_state=0),
        metadata={
            "model_name": "logreg-test",
            "trained_at": "2026-01-01T00:00:00+00:00",
            "n_train": len(X),
            "test_metrics": {"roc_auc": 0.8},
        },
    )
    path = tmp_path_factory.mktemp("models") / "model.joblib"
    model.save(path)
    return model, str(path)
