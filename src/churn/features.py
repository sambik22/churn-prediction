"""Feature engineering і препроцесинг як частина sklearn-пайплайну.

Усе, що робиться з даними, живе всередині ``Pipeline``. Тому одні й ті самі
перетворення застосовуються під час навчання, крос-валідації та в API, і немає
витоку інформації з тестової вибірки (data leakage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn import config

ADDON_SERVICES = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

ENGINEERED_NUMERIC = [
    "num_addon_services",
    "avg_monthly_spend",
    "charge_delta",
    "tenure_years",
]
ENGINEERED_BINARY = [
    "is_month_to_month",
    "is_electronic_check",
    "has_fiber",
    "no_protection",
    "is_new_customer",
]


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Додає ознаки, які напряму відображають гіпотези з EDA.

    * ``num_addon_services`` — скільки додаткових послуг підключено.
      Більше послуг означає «вищі витрати на перехід» до конкурента.
    * ``avg_monthly_spend`` — TotalCharges / tenure, середній фактичний платіж.
    * ``charge_delta`` — різниця між поточним і середнім платежем: чи зростав рахунок.
    * ``is_month_to_month``, ``is_electronic_check``, ``has_fiber`` — найсильніші
      сигнали відтоку з EDA у вигляді явних прапорців.
    * ``no_protection`` — інтернет є, але немає ні OnlineSecurity, ні TechSupport.
    * ``is_new_customer`` — перші 6 місяців, найризиковіший період.
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["num_addon_services"] = (X[ADDON_SERVICES] == "Yes").sum(axis=1)
        safe_tenure = X["tenure"].replace(0, np.nan)
        X["avg_monthly_spend"] = (X["TotalCharges"] / safe_tenure).fillna(X["MonthlyCharges"])
        X["charge_delta"] = X["MonthlyCharges"] - X["avg_monthly_spend"]
        X["tenure_years"] = X["tenure"] / 12
        X["is_month_to_month"] = (X["Contract"] == "Month-to-month").astype(int)
        X["is_electronic_check"] = (X["PaymentMethod"] == "Electronic check").astype(int)
        X["has_fiber"] = (X["InternetService"] == "Fiber optic").astype(int)
        X["no_protection"] = (
            (X["InternetService"] != "No")
            & (X["OnlineSecurity"] == "No")
            & (X["TechSupport"] == "No")
        ).astype(int)
        X["is_new_customer"] = (X["tenure"] <= 6).astype(int)
        return X

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else config.RAW_FEATURES
        return np.array(base + ENGINEERED_NUMERIC + ENGINEERED_BINARY)


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """Кодує категорії one-hot і (для лінійних моделей) стандартизує числа.

    Деревам масштабування не потрібне, але воно їм і не шкодить, тому
    для простоти пайплайн однаковий для всіх моделей, крім прапорця ``scale_numeric``.
    """
    numeric = config.NUMERIC_FEATURES + ENGINEERED_NUMERIC
    passthrough = config.BINARY_FEATURES + ENGINEERED_BINARY
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler() if scale_numeric else "passthrough", numeric),
            ("bin", "passthrough", passthrough),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False),
                config.CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )
