"""All project paths and "magic numbers" in one place."""

from __future__ import annotations

import os
from pathlib import Path

# Project root. In Docker the package lives in site-packages, so the root is set
# explicitly via CHURN_ROOT; locally (editable install) it is derived from this file.
ROOT = Path(os.getenv("CHURN_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "telco_churn.csv"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "churn_model.joblib"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
DATA_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

TARGET = "Churn"
ID_COL = "customerID"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_FEATURES = ["SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
RAW_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# Business assumptions used to pick the decision threshold (see README, "Threshold").
# Retention: we offer the customer a discount that costs OFFER_COST.
# If the customer really was about to leave and the offer works (probability
# OFFER_SUCCESS_RATE), we keep CUSTOMER_VALUE of revenue.
CUSTOMER_VALUE = 500.0
OFFER_COST = 50.0
OFFER_SUCCESS_RATE = 0.3
