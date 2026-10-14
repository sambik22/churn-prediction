"""Усі шляхи та «магічні числа» проєкту в одному місці."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
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

# Бізнес-припущення для вибору порогу (див. README, розділ «Поріг»).
# Утримання: пропонуємо клієнту знижку, що коштує OFFER_COST.
# Якщо клієнт справді збирався піти і пропозиція спрацювала (ймовірність
# OFFER_SUCCESS_RATE), ми зберігаємо CUSTOMER_VALUE доходу.
CUSTOMER_VALUE = 500.0
OFFER_COST = 50.0
OFFER_SUCCESS_RATE = 0.3
