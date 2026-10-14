"""Завантаження та очищення сирих даних."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

from churn import config


def download_data(path: Path = config.RAW_DATA_PATH, force: bool = False) -> Path:
    """Завантажує датасет IBM Telco Customer Churn і перевіряє контрольну суму."""
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(config.DATA_URL, timeout=60) as resp:
        content = resp.read()
    digest = hashlib.sha256(content).hexdigest()
    if digest != config.DATA_SHA256:
        raise ValueError(f"Неочікувана SHA-256 датасету: {digest}")
    path.write_bytes(content)
    return path


def load_raw(path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Виправляє відомі проблеми датасету.

    * ``TotalCharges`` зберігається як рядок, і в 11 нових клієнтів (tenure = 0)
      це пробіл. Вони ще не отримали жодного рахунку, тож чесне значення — 0.
    * Ціль ``Churn`` перетворюємо з "Yes"/"No" на 1/0.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(r"^\s*$", "0", regex=True))
    if config.TARGET in df.columns:
        df[config.TARGET] = (df[config.TARGET] == "Yes").astype(int)
    return df


def load_dataset(path: Path = config.RAW_DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Повертає (X, y), готові для пайплайну."""
    df = clean(load_raw(path))
    X = df[config.RAW_FEATURES]
    y = df[config.TARGET]
    return X, y
