"""Loading and cleaning the raw data."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

from churn import config


def download_data(path: Path = config.RAW_DATA_PATH, force: bool = False) -> Path:
    """Download the IBM Telco Customer Churn dataset and verify its checksum."""
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(config.DATA_URL, timeout=60) as resp:
        content = resp.read()
    digest = hashlib.sha256(content).hexdigest()
    if digest != config.DATA_SHA256:
        raise ValueError(f"Unexpected dataset SHA-256: {digest}")
    path.write_bytes(content)
    return path


def load_raw(path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the known issues of the dataset.

    * ``TotalCharges`` is stored as a string, and for 11 brand-new customers
      (tenure = 0) it is a blank. They have not been billed yet, so the honest value is 0.
    * The ``Churn`` target is converted from "Yes"/"No" to 1/0.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(r"^\s*$", "0", regex=True))
    if config.TARGET in df.columns:
        df[config.TARGET] = (df[config.TARGET] == "Yes").astype(int)
    return df


def load_dataset(path: Path = config.RAW_DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) ready for the pipeline."""
    df = clean(load_raw(path))
    X = df[config.RAW_FEATURES]
    y = df[config.TARGET]
    return X, y
