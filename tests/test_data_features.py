import numpy as np

from churn import config
from churn.data import clean
from churn.evaluate import best_profit_threshold, expected_profit
from churn.features import ENGINEERED_BINARY, ENGINEERED_NUMERIC, ChurnFeatureEngineer


def test_clean_fixes_blank_total_charges_and_target(raw_df):
    df = clean(raw_df)
    assert df["TotalCharges"].dtype.kind == "f"
    assert (df.loc[df["tenure"] == 0, "TotalCharges"] == 0).all()
    assert set(df[config.TARGET].unique()) <= {0, 1}


def test_feature_engineer_adds_expected_columns(raw_df):
    X = clean(raw_df)[config.RAW_FEATURES]
    out = ChurnFeatureEngineer().fit_transform(X)
    for col in ENGINEERED_NUMERIC + ENGINEERED_BINARY:
        assert col in out.columns
    assert out["num_addon_services"].between(0, 6).all()
    assert not out["avg_monthly_spend"].isna().any()
    assert out.loc[X["tenure"] == 0, "is_new_customer"].eq(1).all()


def test_profit_logic():
    y = np.array([1, 1, 0, 0])
    proba = np.array([0.9, 0.2, 0.8, 0.1])
    # поріг 0.5: TP=1, FP=1 -> 1*(0.3*500-50) - 1*50 = 50
    assert expected_profit(y, proba, 0.5) == 50.0
    t, p = best_profit_threshold(y, proba)
    assert p >= 50.0
    assert 0.05 <= t <= 0.95
