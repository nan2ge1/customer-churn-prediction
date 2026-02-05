"""Shared fixtures for churn_prediction tests."""

import numpy as np
import pandas as pd
import pytest

from churn_prediction.config import (
    CATEGORICAL_FEATURES,
    ID_COL,
    NUMERICAL_FEATURES,
    TARGET,
)


def _make_df(n: int = 100, churn_rate: float = 0.2, seed: int = 0) -> pd.DataFrame:
    """Generate a synthetic DataFrame matching the real schema."""
    rng = np.random.default_rng(seed)

    contract_types = ["Month-to-month", "One year", "Two year"]
    payment_methods = ["Bank transfer", "Credit card", "Electronic check", "Mailed check"]

    n_churn = int(n * churn_rate)

    data = {
        ID_COL: [f"CUST{i:04d}" for i in range(n)],
        "ContractDuration": rng.integers(1, 60, size=n),
        "MonthlyCharges": rng.uniform(20, 120, size=n).round(2),
        "InternetUsageGB": rng.uniform(0, 100, size=n).round(2),
        "SupportCalls": rng.integers(0, 10, size=n),
        "ContractType": rng.choice(contract_types, size=n),
        "PaymentMethod": rng.choice(payment_methods, size=n),
        TARGET: np.array([1] * n_churn + [0] * (n - n_churn)),
    }
    df = pd.DataFrame(data)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


@pytest.fixture()
def sample_df():
    """A 100-row synthetic DataFrame with ~20% churn."""
    return _make_df()


@pytest.fixture()
def sample_Xy(sample_df):
    """Features (X) and target (y) from the synthetic DataFrame."""
    df = sample_df.drop(columns=[ID_COL])
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


@pytest.fixture()
def sample_customer():
    """A single customer dict for prediction tests."""
    return {
        "ContractDuration": 12,
        "MonthlyCharges": 95.0,
        "InternetUsageGB": 60.0,
        "SupportCalls": 4,
        "ContractType": "Month-to-month",
        "PaymentMethod": "Electronic check",
    }
