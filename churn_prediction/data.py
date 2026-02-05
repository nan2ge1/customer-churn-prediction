# --------------------------------------------------------------------------- #
# data.py – Data loading and initial exploration.
# --------------------------------------------------------------------------- #

import logging

import pandas as pd

from churn_prediction.config import DATA_PATH, ID_COL, TARGET

logger = logging.getLogger(__name__)


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the CSV and return a raw DataFrame."""
    df = pd.read_csv(path)
    return df


def explore(df: pd.DataFrame) -> None:
    """Print basic statistics required by the task spec."""
    logger.info("=== DataFrame Info ===")
    df.info()

    logger.info("=== First 5 Rows ===\n%s", df.head())

    logger.info("=== Class Distribution (%s) ===\n%s", TARGET, df[TARGET].value_counts(normalize=True))


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop the ID column and split into X / y."""
    df = df.drop(columns=[ID_COL])
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y
