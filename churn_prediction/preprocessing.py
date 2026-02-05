# --------------------------------------------------------------------------- #
# preprocessing.py – Train/test split and pipeline construction.
# --------------------------------------------------------------------------- #

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

from churn_prediction.config import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
    TEST_SIZE,
)


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split to preserve class distribution."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info("Train set: %d samples  (churn=%d)", len(y_train), y_train.sum())
    logger.info("Test  set: %d samples  (churn=%d)", len(y_test), y_test.sum())
    return X_train, X_test, y_train, y_test


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer with scaling + one-hot encoding."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def build_pipeline(preprocessor: ColumnTransformer, estimator) -> Pipeline:
    """Wrap a preprocessor and estimator into a single sklearn Pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )


def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    """Calculate XGBoost scale_pos_weight = count(neg) / count(pos)."""
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    weight = n_neg / n_pos
    logger.info("scale_pos_weight = %d/%d = %.2f", n_neg, n_pos, weight)
    return weight
