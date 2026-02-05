"""Tests for churn_prediction.preprocessing."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from churn_prediction.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from churn_prediction.preprocessing import (
    build_pipeline,
    build_preprocessor,
    compute_scale_pos_weight,
    split_data,
)


def test_split_data_preserves_size(sample_Xy):
    X, y = sample_Xy
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)


def test_split_data_stratifies(sample_Xy):
    X, y = sample_Xy
    original_rate = y.mean()
    _, _, y_train, y_test = split_data(X, y)
    # Stratification should keep rates roughly similar
    assert abs(y_train.mean() - original_rate) < 0.05
    assert abs(y_test.mean() - original_rate) < 0.05


def test_build_preprocessor_type():
    preprocessor = build_preprocessor()
    assert isinstance(preprocessor, ColumnTransformer)


def test_build_preprocessor_transformers():
    preprocessor = build_preprocessor()
    names = [name for name, _, _ in preprocessor.transformers]
    assert "num" in names
    assert "cat" in names


def test_build_pipeline_returns_pipeline():
    preprocessor = build_preprocessor()
    estimator = DecisionTreeClassifier()
    pipeline = build_pipeline(preprocessor, estimator)
    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_compute_scale_pos_weight():
    y = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    weight = compute_scale_pos_weight(y)
    assert weight == 4.0  # 8 neg / 2 pos


def test_compute_scale_pos_weight_balanced():
    y = pd.Series([0, 0, 1, 1])
    weight = compute_scale_pos_weight(y)
    assert weight == 1.0
