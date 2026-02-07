"""Tests for churn_prediction.evaluation."""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for CI

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from churn_prediction.evaluation import (
    _get_feature_names,
    evaluate_model,
    plot_feature_importance,
    train_model,
)
from churn_prediction.config import NUMERICAL_FEATURES
from churn_prediction.models import create_rf_pipeline


def _fitted_pipeline(sample_Xy):
    X, y = sample_Xy
    pipeline = create_rf_pipeline()
    pipeline.fit(X, y)
    return pipeline, X, y


def test_train_model_returns_fitted_pipeline(sample_Xy):
    X, y = sample_Xy
    pipeline = create_rf_pipeline()
    result = train_model(pipeline, X, y, "TestModel")
    assert result is pipeline
    # Should be fitted — predict should work
    preds = result.predict(X)
    assert len(preds) == len(y)


def test_evaluate_model_runs(sample_Xy):
    pipeline, X, y = _fitted_pipeline(sample_Xy)
    # Should not raise
    evaluate_model(pipeline, X, y, "TestModel")


def test_get_feature_names(sample_Xy):
    pipeline, X, y = _fitted_pipeline(sample_Xy)
    names = _get_feature_names(pipeline)
    assert isinstance(names, list)
    assert len(names) > 0
    # Numerical features should appear first
    for feat in NUMERICAL_FEATURES:
        assert feat in names


def test_plot_feature_importance_runs(sample_Xy):
    pipeline, X, y = _fitted_pipeline(sample_Xy)
    plot_feature_importance(pipeline, "TestModel")
