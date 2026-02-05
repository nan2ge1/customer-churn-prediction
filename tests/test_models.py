"""Tests for churn_prediction.models."""

import pandas as pd
from sklearn.pipeline import Pipeline

from churn_prediction.models import create_rf_pipeline, create_xgb_pipeline


def test_create_rf_pipeline_type():
    pipeline = create_rf_pipeline()
    assert isinstance(pipeline, Pipeline)


def test_rf_pipeline_has_balanced_weight():
    pipeline = create_rf_pipeline()
    clf = pipeline.named_steps["classifier"]
    assert clf.class_weight == "balanced"


def test_create_xgb_pipeline_type(sample_Xy):
    _, y = sample_Xy
    pipeline = create_xgb_pipeline(y)
    assert isinstance(pipeline, Pipeline)


def test_xgb_pipeline_has_scale_pos_weight(sample_Xy):
    _, y = sample_Xy
    pipeline = create_xgb_pipeline(y)
    clf = pipeline.named_steps["classifier"]
    assert clf.scale_pos_weight > 0


def test_rf_pipeline_fits_and_predicts(sample_Xy):
    X, y = sample_Xy
    pipeline = create_rf_pipeline()
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})


def test_xgb_pipeline_fits_and_predicts(sample_Xy):
    X, y = sample_Xy
    pipeline = create_xgb_pipeline(y)
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})
