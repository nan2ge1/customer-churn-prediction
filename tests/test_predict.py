"""Tests for churn_prediction.predict."""

from churn_prediction.models import create_rf_pipeline
from churn_prediction.predict import predict_churn


def test_predict_churn_returns_tuple(sample_Xy, sample_customer):
    X, y = sample_Xy
    pipeline = create_rf_pipeline()
    pipeline.fit(X, y)

    result = predict_churn(sample_customer, pipeline)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_predict_churn_class_is_binary(sample_Xy, sample_customer):
    X, y = sample_Xy
    pipeline = create_rf_pipeline()
    pipeline.fit(X, y)

    pred_class, prob = predict_churn(sample_customer, pipeline)
    assert pred_class in (0, 1)


def test_predict_churn_probability_range(sample_Xy, sample_customer):
    X, y = sample_Xy
    pipeline = create_rf_pipeline()
    pipeline.fit(X, y)

    _, prob = predict_churn(sample_customer, pipeline)
    assert 0.0 <= prob <= 1.0
