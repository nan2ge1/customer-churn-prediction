"""Tests for churn_prediction.config."""

from churn_prediction import config


def test_feature_lists_are_non_empty():
    assert len(config.NUMERICAL_FEATURES) > 0
    assert len(config.CATEGORICAL_FEATURES) > 0


def test_no_feature_overlap():
    overlap = set(config.NUMERICAL_FEATURES) & set(config.CATEGORICAL_FEATURES)
    assert overlap == set(), f"Features appear in both lists: {overlap}"


def test_target_not_in_features():
    all_features = config.NUMERICAL_FEATURES + config.CATEGORICAL_FEATURES
    assert config.TARGET not in all_features


def test_split_params_sensible():
    assert 0 < config.TEST_SIZE < 1
    assert isinstance(config.RANDOM_STATE, int)
