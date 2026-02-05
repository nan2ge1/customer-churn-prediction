"""Tests for churn_prediction.data."""

import os
import tempfile

import pandas as pd

from churn_prediction.config import ID_COL, TARGET
from churn_prediction.data import explore, load_data, prepare_features


def test_load_data_returns_dataframe():
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_load_data_has_expected_columns():
    df = load_data()
    assert ID_COL in df.columns
    assert TARGET in df.columns


def test_load_data_custom_path(sample_df, tmp_path):
    csv_path = tmp_path / "test.csv"
    sample_df.to_csv(csv_path, index=False)
    df = load_data(str(csv_path))
    assert len(df) == len(sample_df)


def test_explore_runs_without_error(sample_df):
    explore(sample_df)


def test_prepare_features_drops_id(sample_df):
    X, y = prepare_features(sample_df)
    assert ID_COL not in X.columns
    assert TARGET not in X.columns


def test_prepare_features_target_shape(sample_df):
    X, y = prepare_features(sample_df)
    assert len(X) == len(y)
    assert set(y.unique()).issubset({0, 1})
