"""Tests for churn_prediction.eda."""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

from churn_prediction.eda import (
    plot_correlation_matrix,
    plot_histogram,
    ALL_FEATURE_COLS,
)


def test_plot_histogram_numerical(sample_df, tmp_path):
    plot_histogram(sample_df, "MonthlyCharges", output_dir=tmp_path)
    assert (tmp_path / "MonthlyCharges.png").exists()


def test_plot_histogram_categorical(sample_df, tmp_path):
    plot_histogram(sample_df, "ContractType", output_dir=tmp_path)
    assert (tmp_path / "ContractType.png").exists()


def test_plot_histogram_id_column(sample_df, tmp_path):
    plot_histogram(sample_df, "CustomerID", output_dir=tmp_path)
    assert (tmp_path / "CustomerID.png").exists()


def test_plot_all_feature_histograms(sample_df, tmp_path):
    for col in ALL_FEATURE_COLS:
        plot_histogram(sample_df, col, output_dir=tmp_path)
    assert len(list(tmp_path.glob("*.png"))) == len(ALL_FEATURE_COLS)


def test_plot_correlation_matrix(sample_df, tmp_path):
    df = sample_df.drop(columns=["CustomerID"])
    plot_correlation_matrix(df, output_dir=tmp_path)
    assert (tmp_path / "correlation_matrix.png").exists()
