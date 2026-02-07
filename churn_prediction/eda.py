# --------------------------------------------------------------------------- #
# eda.py – Exploratory Data Analysis: histograms and correlation matrix.
# --------------------------------------------------------------------------- #

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

from churn_prediction.config import (
    CATEGORICAL_FEATURES,
    ID_COL,
    TARGET,
)
from churn_prediction.data import load_data

logger = logging.getLogger(__name__)

FIGURES_DIR = Path("figures")
FIGSIZE = (8, 5)

ALL_FEATURE_COLS = [
    "CustomerID",
    "ContractDuration",
    "MonthlyCharges",
    "InternetUsageGB",
    "SupportCalls",
    "ContractType",
    "PaymentMethod",
]


def plot_histogram(df: pd.DataFrame, column: str, output_dir: Path = FIGURES_DIR) -> None:
    """Plot overlaid histograms for churners vs non-churners for a single column.

    Both groups share the same bin edges so the distributions are directly
    comparable.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)

    if column == ID_COL:
        # Convert last 4 digits of CustomerID to integers and histogram them
        id_ints = df[column].str[-4:].astype(int)
        non_churners = id_ints[df[TARGET] == 0]
        churners = id_ints[df[TARGET] == 1]
        bins = np.arange(id_ints.min(), id_ints.max() + 2)  # bin width = 1

        ax.hist(non_churners, bins=bins, alpha=0.5, label="No Churn")
        ax.hist(churners, bins=bins, alpha=0.5, label="Churn")
        ax.set_xlabel(ID_COL)
        ax.set_ylabel("Count")
    elif not pd.api.types.is_numeric_dtype(df[column]):
        # Categorical column: use density (normalised to sum to 1 per group)
        categories = sorted(df[column].unique())
        x_idx = np.arange(len(categories))
        width = 0.35

        non_churners = df.loc[df[TARGET] == 0, column]
        churners = df.loc[df[TARGET] == 1, column]

        counts_no = non_churners.value_counts().reindex(categories, fill_value=0)
        counts_yes = churners.value_counts().reindex(categories, fill_value=0)

        density_no = counts_no / counts_no.sum()
        density_yes = counts_yes / counts_yes.sum() if counts_yes.sum() > 0 else counts_yes

        ax.bar(x_idx - width / 2, density_no, width, label="No Churn", alpha=0.5)
        ax.bar(x_idx + width / 2, density_yes, width, label="Churn", alpha=0.5)
        ax.set_xticks(x_idx)
        ax.set_xticklabels(categories)
        ax.set_xlabel(column)
        ax.set_ylabel("Density")
    else:
        # Numerical column: compute shared bin edges
        non_churners = df.loc[df[TARGET] == 0, column]
        churners = df.loc[df[TARGET] == 1, column]
        combined = df[column].dropna()
        bins = np.histogram_bin_edges(combined, bins="auto")

        ax.hist(non_churners, bins=bins, density=True, alpha=0.5, label="No Churn")
        ax.hist(churners, bins=bins, density=True, alpha=0.5, label="Churn")
        ax.set_xlabel(column)
        ax.set_ylabel("Density")
    ax.set_title(f"Distribution of {column} by Churn Status")
    ax.legend()
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{column}.png")
    plt.close(fig)
    logger.info("Saved histogram: %s/%s.png", output_dir, column)


def plot_correlation_matrix(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> None:
    """Heatmap of Pearson correlations across all columns (including Churn).

    Categorical columns are one-hot encoded before computing correlations.
    The colormap is centred on zero and scaled so that the strongest
    correlation with Churn sits at the colour-range boundary.
    """
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_FEATURES)
    corr = df_encoded.corr()

    churn_corr = corr[TARGET].drop(TARGET)
    vmax = churn_corr.abs().max()
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        norm=norm,
        ax=ax,
    )
    ax.set_title("Feature Correlation Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "correlation_matrix.png")
    plt.close(fig)
    logger.info("Saved correlation matrix: %s/correlation_matrix.png", output_dir)


def run_eda() -> None:
    """Run the full EDA pipeline: load data, plot histograms, plot correlation matrix."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    df = load_data()

    for col in ALL_FEATURE_COLS:
        plot_histogram(df, col)

    plot_correlation_matrix(df.drop(columns=[ID_COL]))

    logger.info("EDA complete. Figures saved to %s/", FIGURES_DIR)


if __name__ == "__main__":
    run_eda()
