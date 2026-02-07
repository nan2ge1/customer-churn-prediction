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

from churn_prediction.config import (
    CATEGORICAL_FEATURES,
    ID_COL,
    TARGET,
)
from churn_prediction.data import load_data

logger = logging.getLogger(__name__)

FIGURES_DIR = Path("figures")
FIGSIZE = (10, 6)

ALL_FEATURE_COLS = [
    "CustomerID",
    "ContractDuration",
    "MonthlyCharges",
    "InternetUsageGB",
    "SupportCalls",
    "ContractType",
    "PaymentMethod",
]

# Set global style
sns.set_theme(style="whitegrid", context="talk")
# Custom palette: distinct colors for Churn vs No Churn
# e.g., muted blue for No Churn, muted red/orange for Churn
CHURN_PALETTE = {0: "#4C72B0", 1: "#C44E52"}  # Seaborn deep blue and red
CHURN_LABELS = {0: "No Churn", 1: "Churn"}


def plot_histogram(df: pd.DataFrame, column: str, output_dir: Path = FIGURES_DIR) -> None:
    """Plot overlaid histograms/density plots for churners vs non-churners.

    Refined for better aesthetics.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)

    if column == ID_COL:
        # CustomerID: plot count distribution of last 4 digits
        id_ints = df[column].str[-4:].astype(int)
        
        # Prepare data for plotting
        data_no = id_ints[df[TARGET] == 0]
        data_yes = id_ints[df[TARGET] == 1]
        
        bins = np.arange(id_ints.min(), id_ints.max() + 2)
        
        ax.hist(data_no, bins=bins, alpha=0.6, label=CHURN_LABELS[0], color=CHURN_PALETTE[0], edgecolor=None)
        ax.hist(data_yes, bins=bins, alpha=0.6, label=CHURN_LABELS[1], color=CHURN_PALETTE[1], edgecolor=None)
        
        ax.set_ylabel("Count")
        ax.set_xlabel("CustomerID (Last 4 Digits)")
        
    elif not pd.api.types.is_numeric_dtype(df[column]):
        # Categorical: Density bar chart
        categories = sorted(df[column].unique())
        x_idx = np.arange(len(categories))
        width = 0.35

        non_churners = df.loc[df[TARGET] == 0, column]
        churners = df.loc[df[TARGET] == 1, column]

        counts_no = non_churners.value_counts().reindex(categories, fill_value=0)
        counts_yes = churners.value_counts().reindex(categories, fill_value=0)

        density_no = counts_no / counts_no.sum()
        # Handle case where there are no churners (division by zero protection)
        total_churn = counts_yes.sum()
        density_yes = counts_yes / total_churn if total_churn > 0 else counts_yes

        ax.bar(x_idx - width / 2, density_no, width, label=CHURN_LABELS[0], 
               color=CHURN_PALETTE[0], alpha=0.8)
        ax.bar(x_idx + width / 2, density_yes, width, label=CHURN_LABELS[1], 
               color=CHURN_PALETTE[1], alpha=0.8)
        
        ax.set_xticks(x_idx)
        ax.set_xticklabels(categories)
        ax.set_ylabel("Density")
        ax.set_xlabel(column)

    else:
        # Numerical: Density histograms
        non_churners = df.loc[df[TARGET] == 0, column]
        churners = df.loc[df[TARGET] == 1, column]
        
        # Compute common bins
        combined = df[column].dropna()
        bins = np.histogram_bin_edges(combined, bins="auto")

        ax.hist(non_churners, bins=bins, density=True, alpha=0.6, 
                label=CHURN_LABELS[0], color=CHURN_PALETTE[0], edgecolor="white", linewidth=0.5)
        ax.hist(churners, bins=bins, density=True, alpha=0.6, 
                label=CHURN_LABELS[1], color=CHURN_PALETTE[1], edgecolor="white", linewidth=0.5)
        
        ax.set_ylabel("Density")
        ax.set_xlabel(column)

    # Common aesthetics
    ax.set_title(f"{column}", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{column}.png", dpi=150)
    plt.close(fig)
    logger.info("Saved histogram: %s/%s.png", output_dir, column)


def plot_correlation_matrix(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> None:
    """Heatmap of Pearson correlations using a masked upper triangle."""
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_FEATURES)
    corr = df_encoded.corr()

    # Mask for the main diagonal only
    mask = np.eye(len(corr), dtype=bool)

    # Calculate vmin/vmax based on max correlation with Churn (excluding self)
    churn_corr = corr[TARGET].drop(TARGET)
    vmax = churn_corr.abs().max()

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=vmax, 
        vmin=-vmax,
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .5},
        annot=True,
        fmt=".2f",
        annot_kws={"size": 9},
        ax=ax
    )

    ax.set_title("Feature Correlation Matrix", loc="left", fontweight="bold", fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "correlation_matrix.png", dpi=150)
    plt.close(fig)
    logger.info("Saved correlation matrix: %s/correlation_matrix.png", output_dir)


def run_eda() -> None:
    """Run the full EDA pipeline: load data, plot histograms, plot correlation matrix."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    # Ensure clear figures
    plt.close("all")

    df = load_data()

    logger.info("Starting EDA...")
    for col in ALL_FEATURE_COLS:
        plot_histogram(df, col)

    plot_correlation_matrix(df.drop(columns=[ID_COL]))

    logger.info("EDA complete. Figures saved to %s/", FIGURES_DIR)


if __name__ == "__main__":
    run_eda()
