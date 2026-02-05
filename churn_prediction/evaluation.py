# --------------------------------------------------------------------------- #
# evaluation.py – Training, metrics, and feature-importance plots.
# --------------------------------------------------------------------------- #

import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from matplotlib.colors import TwoSlopeNorm

logger = logging.getLogger(__name__)

from churn_prediction.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET


# ---- Correlation Matrix -------------------------------------------------- #

def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """Heatmap of Pearson correlations across all columns (including Churn).

    Categorical columns are one-hot encoded before computing correlations.
    The colormap is centred on zero and scaled so that the strongest
    correlation with Churn sits at the colour-range boundary, making
    Churn-correlated features visually prominent.
    """
    # One-hot encode categoricals, keep numerics as-is
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_FEATURES)
    corr = df_encoded.corr()

    # Scale the colour map around the Churn column's correlation range
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
    plt.show()


# ---- Training ------------------------------------------------------------ #

def train_model(pipeline: Pipeline, X_train, y_train, name: str) -> Pipeline:
    """Fit a pipeline and print confirmation."""
    logger.info("Training %s …", name)
    pipeline.fit(X_train, y_train)
    logger.info("%s training complete.", name)
    return pipeline


# ---- Metrics ------------------------------------------------------------- #

def evaluate_model(pipeline: Pipeline, X_test, y_test, name: str) -> None:
    """Print classification report and confusion matrix."""
    y_pred = pipeline.predict(X_test)

    logger.info("=" * 60)
    logger.info("  %s – Evaluation", name)
    logger.info("=" * 60)

    logger.info("Classification Report:\n%s", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    logger.info("Confusion Matrix:\n%s", cm)

    # Visual confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix – {name}")
    plt.tight_layout()
    plt.show()


# ---- Feature Importance -------------------------------------------------- #

def _get_feature_names(pipeline: Pipeline) -> list[str]:
    """Reconstruct feature names from the ColumnTransformer inside the pipeline."""
    preprocessor = pipeline.named_steps["preprocessor"]
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    return NUMERICAL_FEATURES + cat_names


def plot_feature_importance(pipeline: Pipeline, name: str) -> None:
    """Bar chart of feature importances for tree-based models."""
    classifier = pipeline.named_steps["classifier"]

    # Both RandomForest and XGBoost expose feature_importances_
    importances = classifier.feature_importances_
    feature_names = _get_feature_names(pipeline)

    idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(np.array(feature_names)[idx], importances[idx])
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance – {name}")
    plt.tight_layout()
    plt.show()
