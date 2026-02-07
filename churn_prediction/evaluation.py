# --------------------------------------------------------------------------- #
# evaluation.py – Training, metrics, and feature-importance plots.
# --------------------------------------------------------------------------- #

import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    make_scorer,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

from churn_prediction.config import (
    CATEGORICAL_FEATURES,
    CV_N_REPEATS,
    CV_N_SPLITS,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
)


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


# ---- Cross-Validation --------------------------------------------------- #

def cross_validate_models(
    models: dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, dict[str, float]]:
    """Run Repeated Stratified K-Fold CV and log mean +/- std for each metric.

    Parameters
    ----------
    models : dict
        Mapping of model name to (unfitted) sklearn Pipeline.
    X, y : DataFrame / Series
        Full feature matrix and target vector (no train/test split).

    Returns
    -------
    results : dict
        ``{model_name: {"f1": mean, "recall": mean, ...}}`` for downstream
        model selection.
    """
    cv = RepeatedStratifiedKFold(n_splits=CV_N_SPLITS, n_repeats=CV_N_REPEATS, random_state=RANDOM_STATE)

    scoring = {
        "recall": make_scorer(recall_score, zero_division=0),
        "precision": make_scorer(precision_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
    }

    results: dict[str, dict[str, float]] = {}

    for name, pipeline in models.items():
        logger.info("Cross-validating %s (5 folds × 10 repeats) …", name)
        cv_results = cross_validate(
            pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1,
        )

        logger.info("=" * 60)
        logger.info("  %s – Cross-Validation Results", name)
        logger.info("=" * 60)

        means: dict[str, float] = {}
        for metric in scoring:
            key = f"test_{metric}"
            mean = cv_results[key].mean()
            std = cv_results[key].std()
            means[metric] = mean
            label = metric.replace("_", " ").title()
            logger.info("  %s: %.4f (± %.4f)", label, mean, std)

        results[name] = means

    return results
