# --------------------------------------------------------------------------- #
# main.py – Orchestrate the full churn-prediction workflow.
# --------------------------------------------------------------------------- #

import logging
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)

from churn_prediction.config import MODEL_DIR
from churn_prediction.data import load_data, explore, prepare_features
from churn_prediction.preprocessing import split_data
from churn_prediction.models import create_rf_pipeline, create_xgb_pipeline
from churn_prediction.evaluation import (
    plot_correlation_matrix,
    train_model,
    evaluate_model,
    plot_feature_importance,
)
from churn_prediction.predict import predict_churn


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # ── Step 1: Load & explore ────────────────────────────────────────────
    df = load_data()
    explore(df)

    # ── Step 2: Feature engineering ───────────────────────────────────────
    X, y = prepare_features(df)
    plot_correlation_matrix(df.drop(columns=["CustomerID"]))

    # ── Step 3: Stratified train/test split ───────────────────────────────
    X_train, X_test, y_train, y_test = split_data(X, y)

    # ── Step 4: Build model pipelines ─────────────────────────────────────
    rf_pipeline = create_rf_pipeline()
    xgb_pipeline = create_xgb_pipeline(y_train)

    # ── Step 5: Train & evaluate ──────────────────────────────────────────
    train_model(rf_pipeline, X_train, y_train, "Random Forest")
    train_model(xgb_pipeline, X_train, y_train, "XGBoost")

    evaluate_model(rf_pipeline, X_test, y_test, "Random Forest")
    evaluate_model(xgb_pipeline, X_test, y_test, "XGBoost")

    # Feature importance
    plot_feature_importance(rf_pipeline, "Random Forest")
    plot_feature_importance(xgb_pipeline, "XGBoost")

    # ── Step 6: Save trained XGBoost pipeline ────────────────────────────
    model_path = Path(MODEL_DIR)
    model_path.mkdir(exist_ok=True)
    out = model_path / "xgb_pipeline.joblib"
    joblib.dump(xgb_pipeline, out)
    logger.info("XGBoost pipeline saved to %s", out)

    # ── Step 7: Prediction on a test customer ─────────────────────────────
    test_customer = {
        "ContractDuration": 12,
        "MonthlyCharges": 95.0,
        "InternetUsageGB": 60.0,
        "SupportCalls": 4,
        "ContractType": "Month-to-month",
        "PaymentMethod": "Electronic check",
    }

    logger.info("=" * 60)
    logger.info("  Single Customer Prediction")
    logger.info("=" * 60)
    logger.info("Customer: %s", test_customer)

    for name, pipeline in [("Random Forest", rf_pipeline), ("XGBoost", xgb_pipeline)]:
        pred_class, prob = predict_churn(test_customer, pipeline)
        label = "Churn" if pred_class == 1 else "No Churn"
        logger.info("  %s: %s  (class=%s, churn_prob=%.4f)", name, label, pred_class, prob)


if __name__ == "__main__":
    main()
