# --------------------------------------------------------------------------- #
# main.py – Orchestrate the full churn-prediction workflow.
# --------------------------------------------------------------------------- #

import logging
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)

from churn_prediction.config import ID_COL, MODEL_DIR, MODEL_FILENAME
from churn_prediction.data import load_data, explore, prepare_features
from churn_prediction.models import create_rf_pipeline, create_xgb_pipeline
from churn_prediction.evaluation import (
    plot_correlation_matrix,
    cross_validate_models,
    train_model,
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
    plot_correlation_matrix(df.drop(columns=[ID_COL]))

    # ── Step 3: Build model pipelines ─────────────────────────────────────
    rf_pipeline = create_rf_pipeline()
    xgb_pipeline = create_xgb_pipeline(y)

    models = {
        "Random Forest": rf_pipeline,
        "XGBoost": xgb_pipeline,
    }

    # ── Step 4: Repeated Stratified K-Fold Cross-Validation ───────────────
    cv_results = cross_validate_models(models, X, y)

    # ── Step 5: Select best model by F1 then Recall ───────────────────────
    best_name = max(
        cv_results,
        key=lambda n: (cv_results[n]["f1"], cv_results[n]["recall"]),
    )
    logger.info("Best model: %s", best_name)

    # ── Step 6: Retrain best pipeline on ALL data (Golden Model) ──────────
    # Re-initialise a fresh pipeline so CV fold state is discarded
    if best_name == "Random Forest":
        golden_pipeline = create_rf_pipeline()
    else:
        golden_pipeline = create_xgb_pipeline(y)

    train_model(golden_pipeline, X, y, f"{best_name} (Golden Model)")
    plot_feature_importance(golden_pipeline, f"{best_name} (Golden Model)")

    # ── Step 7: Save golden model ─────────────────────────────────────────
    model_path = Path(MODEL_DIR)
    model_path.mkdir(exist_ok=True)
    out = model_path / MODEL_FILENAME
    joblib.dump(golden_pipeline, out)
    logger.info(
        "Robust evaluation complete. Final model trained on full dataset and saved."
    )

    # ── Step 8: Prediction on a test customer ─────────────────────────────
    test_customer = {
        "ContractDuration": 12,
        "MonthlyCharges": 95.0,
        "InternetUsageGB": 60.0,
        "SupportCalls": 4,
        "ContractType": "Month-to-month",
        "PaymentMethod": "Electronic check",
    }

    logger.info("=" * 60)
    logger.info("  Single Customer Prediction (Golden Model)")
    logger.info("=" * 60)
    logger.info("Customer: %s", test_customer)

    pred_class, prob = predict_churn(test_customer, golden_pipeline)
    label = "Churn" if pred_class == 1 else "No Churn"
    logger.info("  %s: %s  (class=%s, churn_prob=%.4f)", best_name, label, pred_class, prob)


if __name__ == "__main__":
    main()
