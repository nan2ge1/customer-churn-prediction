# --------------------------------------------------------------------------- #
# models.py – Model initialisation with imbalance handling.
# --------------------------------------------------------------------------- #

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churn_prediction.config import N_ESTIMATORS, RANDOM_STATE
from churn_prediction.preprocessing import build_pipeline, build_preprocessor, compute_scale_pos_weight


def create_rf_pipeline() -> Pipeline:
    """Random Forest pipeline with class_weight='balanced'."""
    estimator = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    return build_pipeline(build_preprocessor(), estimator)


def create_xgb_pipeline(y_train) -> Pipeline:
    """XGBoost pipeline with scale_pos_weight for imbalance."""
    spw = compute_scale_pos_weight(y_train)
    estimator = XGBClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        objective="binary:logistic",
        scale_pos_weight=spw,
    )
    return build_pipeline(build_preprocessor(), estimator)
