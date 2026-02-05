# --------------------------------------------------------------------------- #
# predict.py – Single-customer prediction function.
# --------------------------------------------------------------------------- #

import pandas as pd
from sklearn.pipeline import Pipeline


def predict_churn(customer_dict: dict, model: Pipeline) -> tuple[int, float]:
    """
    Predict churn for a single customer.

    Parameters
    ----------
    customer_dict : dict
        Feature values for one customer (must match training feature names).
    model : Pipeline
        A fitted sklearn Pipeline (preprocessor + classifier).

    Returns
    -------
    predicted_class : int
        0 (no churn) or 1 (churn).
    churn_probability : float
        Estimated probability of churn (class 1).
    """
    customer_df = pd.DataFrame([customer_dict])
    predicted_class = int(model.predict(customer_df)[0])
    churn_probability = float(model.predict_proba(customer_df)[0, 1])
    return predicted_class, churn_probability
