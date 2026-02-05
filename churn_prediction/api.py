# --------------------------------------------------------------------------- #
# api.py – FastAPI service for customer churn prediction.
# --------------------------------------------------------------------------- #

import logging
from enum import Enum
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from churn_prediction.config import MODEL_DIR

# ── App ────────────────────────────────────────────────────────────────────

app = FastAPI(title="Customer Churn Prediction API", version="1.0")

# ── Enums (categorical constraints) ───────────────────────────────────────


class ContractTypeEnum(str, Enum):
    month_to_month = "Month-to-month"
    one_year = "One year"
    two_year = "Two year"


class PaymentMethodEnum(str, Enum):
    bank_transfer = "Bank transfer"
    credit_card = "Credit card"
    electronic_check = "Electronic check"
    mailed_check = "Mailed check"


# ── Pydantic data model ──────────────────────────────────────────────────


class CustomerData(BaseModel):
    ContractDuration: int = Field(..., ge=1, description="Duration in months")
    MonthlyCharges: float = Field(..., gt=0, description="Monthly cost")
    InternetUsageGB: float = Field(..., ge=0, description="Internet usage in GB")
    SupportCalls: int = Field(..., ge=0, description="Number of support calls")
    ContractType: ContractTypeEnum
    PaymentMethod: PaymentMethodEnum

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "ContractDuration": 12,
                    "MonthlyCharges": 95.0,
                    "InternetUsageGB": 60.0,
                    "SupportCalls": 4,
                    "ContractType": "Month-to-month",
                    "PaymentMethod": "Electronic check",
                }
            ]
        }
    }


# ── Model loading ────────────────────────────────────────────────────────

MODEL_PATH = Path(MODEL_DIR) / "xgb_pipeline.joblib"
model = None

try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded from %s", MODEL_PATH)
except Exception as exc:
    logger.warning("Could not load model from %s: %s", MODEL_PATH, exc)


# ── Prediction endpoint ──────────────────────────────────────────────────


@app.post("/predict")
def predict(customer: CustomerData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    # Build a DataFrame with column names matching the training data
    row = {
        "ContractDuration": customer.ContractDuration,
        "MonthlyCharges": customer.MonthlyCharges,
        "InternetUsageGB": customer.InternetUsageGB,
        "SupportCalls": customer.SupportCalls,
        "ContractType": customer.ContractType.value,
        "PaymentMethod": customer.PaymentMethod.value,
    }
    df = pd.DataFrame([row])

    try:
        prediction = int(model.predict(df)[0])
        churn_probability = round(float(model.predict_proba(df)[0, 1]), 4)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    return {
        "prediction": prediction,
        "churn_probability": churn_probability,
        "risk_level": "High" if churn_probability > 0.5 else "Low",
    }


# ── Run ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
