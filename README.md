# Customer Churn Prediction

Binary classification model to predict customer churn for a telecom scenario. Includes a training pipeline and a FastAPI inference service.

## Dataset

The dataset (`data/customer_churn.csv`) contains 1,000 customers with the following features:

| Feature | Type | Description |
|---------|------|-------------|
| ContractDuration | int | Contract length in months |
| MonthlyCharges | float | Monthly service cost (USD) |
| InternetUsageGB | float | Monthly internet usage (GB) |
| SupportCalls | int | Number of support calls |
| ContractType | categorical | Month-to-month, One year, Two year |
| PaymentMethod | categorical | Bank transfer, Credit card, Electronic check, Mailed check |
| Churn | int | Target (0 = No Churn, 1 = Churn) |

**Note:** The dataset is highly imbalanced (~1.8% churners).

## Installation

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Training Pipeline

Run the full training workflow:

```bash
python -m churn_prediction.main
```

This will:
1. Load and explore the data
2. Plot feature correlation matrix
3. Train Random Forest and XGBoost models (with class imbalance handling)
4. Evaluate on test dataset
5. Save the XGBoost model to `models/xgb_pipeline.joblib`
6. Run a sample prediction

## API Service

The API is deployed on Azure Cloud at https://customer-churn-prediction-demo.azurewebsites.net/docs.

To run locally, start the FastAPI server:

```bash
python -m churn_prediction.api
```

The local API runs at `http://127.0.0.1:8000`. Interactive docs available at `/docs`.

### Prediction Endpoint

**POST** `/predict`

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ContractDuration": 12,
    "MonthlyCharges": 95.0,
    "InternetUsageGB": 60.0,
    "SupportCalls": 4,
    "ContractType": "Month-to-month",
    "PaymentMethod": "Electronic check"
  }'
```

**Response:**

```json
{
  "prediction": 1,
  "churn_probability": 0.9949,
  "risk_level": "High"
}
```

## Project Structure

```
churn_prediction/
├── config.py         # Constants and paths
├── data.py           # Data loading
├── preprocessing.py  # Pipeline construction
├── models.py         # Model creation
├── evaluation.py     # Metrics and plots
├── predict.py        # Single-customer prediction
├── main.py           # Training orchestrator
└── api.py            # FastAPI service
```

## License

MIT
