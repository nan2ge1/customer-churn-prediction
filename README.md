# Customer Churn Prediction

Binary classification model to predict customer churn for a telecom scenario. Includes an exploratory data analysis, a training pipeline and a FastAPI inference service.

## Dataset

The dataset (`data/customer_churn.csv`) contains 1,000 customers with the following columns:

| Column | Type | Description |
|---------|------|-------------|
| CustomerID | string | Customer ID |
| ContractDuration | int | Contract length in months |
| MonthlyCharges | float | Monthly service cost (USD) |
| InternetUsageGB | float | Monthly internet usage (GB) |
| SupportCalls | int | Number of support calls |
| ContractType | categorical | `Month-to-month`, `One year`, `Two year` |
| PaymentMethod | categorical | `Bank transfer`, `Credit card`, `Electronic check`, `Mailed check` |
| Churn | int | Target (0 = No Churn, 1 = Churn) |

**Note:** The dataset is highly imbalanced (~1.8% churners).

## Installation

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Exploratory Data Analysis

Run the EDA script to generate histograms and a correlation matrix:

```bash
python -m churn_prediction.eda
```

This will save plots to the `figures/` directory:
- One histogram per feature comparing churners vs non-churners
- A correlation matrix heatmap of all features

## Training Pipeline

Run the full training workflow:

```bash
python -m churn_prediction.main
```

This will:
1. Load and explore the data
2. Run Repeated Stratified K-Fold Cross-Validation (5 folds × 10 repeats) on Random Forest and XGBoost models, reporting mean and std for Recall, Precision, F1, PR-AUC, and ROC-AUC
3. Select the best model by F1-Score and Recall
4. Retrain the best model on the full dataset (Golden Model)
5. Save the golden model to `models/churn_model.joblib`
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
  "churn_probability": 0.9991,
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
├── eda.py            # Exploratory Data Analysis
├── predict.py        # Single-customer prediction
├── main.py           # Training orchestrator
└── api.py            # FastAPI service
```

## License

MIT
