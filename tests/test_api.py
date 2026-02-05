"""Tests for churn_prediction.api."""

import numpy as np
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


VALID_CUSTOMER = {
    "ContractDuration": 12,
    "MonthlyCharges": 95.0,
    "InternetUsageGB": 60.0,
    "SupportCalls": 4,
    "ContractType": "Month-to-month",
    "PaymentMethod": "Electronic check",
}


@pytest.fixture()
def client():
    """TestClient with a mocked model."""
    import churn_prediction.api as api_module

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.05, 0.95]])

    original_model = api_module.model
    api_module.model = mock_model
    yield TestClient(api_module.app)
    api_module.model = original_model


@pytest.fixture()
def client_no_model():
    """TestClient with model set to None."""
    import churn_prediction.api as api_module

    original_model = api_module.model
    api_module.model = None
    yield TestClient(api_module.app)
    api_module.model = original_model


def test_predict_success(client):
    resp = client.post("/predict", json=VALID_CUSTOMER)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] == 1
    assert body["churn_probability"] == 0.95
    assert body["risk_level"] == "High"


def test_predict_low_risk(client):
    import churn_prediction.api as api_module
    api_module.model.predict.return_value = np.array([0])
    api_module.model.predict_proba.return_value = np.array([[0.8, 0.2]])

    resp = client.post("/predict", json=VALID_CUSTOMER)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] == 0
    assert body["risk_level"] == "Low"


def test_predict_no_model(client_no_model):
    resp = client_no_model.post("/predict", json=VALID_CUSTOMER)
    assert resp.status_code == 500
    assert "Model not loaded" in resp.json()["detail"]


def test_predict_invalid_contract_type(client):
    bad = {**VALID_CUSTOMER, "ContractType": "Invalid"}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_missing_field(client):
    incomplete = {k: v for k, v in VALID_CUSTOMER.items() if k != "MonthlyCharges"}
    resp = client.post("/predict", json=incomplete)
    assert resp.status_code == 422


def test_predict_negative_charges(client):
    bad = {**VALID_CUSTOMER, "MonthlyCharges": -10.0}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_zero_contract_duration(client):
    bad = {**VALID_CUSTOMER, "ContractDuration": 0}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422
