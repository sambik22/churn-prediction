from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.schemas import Customer

EXAMPLE = Customer.model_config["json_schema_extra"]["example"]


@pytest.fixture()
def client(trained_model, monkeypatch):
    import api.main as main

    monkeypatch.setattr(main, "MODEL_PATH", Path(trained_model[1]))
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_predict_with_explanation(client):
    resp = client.post("/predict", json=EXAMPLE)
    assert resp.status_code == 200
    body = resp.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["risk_level"] in {"low", "medium", "high"}
    assert len(body["top_factors"]) == 5


def test_predict_validation_error(client):
    bad = {**EXAMPLE, "Contract": "Forever"}
    assert client.post("/predict", json=bad).status_code == 422


def test_batch(client):
    resp = client.post("/predict/batch", json={"customers": [EXAMPLE, EXAMPLE]})
    assert resp.status_code == 200
    preds = resp.json()["predictions"]
    assert len(preds) == 2
    assert preds[0]["top_factors"] is None


def test_model_info(client):
    info = client.get("/model").json()
    assert info["model_name"] == "logreg-test"
