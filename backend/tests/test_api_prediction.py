"""Testy pre API endpoint predikcie (/api/predict)."""
import pytest
from fastapi.testclient import TestClient

_IRIS_SCHEMA = {
    "sepal_length": "numeric",
    "sepal_width": "numeric",
    "petal_length": "numeric",
    "petal_width": "numeric",
    "species": "target",
}
_IRIS_HYPERPARAMS = {"kernel": "rbf", "C": 1.0, "gamma": "scale", "auto_tune": False}

_IRIS_SAMPLE = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}


def _train_iris(client: TestClient) -> None:
    client.post("/api/datasets/examples/iris/load")
    client.post(
        "/api/train/",
        json={"column_schema": _IRIS_SCHEMA, "hyperparameters": _IRIS_HYPERPARAMS},
    )


def test_predict_after_training_returns_valid_class(client: TestClient) -> None:
    """Predikcia po trénovaní na Iris musi vrátit platnu triedu."""
    _train_iris(client)
    resp = client.post("/api/predict/", json={"input_data": _IRIS_SAMPLE})
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_class"] in {"setosa", "versicolor", "virginica"}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-5


def test_predict_without_model_returns_400(client: TestClient) -> None:
    """Predikcia bez natrénovaného modelu musi vrátit 400."""
    resp = client.post("/api/predict/", json={"input_data": _IRIS_SAMPLE})
    assert resp.status_code == 400


def test_predict_with_missing_columns_returns_422(client: TestClient) -> None:
    """Predikcia s chybajucimi stlpcami vo vstupe musi vrátit 422."""
    _train_iris(client)
    incomplete = {"sepal_length": 5.1}
    resp = client.post("/api/predict/", json={"input_data": incomplete})
    assert resp.status_code == 422
