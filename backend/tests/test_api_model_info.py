"""Testy pre API endpointy informácií o modeli (/api/model)."""
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


def _train_iris(client: TestClient) -> None:
    client.post("/api/datasets/examples/iris/load")
    client.post(
        "/api/train/",
        json={"column_schema": _IRIS_SCHEMA, "hyperparameters": _IRIS_HYPERPARAMS},
    )


def test_status_before_training_is_not_trained(client: TestClient) -> None:
    """GET /api/model/status pred trénovanim musi vrátit is_trained=False."""
    resp = client.get("/api/model/status")
    assert resp.status_code == 200
    assert resp.json()["is_trained"] is False


def test_status_after_training_is_trained(client: TestClient) -> None:
    """GET /api/model/status po trenovaní musi vrátit is_trained=True so zoznamom tried."""
    _train_iris(client)
    resp = client.get("/api/model/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_trained"] is True
    assert len(body["classes"]) == 3
    assert body["dataset_name"] == "iris"


def test_download_model_after_training(client: TestClient) -> None:
    """GET /api/model/download po trenovaní musi vrátit súbor s obsahom."""
    _train_iris(client)
    resp = client.get("/api/model/download")
    assert resp.status_code == 200
    assert len(resp.content) > 0


def test_download_model_before_training_returns_404(client: TestClient) -> None:
    """GET /api/model/download pred trénovaním musi vrátit 404."""
    resp = client.get("/api/model/download")
    assert resp.status_code == 404
