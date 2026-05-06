"""Testy pre API endpoint trénovania SVM modelu (/api/train)."""
from pathlib import Path

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


def _load_iris(client: TestClient) -> None:
    client.post("/api/datasets/examples/iris/load")


def test_train_on_iris_returns_200(client: TestClient) -> None:
    """Trénovanie na Iris s platnou schemou musi vrátit 200 a TrainingResponse."""
    _load_iris(client)
    resp = client.post(
        "/api/train/",
        json={"column_schema": _IRIS_SCHEMA, "hyperparameters": _IRIS_HYPERPARAMS},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "accuracy" in body
    assert body["accuracy"] > 0.90
    assert len(body["classes"]) == 3


def test_train_without_dataset_returns_400(client: TestClient) -> None:
    """Trénovanie bez nacitaneho datasetu musi vrátit 400."""
    resp = client.post(
        "/api/train/",
        json={"column_schema": _IRIS_SCHEMA, "hyperparameters": _IRIS_HYPERPARAMS},
    )
    assert resp.status_code == 400


def test_train_with_invalid_schema_no_target_returns_422(client: TestClient) -> None:
    """Trénovanie so schemou bez TARGET stlpca musi vrátit 422."""
    _load_iris(client)
    schema_no_target = {
        "sepal_length": "numeric",
        "sepal_width": "numeric",
        "petal_length": "numeric",
        "petal_width": "numeric",
    }
    resp = client.post(
        "/api/train/",
        json={"column_schema": schema_no_target, "hyperparameters": _IRIS_HYPERPARAMS},
    )
    assert resp.status_code == 422


def test_train_creates_model_file(client: TestClient) -> None:
    """Po uspesnom trenovaní musi existovat súbor modelu v backend/storage/."""
    _load_iris(client)
    client.post(
        "/api/train/",
        json={"column_schema": _IRIS_SCHEMA, "hyperparameters": _IRIS_HYPERPARAMS},
    )
    model_path = Path("backend/storage/current_model.joblib")
    assert model_path.exists()
