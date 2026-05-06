"""Testy pre API endpointy správy datasetov (/api/datasets)."""
import io

import pytest
from fastapi.testclient import TestClient

_SMALL_CSV = (
    "col_a,col_b,label\n"
    "1,2,yes\n2,3,no\n3,4,yes\n4,5,no\n5,6,yes\n"
    "6,7,no\n7,8,yes\n8,9,no\n9,10,yes\n10,11,no\n"
).encode()

_NON_CSV = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00"


def test_examples_returns_four_datasets(client: TestClient) -> None:
    """GET /api/datasets/examples musi vrátit zoznam 4 zabudovaných datasetov."""
    resp = client.get("/api/datasets/examples")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 4
    names = {d["name"] for d in data}
    assert names == {"iris", "wine", "bank_marketing", "heart_disease"}


def test_load_iris_example_returns_200(client: TestClient) -> None:
    """POST /api/datasets/examples/iris/load musi vrátit 200 a nacitat dataset."""
    resp = client.post("/api/datasets/examples/iris/load")
    assert resp.status_code == 200
    body = resp.json()
    assert body["dataset_name"] == "iris"
    assert body["n_rows"] == 150
    assert body["n_columns"] == 5


def test_load_nonexistent_example_returns_404(client: TestClient) -> None:
    """POST /api/datasets/examples/nonexistent/load musi vrátit 404."""
    resp = client.post("/api/datasets/examples/nonexistent/load")
    assert resp.status_code == 404


def test_schema_after_iris_load_has_correct_types(client: TestClient) -> None:
    """GET /api/datasets/current/schema po nacitani iris musi vrátit spravne typy stlpcov."""
    client.post("/api/datasets/examples/iris/load")
    resp = client.get("/api/datasets/current/schema")
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_rows"] == 150
    col_map = {c["name"]: c["suggested_type"] for c in body["columns"]}
    assert col_map["sepal_length"] == "numeric"
    assert col_map["sepal_width"] == "numeric"
    assert col_map["species"] == "target"


def test_schema_without_dataset_returns_400(client: TestClient) -> None:
    """GET /api/datasets/current/schema bez nacitaneho datasetu musi vrátit 400."""
    resp = client.get("/api/datasets/current/schema")
    assert resp.status_code == 400


def test_upload_valid_csv_returns_200(client: TestClient) -> None:
    """POST /api/datasets/upload s platnym CSV musi vrátit 200."""
    resp = client.post(
        "/api/datasets/upload",
        files={"file": ("test.csv", io.BytesIO(_SMALL_CSV), "text/csv")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_rows"] == 10
    assert body["n_columns"] == 3


def test_upload_non_csv_returns_400(client: TestClient) -> None:
    """POST /api/datasets/upload s binarnym suborom musi vrátit 400."""
    resp = client.post(
        "/api/datasets/upload",
        files={"file": ("image.png", io.BytesIO(_NON_CSV), "image/png")},
    )
    assert resp.status_code == 400
