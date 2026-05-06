import sys, json
sys.path.insert(0, ".")
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_dataset(name):
    client.post(f"/api/datasets/examples/{name}/load")
    schema_r = client.get("/api/datasets/current/schema")
    cols = schema_r.json()["columns"]
    column_schema = {col["name"]: col["suggested_type"] for col in cols}

    client.post("/api/train/", json={
        "column_schema": column_schema,
        "hyperparameters": {"kernel": "rbf", "C": 1.0, "gamma": "scale", "auto_tune": False},
        "test_size": 0.2, "cv_folds": 5
    })

    feature_cols = [c for c in cols if c["suggested_type"] != "target"]
    payload = {}
    for col in feature_cols:
        t = col["suggested_type"]
        val = col["sample_values"][0] if col.get("sample_values") else None
        if t == "numeric":
            try:
                payload[col["name"]] = float(val)
            except Exception:
                payload[col["name"]] = 0.0
        else:
            # binary and categorical: send original string value
            payload[col["name"]] = str(val) if val is not None else ""

    r = client.post("/api/predict/", json={"input_data": payload})
    result = r.json() if r.status_code == 200 else r.text[:200]
    print(f"{name}: status={r.status_code} predicted={result.get('predicted_class') if r.status_code == 200 else result}")

for ds in ["iris", "bank_marketing", "heart_disease", "wine"]:
    test_dataset(ds)
