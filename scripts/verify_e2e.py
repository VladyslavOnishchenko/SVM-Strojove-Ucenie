import sys
sys.path.insert(0, ".")
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)
client.post("/api/datasets/examples/iris/load")
schema_r = client.get("/api/datasets/current/schema")
column_schema = {col["name"]: col["suggested_type"] for col in schema_r.json()["columns"]}

train_r = client.post("/api/train/", json={
    "column_schema": column_schema,
    "hyperparameters": {"kernel": "rbf", "C": 1.0, "gamma": "scale", "auto_tune": False},
    "test_size": 0.2,
    "cv_folds": 5
})
data = train_r.json()
print("accuracy:", data.get("accuracy"))
print("cv_mean_accuracy:", data.get("cv_mean_accuracy"))
print("best_hyperparameters:", data.get("best_hyperparameters"))
print("training_time_seconds:", data.get("training_time_seconds"))
classes = data.get("classes", [])
for cls in classes:
    metrics = data.get("per_class_metrics", {}).get(cls, {})
    print("class " + cls + ": f1_score=" + str(metrics.get("f1_score", "MISSING")))

schema_cols = schema_r.json()["columns"]
print("schema col[0] keys:", list(schema_cols[0].keys()))
print("unique_count sample:", schema_cols[0].get("unique_count"))
