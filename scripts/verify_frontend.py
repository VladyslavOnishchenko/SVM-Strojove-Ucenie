import sys
sys.path.insert(0, ".")
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)
r = client.get("/")
print("Status:", r.status_code)
checks = ["tailwindcss", "plotly", "chart.js", "SVM", "6366f1", "302b63"]
for c in checks:
    found = c.lower() in r.text.lower()
    print(c + ":", "FOUND" if found else "MISSING")
