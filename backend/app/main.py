"""Hlavný vstupný bod FastAPI aplikácie."""
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.api import datasets, model_info, prediction, training
from backend.app.core.config import PROJECT_NAME

app = FastAPI(
    title=PROJECT_NAME,
    description="Web aplikácia pre trénovanie a používanie SVM klasifikátorov na CSV dátach.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API endpointy — musia byť zaregistrované PRED statickým mountom
@app.get("/api/health")
async def health_check() -> dict:
    """Kontrola stavu aplikácie."""
    return {"status": "ok"}

app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(training.router, prefix="/api/train", tags=["training"])
app.include_router(prediction.router, prefix="/api/predict", tags=["prediction"])
app.include_router(model_info.router, prefix="/api/model", tags=["model"])

frontend_dir = Path(__file__).parent.parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
