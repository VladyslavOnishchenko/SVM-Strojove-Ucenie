"""
Hlavný vstupný bod FastAPI aplikácie.
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.core.config import PROJECT_NAME

# Inicializácia FastAPI aplikácie
app = FastAPI(
    title=PROJECT_NAME,
    description="Web aplikácia pre trénovanie a používanie SVM klasifikátorov na CSV dátach.",
)

# CORS middleware pre vývoj - povolenie všetkých pôvodov
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API endpointy


@app.get("/api/health")
async def health_check() -> dict:
    """
    Kontrola stavu aplikácie.
    
    Returns:
        dict: {"status": "ok"}
    """
    return {"status": "ok"}


# Statické súbory - frontend
frontend_dir = Path(__file__).parent.parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
