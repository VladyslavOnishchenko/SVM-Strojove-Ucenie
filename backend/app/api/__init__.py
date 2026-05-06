"""API balík — FastAPI routery pre datasety, trénovanie, predikciu a informácie o modeli."""
from backend.app.api import datasets, model_info, prediction, training

__all__ = ["datasets", "training", "prediction", "model_info"]
