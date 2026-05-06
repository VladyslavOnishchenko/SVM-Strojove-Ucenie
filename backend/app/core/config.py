"""
Konfigurácia a nastavenia aplikácie.
"""
from pathlib import Path

PROJECT_NAME = "SVM Strojové Učenie"
STORAGE_DIR = Path(__file__).parent.parent.parent / "storage"
MODEL_FILENAME = "current_model.joblib"
