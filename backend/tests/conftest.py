"""Fixtures pre API testy — reset stavu aplikácie a TestClient."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def reset_app_state():
    """Vyčistí globálny stav aplikácie pred každým testom a po ňom."""
    from backend.app.core.config import MODEL_FILENAME, STORAGE_DIR
    from backend.app.core.state import app_state

    def _clear():
        app_state.current_dataset = None
        app_state.current_dataset_name = None
        app_state.current_model = None
        app_state.model_trained_at = None
        model_path = STORAGE_DIR / MODEL_FILENAME
        if model_path.exists():
            model_path.unlink()

    _clear()
    yield
    _clear()


@pytest.fixture
def client() -> TestClient:
    """Vráti TestClient pre FastAPI aplikáciu."""
    from backend.app.main import app
    return TestClient(app)


@pytest.fixture
def iris_path() -> Path:
    """Cesta k iris.csv v data/examples/."""
    return Path("data/examples/iris.csv")
