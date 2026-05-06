"""Testy automatického ladenia hyperparametrov SVM (GridSearchCV)."""
from pathlib import Path

import pandas as pd
import pytest

from backend.app.ml.model import SVMClassifier
from backend.app.ml.types import ColumnType

IRIS_PATH = Path("data/examples/iris.csv")

IRIS_SCHEMA = {
    "sepal_length": ColumnType.NUMERIC,
    "sepal_width": ColumnType.NUMERIC,
    "petal_length": ColumnType.NUMERIC,
    "petal_width": ColumnType.NUMERIC,
    "species": ColumnType.TARGET,
}


@pytest.mark.slow
def test_autotune_completes_and_accuracy() -> None:
    """Auto-tune musí dokončiť tréning a dosiahnuť presnosť > 0.90 na Iris."""
    df = pd.read_csv(IRIS_PATH)
    model = SVMClassifier(IRIS_SCHEMA, {"auto_tune": True})
    results = model.fit(df)

    assert results["accuracy"] > 0.90, f"Presnosť {results['accuracy']:.3f} je príliš nízka"


@pytest.mark.slow
def test_autotune_returns_best_params() -> None:
    """Auto-tune musí vrátiť najlepšie nájdené hyperparametre."""
    df = pd.read_csv(IRIS_PATH)
    model = SVMClassifier(IRIS_SCHEMA, {"auto_tune": True})
    results = model.fit(df)

    assert results["best_params"] is not None
    assert "svm__kernel" in results["best_params"]
    assert "svm__C" in results["best_params"]


@pytest.mark.slow
def test_autotune_predict_after_training() -> None:
    """Model natrénovaný s auto-tune musí vedieť predikovia na novom vzore."""
    df = pd.read_csv(IRIS_PATH)
    model = SVMClassifier(IRIS_SCHEMA, {"auto_tune": True})
    model.fit(df)

    sample = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    result = model.predict(sample)
    assert result["predicted_class"] in {"setosa", "versicolor", "virginica"}
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-5
