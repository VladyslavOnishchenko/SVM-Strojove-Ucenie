"""Testy SVM modelu na dátovej sade Iris — číselné príznaky, 3 triedy."""
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

IRIS_HYPERPARAMS = {
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",
    "auto_tune": False,
}


@pytest.fixture(scope="module")
def trained_iris() -> tuple[SVMClassifier, dict]:
    """Fixture — natrénovaný model Iris a výsledky trénovania."""
    df = pd.read_csv(IRIS_PATH)
    model = SVMClassifier(IRIS_SCHEMA, IRIS_HYPERPARAMS)
    results = model.fit(df)
    return model, results


def test_iris_accuracy(trained_iris: tuple) -> None:
    """Presnosť na Iris musí byť vyššia ako 0.90."""
    _, results = trained_iris
    assert results["accuracy"] > 0.90, f"Presnosť {results['accuracy']:.3f} je príliš nízka"


def test_iris_confusion_matrix_shape(trained_iris: tuple) -> None:
    """Matica zámen pre Iris musí byť 3×3."""
    _, results = trained_iris
    cm = results["confusion_matrix"]
    assert len(cm) == 3
    assert all(len(row) == 3 for row in cm)


def test_iris_three_classes(trained_iris: tuple) -> None:
    """Model musí rozpoznať presne 3 triedy."""
    _, results = trained_iris
    assert len(results["classes"]) == 3


def test_iris_cv_scores_present(trained_iris: tuple) -> None:
    """Výsledky musia obsahovať CV skóre."""
    _, results = trained_iris
    assert len(results["cv_scores"]) == 5
    assert 0.0 <= results["cv_mean_accuracy"] <= 1.0


def test_iris_predict_class_valid(trained_iris: tuple) -> None:
    """Predikovaná trieda musí byť jednou z troch druhov Iris."""
    model, _ = trained_iris
    sample = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    result = model.predict(sample)
    assert result["predicted_class"] in {"setosa", "versicolor", "virginica"}


def test_iris_predict_probabilities_sum_to_one(trained_iris: tuple) -> None:
    """Súčet pravdepodobností predikcie musí byť ~1.0."""
    model, _ = trained_iris
    sample = {
        "sepal_length": 6.3,
        "sepal_width": 3.3,
        "petal_length": 6.0,
        "petal_width": 2.5,
    }
    result = model.predict(sample)
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-5


def test_iris_predict_returns_all_classes(trained_iris: tuple) -> None:
    """Slovník pravdepodobností musí obsahovať záznamy pre všetky 3 triedy."""
    model, _ = trained_iris
    sample = {
        "sepal_length": 5.8,
        "sepal_width": 2.7,
        "petal_length": 5.1,
        "petal_width": 1.9,
    }
    result = model.predict(sample)
    assert len(result["probabilities"]) == 3
