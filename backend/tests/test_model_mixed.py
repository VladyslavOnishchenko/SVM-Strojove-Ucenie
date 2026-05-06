"""Testy SVM modelu na zmiešanej dátovej sade (numerické, kategorické, binárne príznaky)."""
from pathlib import Path

import pandas as pd
import pytest

from backend.app.ml.model import SVMClassifier
from backend.app.ml.types import ColumnType

BANK_PATH = Path("data/examples/bank_marketing_sample.csv")

BANK_SCHEMA = {
    "age": ColumnType.NUMERIC,
    "job": ColumnType.CATEGORICAL,
    "marital": ColumnType.CATEGORICAL,
    "education": ColumnType.CATEGORICAL,
    "default": ColumnType.BINARY,
    "balance": ColumnType.NUMERIC,
    "housing": ColumnType.BINARY,
    "loan": ColumnType.BINARY,
    "duration": ColumnType.NUMERIC,
    "subscribed": ColumnType.TARGET,
}

BANK_HYPERPARAMS = {
    "kernel": "linear",
    "C": 1.0,
    "gamma": "scale",
    "auto_tune": False,
}

SAMPLE_INPUT = {
    "age": 35,
    "job": "admin",
    "marital": "single",
    "education": "tertiary",
    "default": "no",
    "balance": 1000,
    "housing": "yes",
    "loan": "no",
    "duration": 300,
}


@pytest.fixture(scope="module")
def trained_bank() -> tuple[SVMClassifier, dict]:
    """Fixture — natrénovaný model na bank_marketing dátach a výsledky."""
    df = pd.read_csv(BANK_PATH)
    model = SVMClassifier(BANK_SCHEMA, BANK_HYPERPARAMS)
    results = model.fit(df)
    return model, results


def test_bank_trains_without_errors(trained_bank: tuple) -> None:
    """Model musí natrénovať bez chýb a výsledky musia obsahovať presnosť."""
    _, results = trained_bank
    assert "accuracy" in results


def test_bank_accuracy_above_chance(trained_bank: tuple) -> None:
    """Presnosť musí byť nad 0.55 (lepšia ako náhodné hádanie pre binárnu klasifikáciu)."""
    _, results = trained_bank
    assert results["accuracy"] > 0.55, f"Presnosť {results['accuracy']:.3f} je príliš nízka"


def test_bank_confusion_matrix_binary(trained_bank: tuple) -> None:
    """Matica zámen pre binárnu klasifikáciu musí byť 2×2."""
    _, results = trained_bank
    cm = results["confusion_matrix"]
    assert len(cm) == 2
    assert all(len(row) == 2 for row in cm)


def test_bank_save_load_predictions_match(trained_bank: tuple, tmp_path: Path) -> None:
    """Po uložení a načítaní modelu musia predikcie zostať rovnaké."""
    model, _ = trained_bank
    save_path = tmp_path / "bank_model.joblib"
    model.save(save_path)

    loaded = SVMClassifier.load(save_path)
    orig_result = model.predict(SAMPLE_INPUT)
    loaded_result = loaded.predict(SAMPLE_INPUT)

    assert orig_result["predicted_class"] == loaded_result["predicted_class"]


def test_bank_predict_class_valid(trained_bank: tuple) -> None:
    """Predikovaná trieda musí byť 'yes' alebo 'no'."""
    model, _ = trained_bank
    result = model.predict(SAMPLE_INPUT)
    assert result["predicted_class"] in {"yes", "no"}


def test_bank_predict_probabilities_sum_to_one(trained_bank: tuple) -> None:
    """Súčet pravdepodobností predikcie musí byť ~1.0."""
    model, _ = trained_bank
    result = model.predict(SAMPLE_INPUT)
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-5
