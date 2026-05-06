"""Testy predspracovania dát — validate_schema, build_preprocessor, split_features_target."""
import pandas as pd
import pytest

from backend.app.ml.preprocessing import build_preprocessor, split_features_target, validate_schema
from backend.app.ml.types import ColumnType


# ---------------------------------------------------------------------------
# validate_schema
# ---------------------------------------------------------------------------

def test_validate_schema_no_target_raises():
    """Chýbajúci TARGET stĺpec musí vyvolať ValueError."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    schema = {"a": ColumnType.NUMERIC, "b": ColumnType.NUMERIC}
    with pytest.raises(ValueError, match="TARGET"):
        validate_schema(df, schema)


def test_validate_schema_multiple_targets_raises():
    """Viac ako jeden TARGET stĺpec musí vyvolať ValueError."""
    df = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
    schema = {"a": ColumnType.TARGET, "b": ColumnType.TARGET}
    with pytest.raises(ValueError):
        validate_schema(df, schema)


def test_validate_schema_missing_column_raises():
    """Odkaz na neexistujúci stĺpec musí vyvolať ValueError."""
    df = pd.DataFrame({"a": [1, 2]})
    schema = {"a": ColumnType.NUMERIC, "nonexistent": ColumnType.TARGET}
    with pytest.raises(ValueError):
        validate_schema(df, schema)


def test_validate_schema_all_ignore_raises():
    """Všetky príznakové stĺpce ako IGNORE musia vyvolať ValueError."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "label": ["x", "y"]})
    schema = {
        "a": ColumnType.IGNORE,
        "b": ColumnType.IGNORE,
        "label": ColumnType.TARGET,
    }
    with pytest.raises(ValueError, match="IGNORE"):
        validate_schema(df, schema)


def test_validate_schema_valid_passes():
    """Platná schéma nesmie vyvolať žiadnu výnimku."""
    df = pd.DataFrame({"feat": [1.0, 2.0], "label": ["a", "b"]})
    schema = {"feat": ColumnType.NUMERIC, "label": ColumnType.TARGET}
    validate_schema(df, schema)  # no exception


# ---------------------------------------------------------------------------
# build_preprocessor
# ---------------------------------------------------------------------------

def test_build_preprocessor_mixed_schema_transformer_count():
    """Zmiešaná schéma musí vytvoriť presne 3 transformátory (numeric, categorical, binary)."""
    schema = {
        "num1": ColumnType.NUMERIC,
        "num2": ColumnType.NUMERIC,
        "cat1": ColumnType.CATEGORICAL,
        "bin1": ColumnType.BINARY,
        "label": ColumnType.TARGET,
    }
    preprocessor = build_preprocessor(schema)
    assert len(preprocessor.transformers) == 3


def test_build_preprocessor_numeric_only():
    """Len NUMERIC stĺpce musia vytvoriť presne 1 transformátor."""
    schema = {
        "x": ColumnType.NUMERIC,
        "y": ColumnType.NUMERIC,
        "label": ColumnType.TARGET,
    }
    preprocessor = build_preprocessor(schema)
    assert len(preprocessor.transformers) == 1
    assert preprocessor.transformers[0][0] == "numeric"


def test_build_preprocessor_transforms_data():
    """Preprocessor musí správne transformovať dáta bez chýb."""
    schema = {
        "num": ColumnType.NUMERIC,
        "cat": ColumnType.CATEGORICAL,
        "label": ColumnType.TARGET,
    }
    df = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0],
        "cat": ["a", "b", "a", "b"],
        "label": ["x", "y", "x", "y"],
    })
    preprocessor = build_preprocessor(schema)
    X = df[["num", "cat"]]
    result = preprocessor.fit_transform(X)
    assert result.shape[0] == 4


# ---------------------------------------------------------------------------
# split_features_target
# ---------------------------------------------------------------------------

def test_split_features_target_shapes():
    """X a y musia mať správne tvary po rozdelení."""
    df = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0],
        "f2": [4.0, 5.0, 6.0],
        "label": ["a", "b", "a"],
    })
    schema = {
        "f1": ColumnType.NUMERIC,
        "f2": ColumnType.NUMERIC,
        "label": ColumnType.TARGET,
    }
    X, y, le = split_features_target(df, schema)
    assert X.shape == (3, 2)
    assert len(y) == 3


def test_split_features_target_label_encoder_classes():
    """LabelEncoder musí správne zakódovať triedy."""
    df = pd.DataFrame({
        "x": [1, 2, 3],
        "target": ["cat", "dog", "cat"],
    })
    schema = {"x": ColumnType.NUMERIC, "target": ColumnType.TARGET}
    X, y, le = split_features_target(df, schema)
    assert list(le.classes_) == ["cat", "dog"]
    assert list(y) == [0, 1, 0]


def test_split_features_target_excludes_ignore():
    """IGNORE stĺpce nesmú byť zahrnuté v X."""
    df = pd.DataFrame({
        "feat": [1.0, 2.0],
        "ignored": [99.0, 88.0],
        "label": ["a", "b"],
    })
    schema = {
        "feat": ColumnType.NUMERIC,
        "ignored": ColumnType.IGNORE,
        "label": ColumnType.TARGET,
    }
    X, y, le = split_features_target(df, schema)
    assert "ignored" not in X.columns
    assert X.shape[1] == 1
