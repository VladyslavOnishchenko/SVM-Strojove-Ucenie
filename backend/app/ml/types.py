"""Definície typov a enumerácií pre ML modul."""
from enum import Enum
from typing import TypedDict


class ColumnType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    TARGET = "target"
    IGNORE = "ignore"


class KernelType(str, Enum):
    LINEAR = "linear"
    RBF = "rbf"
    POLY = "poly"
    SIGMOID = "sigmoid"


# Dynamické mapovania (ColumnType závisí od mena stĺpca, TypedDict nepodporuje dynamické kľúče)
ColumnSchema = dict[str, ColumnType]
PredictionInput = dict[str, object]


class Hyperparameters(TypedDict):
    """Hyperparametre SVM klasifikátora."""
    kernel: str
    C: float
    gamma: str | float
    auto_tune: bool


class TrainingResults(TypedDict):
    """Výsledky trénovania modelu."""
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    confusion_matrix: list[list[int]]
    classes: list[str]
    cv_mean_accuracy: float
    cv_std_accuracy: float
    cv_scores: list[float]
    classification_report: dict
    best_params: dict | None


class PredictionResult(TypedDict):
    """Výsledok predikcie pre jeden vzor."""
    predicted_class: str
    probabilities: dict[str, float]
