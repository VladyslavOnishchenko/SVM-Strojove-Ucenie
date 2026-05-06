"""Pydantic schémy pre trénovanie SVM modelu — požiadavky a výsledky."""
from pydantic import BaseModel, Field

from backend.app.ml.types import ColumnType


class HyperparametersRequest(BaseModel):
    """Hyperparametre SVM klasifikátora posielané pri trénovaní."""
    kernel: str = Field(
        default="rbf",
        description="Typ kernelové funkcie: linear, rbf, poly alebo sigmoid",
    )
    C: float = Field(
        default=1.0,
        gt=0,
        description="Regularizačný parameter C (kladné číslo — väčšie C = tvrdšia hranica)",
    )
    gamma: str | float = Field(
        default="scale",
        description="Parameter gamma pre nelineárne kernely (scale, auto alebo kladné číslo)",
    )
    auto_tune: bool = Field(
        default=False,
        description="Ak True, použije GridSearchCV na automatický výber najlepších hyperparametrov",
    )


class TrainingRequest(BaseModel):
    """Kompletná požiadavka na trénovanie SVM klasifikátora."""
    column_schema: dict[str, ColumnType] = Field(
        ...,
        description="Mapovanie meno_stĺpca -> typ stĺpca (numeric, categorical, binary, target, ignore)",
    )
    hyperparameters: HyperparametersRequest = Field(
        default_factory=HyperparametersRequest,
        description="Hyperparametre SVM modelu",
    )
    test_size: float = Field(
        default=0.2,
        ge=0.1,
        le=0.5,
        description="Podiel dát vyhradených pre testovanie (0.1 až 0.5)",
    )
    cv_folds: int = Field(
        default=5,
        ge=2,
        le=10,
        description="Počet fold-ov pre krížovú validáciu (2 až 10)",
    )


class ClassMetrics(BaseModel):
    """Metriky klasifikácie pre jednu triedu alebo priemer."""
    precision: float = Field(..., description="Presnosť (pomer správne pozitívnych k všetkým predikovaným pozitívnym)")
    recall: float = Field(..., description="Citlivosť (pomer správne pozitívnych k všetkým skutočným pozitívnym)")
    f1_score: float = Field(..., description="Harmonický priemer presnosti a citlivosti")
    support: int = Field(..., description="Počet vzoriek danej triedy v testovacej množine")


class TrainingResponse(BaseModel):
    """Výsledky trénovania SVM modelu na testovacej množine."""
    accuracy: float = Field(..., description="Celková presnosť na testovacej množine")
    cv_mean_accuracy: float = Field(..., description="Priemerné CV skóre na trénovacej množine")
    cv_std_accuracy: float = Field(..., description="Smerodajná odchýlka CV skóre")
    classes: list[str] = Field(..., description="Zoznam tried cieľovej premennej")
    per_class_metrics: dict[str, ClassMetrics] = Field(
        ..., description="Metriky pre každú triedu zvlášť"
    )
    macro_avg: ClassMetrics = Field(..., description="Makro priemer metrík (neváhovaný)")
    weighted_avg: ClassMetrics = Field(..., description="Váhovaný priemer metrík (podľa počtu vzoriek)")
    confusion_matrix: list[list[int]] = Field(..., description="Matica zámen")
    best_hyperparameters: dict | None = Field(
        default=None,
        description="Najlepšie nájdené hyperparametre po auto-tune (None ak auto_tune=False)",
    )
    training_time_seconds: float = Field(..., description="Čas trénovania v sekundách")
