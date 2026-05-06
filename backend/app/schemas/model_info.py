"""Pydantic schémy pre stav modelu a vizualizáciu rozhodovacích hraníc."""
from typing import Any

from pydantic import BaseModel, Field


class ModelStatusResponse(BaseModel):
    """Aktuálny stav natrénovaného modelu v aplikácii."""
    is_trained: bool = Field(..., description="Indikátor, či je model natrénovaný a pripravený na predikcie")
    dataset_name: str | None = Field(
        default=None, description="Meno datasetu použitého pri trénovaní"
    )
    classes: list[str] | None = Field(
        default=None, description="Zoznam tried, na ktoré bol model natrénovaný"
    )
    trained_at: str | None = Field(
        default=None, description="Čas dokončenia trénovania vo formáte ISO 8601"
    )


class VisualizationResponse(BaseModel):
    """Dáta pre 2D vizualizáciu rozhodovacích hraníc SVM pomocou PCA.

    Vizualizácia sa vykresluje na frontende; tento endpoint len pripraví dáta.
    """
    points: list[dict[str, Any]] = Field(
        ...,
        description="Body datasetu v 2D PCA priestore (každý bod má polia x, y a class)",
    )
    grid: dict[str, Any] = Field(
        ...,
        description="Mriežka predikcií v 2D priestore (x_range, y_range, resolution, predictions)",
    )
    classes: list[str] = Field(..., description="Zoznam tried modelu")
    explained_variance_ratio: list[float] = Field(
        ..., description="Podiel vysvetleného rozptylu pre každú z 2 PCA komponentov"
    )
