"""Pydantic schémy pre predikciu pomocou natrénovaného SVM modelu."""
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Vstupný vzor pre predikciu — hodnoty príznakových stĺpcov."""
    input_data: dict[str, str | float | int | bool] = Field(
        ...,
        description="Slovník mapujúci mená príznakových stĺpcov na ich hodnoty pre jeden vstup",
    )


class PredictionResponse(BaseModel):
    """Výsledok predikcie pre jeden vstupný vzor."""
    predicted_class: str = Field(..., description="Predikovaná trieda")
    probabilities: dict[str, float] = Field(
        ..., description="Pravdepodobnosti príslušnosti ku každej triede (súčet ~1.0)"
    )
