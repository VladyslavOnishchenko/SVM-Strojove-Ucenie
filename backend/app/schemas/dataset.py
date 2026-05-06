"""Pydantic schémy súvisiace s datasetmi — informácie, schéma a nahrávanie."""
from typing import Literal

from pydantic import BaseModel, Field

from backend.app.ml.types import ColumnType


class ExampleDatasetInfo(BaseModel):
    """Metadáta o zabudovanom príkladovom datasete."""
    name: str = Field(..., description="Technický identifikátor datasetu (používa sa v URL)")
    display_name: str = Field(..., description="Zobrazovaný názov datasetu")
    description: str = Field(..., description="Stručný opis datasetu v slovenčine")
    n_rows: int = Field(..., description="Počet riadkov v datasete")
    n_columns: int = Field(..., description="Celkový počet stĺpcov vrátane cieľového")
    n_classes: int = Field(..., description="Počet tried cieľovej premennej")
    task_type: Literal["binary", "multiclass"] = Field(
        ..., description="Typ klasifikačnej úlohy"
    )
    has_categorical: bool = Field(
        ..., description="Indikátor, či dataset obsahuje kategorické alebo binárne stĺpce"
    )
    filename: str = Field(..., description="Názov súboru v adresári data/examples/")


class ColumnInfo(BaseModel):
    """Informácie o jednom stĺpci datasetu vrátane navrhovaného typu."""
    name: str = Field(..., description="Meno stĺpca")
    dtype: str = Field(..., description="Dátový typ stĺpca v pandas (napr. float64, object)")
    suggested_type: ColumnType = Field(..., description="Automaticky navrhnutý typ stĺpca")
    unique_count: int = Field(..., description="Počet unikátnych hodnôt v stĺpci")
    sample_values: list[str] = Field(
        ..., description="Vzorka až 5 unikátnych hodnôt pre náhľad"
    )


class DatasetSchemaResponse(BaseModel):
    """Schéma aktuálne načítaného datasetu."""
    dataset_name: str = Field(..., description="Identifikátor datasetu")
    n_rows: int = Field(..., description="Počet riadkov")
    n_columns: int = Field(..., description="Počet stĺpcov")
    columns: list[ColumnInfo] = Field(..., description="Zoznam informácií o každom stĺpci")


class UploadResponse(BaseModel):
    """Odpoveď po úspešnom nahraní alebo načítaní datasetu."""
    message: str = Field(..., description="Potvrdzujúca správa")
    dataset_name: str = Field(..., description="Identifikátor načítaného datasetu")
    n_rows: int = Field(..., description="Počet riadkov v datasete")
    n_columns: int = Field(..., description="Počet stĺpcov v datasete")
