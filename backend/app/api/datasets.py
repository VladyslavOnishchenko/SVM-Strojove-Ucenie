"""API endpointy pre správu datasetov — zoznam príkladov, nahrávanie a schéma."""
import io
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.app.core.state import app_state
from backend.app.ml.column_detection import suggest_column_types
from backend.app.schemas.dataset import (
    ColumnInfo,
    DatasetSchemaResponse,
    ExampleDatasetInfo,
    UploadResponse,
)

router = APIRouter()

_EXAMPLES_DIR = Path(__file__).parent.parent.parent.parent / "data" / "examples"

_EXAMPLE_DATASETS: dict[str, ExampleDatasetInfo] = {
    "iris": ExampleDatasetInfo(
        name="iris",
        display_name="Iris",
        description="Klasický dataset s 3 druhmi kosatcov podľa 4 číselných morfologických znakov.",
        n_rows=150,
        n_columns=5,
        n_classes=3,
        task_type="multiclass",
        has_categorical=False,
        filename="iris.csv",
    ),
    "wine": ExampleDatasetInfo(
        name="wine",
        display_name="Wine",
        description="Chemická analýza talianskych vín zo 3 pestovatelských oblastí; 13 číselných znakov.",
        n_rows=178,
        n_columns=14,
        n_classes=3,
        task_type="multiclass",
        has_categorical=False,
        filename="wine.csv",
    ),
    "bank_marketing": ExampleDatasetInfo(
        name="bank_marketing",
        display_name="Bank Marketing",
        description="Syntetické bankové dáta s binárnym cieľom — prihlásenie klienta na termínovaný vklad.",
        n_rows=500,
        n_columns=10,
        n_classes=2,
        task_type="binary",
        has_categorical=True,
        filename="bank_marketing_sample.csv",
    ),
    "heart_disease": ExampleDatasetInfo(
        name="heart_disease",
        display_name="Heart Disease",
        description="Syntetický Cleveland-style dataset srdcových ochorení so zmesou numerických a kategorických znakov.",
        n_rows=300,
        n_columns=12,
        n_classes=2,
        task_type="binary",
        has_categorical=True,
        filename="heart_disease.csv",
    ),
}


@router.get("/examples", response_model=list[ExampleDatasetInfo])
async def list_example_datasets() -> list[ExampleDatasetInfo]:
    """Vráti zoznam metadát o všetkých 4 zabudovaných príkladových datasetoch."""
    return list(_EXAMPLE_DATASETS.values())


@router.post("/examples/{name}/load", response_model=UploadResponse)
async def load_example_dataset(name: str) -> UploadResponse:
    """Načíta pomenovaný príkladový dataset do pamäti aplikácie.

    Args:
        name: Technický identifikátor datasetu (iris, wine, bank_marketing, heart_disease).

    Returns:
        UploadResponse s počtom riadkov a stĺpcov.

    Raises:
        HTTPException 404: Ak názov nepatrí žiadnemu zabudovanému datasetu.
    """
    if name not in _EXAMPLE_DATASETS:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{name}' neexistuje. Dostupne: {list(_EXAMPLE_DATASETS.keys())}",
        )

    info = _EXAMPLE_DATASETS[name]
    csv_path = _EXAMPLES_DIR / info.filename
    df = pd.read_csv(csv_path)

    app_state.current_dataset = df
    app_state.current_dataset_name = name
    app_state.current_model = None
    app_state.model_trained_at = None

    return UploadResponse(
        message=f"Dataset '{name}' uspesne nacitany ({len(df)} riadkov, {len(df.columns)} stlpcov).",
        dataset_name=name,
        n_rows=len(df),
        n_columns=len(df.columns),
    )


@router.post("/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)) -> UploadResponse:
    """Nahrá vlastný CSV súbor a uloží ho ako aktuálny dataset.

    Validácia: súbor musí byť parsovateľný ako CSV, obsahovať aspoň 10 riadkov
    a aspoň 2 stĺpce.

    Args:
        file: Nahrávaný súbor cez multipart/form-data.

    Returns:
        UploadResponse s metadátami nahrátého datasetu.

    Raises:
        HTTPException 400: Neplatný formát, príliš málo riadkov alebo stĺpcov.
    """
    content = await file.read()

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Subor nie je platny CSV format alebo obsahuje necitatelny obsah.",
        )

    if len(df) < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset musi mat aspon 10 riadkov, nahratych len {len(df)}.",
        )
    if len(df.columns) < 2:
        raise HTTPException(
            status_code=400,
            detail="Dataset musi mat aspon 2 stlpce.",
        )

    raw_name = file.filename or "upload"
    dataset_name = raw_name.removesuffix(".csv")

    app_state.current_dataset = df
    app_state.current_dataset_name = dataset_name
    app_state.current_model = None
    app_state.model_trained_at = None

    return UploadResponse(
        message=f"Dataset '{dataset_name}' uspesne nahratý ({len(df)} riadkov, {len(df.columns)} stlpcov).",
        dataset_name=dataset_name,
        n_rows=len(df),
        n_columns=len(df.columns),
    )


@router.get("/current/schema", response_model=DatasetSchemaResponse)
async def get_current_schema() -> DatasetSchemaResponse:
    """Vráti schéma aktuálne načítaného datasetu s navrhovanými typmi stĺpcov.

    Returns:
        DatasetSchemaResponse so zoznamom stĺpcov a ich navrhovanými typmi.

    Raises:
        HTTPException 400: Ak nebol načítaný žiadny dataset.
    """
    if app_state.current_dataset is None:
        raise HTTPException(
            status_code=400,
            detail="Nebol nacitany ziadny dataset. Najprv nacitaj dataset cez /examples alebo /upload.",
        )

    df = app_state.current_dataset
    suggested = suggest_column_types(df)

    columns: list[ColumnInfo] = []
    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        sample = [str(v) for v in unique_vals[:5]]
        columns.append(
            ColumnInfo(
                name=col,
                dtype=str(df[col].dtype),
                suggested_type=suggested[col],
                unique_count=int(df[col].nunique()),
                sample_values=sample,
            )
        )

    return DatasetSchemaResponse(
        dataset_name=app_state.current_dataset_name or "unknown",
        n_rows=len(df),
        n_columns=len(df.columns),
        columns=columns,
    )
