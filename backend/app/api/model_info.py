"""API endpointy pre informácie o modeli — stav, vizualizácia a stiahnutie."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.app.core.config import MODEL_FILENAME, STORAGE_DIR
from backend.app.core.state import app_state
from backend.app.ml.visualization import compute_decision_boundary_data
from backend.app.schemas.model_info import ModelStatusResponse, VisualizationResponse

router = APIRouter()


@router.get("/status", response_model=ModelStatusResponse)
async def get_model_status() -> ModelStatusResponse:
    """Vráti aktuálny stav modelu — is_trained, triedy a čas trénovania."""
    if app_state.current_model is None:
        return ModelStatusResponse(is_trained=False)

    return ModelStatusResponse(
        is_trained=True,
        dataset_name=app_state.current_dataset_name,
        classes=list(app_state.current_model.label_encoder.classes_),
        trained_at=app_state.model_trained_at.isoformat() if app_state.model_trained_at else None,
    )


@router.get("/visualization", response_model=VisualizationResponse)
async def get_visualization() -> VisualizationResponse:
    """Vypočíta PCA vizualizáciu rozhodovacích hraníc; výpočtovo náročné."""
    if app_state.current_model is None:
        raise HTTPException(status_code=400, detail="Model este nie je nauceny.")
    if app_state.current_dataset is None:
        raise HTTPException(status_code=400, detail="Dataset nie je nacitany.")

    data = compute_decision_boundary_data(app_state.current_model, app_state.current_dataset)
    return VisualizationResponse(**data)


@router.get("/download")
async def download_model() -> FileResponse:
    """Stiahne natrénovaný model ako .joblib; 404 ak súbor neexistuje."""
    model_path = STORAGE_DIR / MODEL_FILENAME
    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Subor modelu neexistuje. Najprv natrenuj model cez /api/train.",
        )
    return FileResponse(
        path=model_path,
        media_type="application/octet-stream",
        filename="svm_model.joblib",
    )
