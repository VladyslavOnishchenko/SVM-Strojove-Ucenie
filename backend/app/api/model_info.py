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
    """Vráti aktuálny stav natrénovaného modelu.

    Returns:
        ModelStatusResponse s informáciou, či je model natrénovaný, a ak áno, s jeho metadátami.
    """
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
    """Vypočíta dáta pre 2D PCA vizualizáciu rozhodovacích hraníc SVM.

    Táto operácia je výpočtovo náročnejšia — zahŕňa transformáciu dát, PCA
    a predikcie na mriežke bodov. Pre väčšie datasety to môže trvať niekoľko sekúnd.

    Returns:
        VisualizationResponse s bodmi datasetu, mriežkou predikcií a PCA metadátami.

    Raises:
        HTTPException 400: Ak model alebo dataset nie je načítaný.
    """
    if app_state.current_model is None:
        raise HTTPException(status_code=400, detail="Model este nie je nauceny.")
    if app_state.current_dataset is None:
        raise HTTPException(status_code=400, detail="Dataset nie je nacitany.")

    data = compute_decision_boundary_data(app_state.current_model, app_state.current_dataset)
    return VisualizationResponse(**data)


@router.get("/download")
async def download_model() -> FileResponse:
    """Stiahne natrénovaný model ako súbor .joblib.

    Returns:
        FileResponse so súborom backend/storage/current_model.joblib.

    Raises:
        HTTPException 404: Ak súbor modelu neexistuje (model nebol natrénovaný alebo uložený).
    """
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
