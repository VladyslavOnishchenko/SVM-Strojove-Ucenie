"""API endpoint pre predikciu triedy pomocou natrénovaného SVM modelu."""
from fastapi import APIRouter, HTTPException

from backend.app.core.state import app_state
from backend.app.ml.types import ColumnType
from backend.app.schemas.prediction import PredictionRequest, PredictionResponse

router = APIRouter()


@router.post("/", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Predikuje triedu a pravdepodobnosti pre jeden vstupný vzor.

    Overí, či sú v požiadavke prítomné všetky požadované príznakové stĺpce.

    Args:
        request: PredictionRequest so slovníkom hodnôt príznakov.

    Returns:
        PredictionResponse s predikovanou triedou a pravdepodobnosťami.

    Raises:
        HTTPException 400: Ak model ešte nebol natrénovaný.
        HTTPException 422: Ak chýbajú požadované stĺpce vo vstupe.
    """
    if app_state.current_model is None:
        raise HTTPException(
            status_code=400,
            detail="Model este nie je nauceny. Najprv pouzi /api/train.",
        )

    required_cols = [
        col
        for col, ctype in app_state.current_model.column_schema.items()
        if ctype not in (ColumnType.TARGET, ColumnType.IGNORE)
    ]
    missing = [col for col in required_cols if col not in request.input_data]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Chybajuce pozadovane stlpce: {missing}",
        )

    result = app_state.current_model.predict(dict(request.input_data))
    return PredictionResponse(
        predicted_class=result["predicted_class"],
        probabilities=result["probabilities"],
    )
