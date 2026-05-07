"""API endpoint pre trénovanie SVM klasifikátora na aktuálnom datasete."""
import time
from datetime import datetime

from fastapi import APIRouter, HTTPException

from backend.app.core.config import MODEL_FILENAME, STORAGE_DIR
from backend.app.core.state import app_state
from backend.app.ml.model import SVMClassifier
from backend.app.schemas.training import ClassMetrics, TrainingRequest, TrainingResponse

router = APIRouter()


@router.post("/", response_model=TrainingResponse)
async def train_model(request: TrainingRequest) -> TrainingResponse:
    """Natrénuje SVM na aktuálnom datasete a uloží model; pri auto_tune=True použije GridSearchCV."""
    if app_state.current_dataset is None:
        raise HTTPException(
            status_code=400,
            detail="Nebol nacitany ziadny dataset. Najprv nacitaj dataset cez /api/datasets.",
        )

    hyperparams = request.hyperparameters.model_dump()
    column_schema = dict(request.column_schema)

    start_time = time.perf_counter()
    try:
        model = SVMClassifier(column_schema, hyperparams)
        results = model.fit(
            app_state.current_dataset,
            test_size=request.test_size,
            cv_folds=request.cv_folds,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    training_time = time.perf_counter() - start_time

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    model.save(STORAGE_DIR / MODEL_FILENAME)

    app_state.current_model = model
    app_state.model_trained_at = datetime.now()

    report = results["classification_report"]

    per_class_metrics: dict[str, ClassMetrics] = {}
    for cls in results["classes"]:
        if cls in report:
            entry = report[cls]
            per_class_metrics[cls] = ClassMetrics(
                precision=float(entry["precision"]),
                recall=float(entry["recall"]),
                f1_score=float(entry["f1-score"]),
                support=int(entry["support"]),
            )

    macro = report["macro avg"]
    weighted = report["weighted avg"]

    return TrainingResponse(
        accuracy=results["accuracy"],
        cv_mean_accuracy=results["cv_mean_accuracy"],
        cv_std_accuracy=results["cv_std_accuracy"],
        classes=results["classes"],
        per_class_metrics=per_class_metrics,
        macro_avg=ClassMetrics(
            precision=float(macro["precision"]),
            recall=float(macro["recall"]),
            f1_score=float(macro["f1-score"]),
            support=int(macro["support"]),
        ),
        weighted_avg=ClassMetrics(
            precision=float(weighted["precision"]),
            recall=float(weighted["recall"]),
            f1_score=float(weighted["f1-score"]),
            support=int(weighted["support"]),
        ),
        confusion_matrix=results["confusion_matrix"],
        best_hyperparameters=results["best_params"],
        training_time_seconds=training_time,
    )
