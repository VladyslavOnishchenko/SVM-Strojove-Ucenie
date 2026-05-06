"""Príprava dát pre 2D vizualizáciu rozhodovacích hraníc SVM pomocou PCA."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from backend.app.ml.types import ColumnType

if TYPE_CHECKING:
    from backend.app.ml.model import SVMClassifier


def compute_decision_boundary_data(
    model: "SVMClassifier",
    df: pd.DataFrame,
    grid_resolution: int = 100,
) -> dict:
    """Vypočíta dáta pre vizualizáciu rozhodovacích hraníc v 2D priestore (PCA).

    Aplikuje preprocessor modelu, zredukuje dimenzie pomocou PCA na 2,
    vytvorí mriežku v 2D priestore a predikuje triedu pre každý bod mriežky
    spätnou transformáciou PCA do priestoru príznakov.

    Args:
        model: Natrénovaný SVMClassifier s pipeline a label_encoder.
        df: DataFrame s dátami (musí obsahovať príznakové aj cieľové stĺpce).
        grid_resolution: Počet bodov na každej osi mriežky.

    Returns:
        Slovník s bodmi datasetu v 2D, predikciami mriežky, triedami
        a podielom vysvetleného rozptylu PCA.
    """
    if model.pipeline is None or model.label_encoder is None:
        raise RuntimeError("Model musí byť natrénovaný pred vizualizáciou.")

    # Náhodne obmedzíme na max 1000 bodov pre výkon
    if len(df) > 1000:
        df = df.sample(n=1000, random_state=42)

    feature_cols = [
        c for c, t in model.column_schema.items()
        if t not in (ColumnType.TARGET, ColumnType.IGNORE)
    ]
    target_col = next(
        c for c, t in model.column_schema.items() if t == ColumnType.TARGET
    )

    X = df[feature_cols].copy()
    y_raw = df[target_col].astype(str)
    y_encoded: np.ndarray = model.label_encoder.transform(y_raw)

    preprocessor = model.pipeline.named_steps["preprocessor"]
    svm = model.pipeline.named_steps["svm"]

    X_transformed: np.ndarray = preprocessor.transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_2d: np.ndarray = pca.fit_transform(X_transformed)

    x_min, x_max = float(X_2d[:, 0].min()), float(X_2d[:, 0].max())
    y_min, y_max = float(X_2d[:, 1].min()), float(X_2d[:, 1].max())
    pad_x = (x_max - x_min) * 0.1
    pad_y = (y_max - y_min) * 0.1

    xx, yy = np.meshgrid(
        np.linspace(x_min - pad_x, x_max + pad_x, grid_resolution),
        np.linspace(y_min - pad_y, y_max + pad_y, grid_resolution),
    )
    grid_2d = np.c_[xx.ravel(), yy.ravel()]

    # Spätná PCA transformácia (aproximácia) → predikcia SVM priamo v priestore príznakov
    grid_transformed: np.ndarray = pca.inverse_transform(grid_2d)
    grid_preds: np.ndarray = svm.predict(grid_transformed)

    points = [
        {
            "x": float(X_2d[i, 0]),
            "y": float(X_2d[i, 1]),
            "class": str(model.label_encoder.inverse_transform([int(y_encoded[i])])[0]),
        }
        for i in range(len(X_2d))
    ]

    return {
        "points": points,
        "grid": {
            "x_range": [x_min - pad_x, x_max + pad_x],
            "y_range": [y_min - pad_y, y_max + pad_y],
            "resolution": grid_resolution,
            "predictions": grid_preds.reshape(grid_resolution, grid_resolution).tolist(),
        },
        "classes": list(model.label_encoder.classes_),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }
