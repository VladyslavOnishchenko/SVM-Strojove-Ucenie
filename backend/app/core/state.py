"""Globálny stav aplikácie — aktuálny dataset a natrénovaný model."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

from backend.app.ml.model import SVMClassifier


@dataclass
class AppState:
    """Jednoduchý in-memory stav aplikácie zdieľaný naprieč endpointmi."""
    current_dataset: Optional[pd.DataFrame] = None
    current_dataset_name: Optional[str] = None
    current_model: Optional[SVMClassifier] = None
    model_trained_at: Optional[datetime] = None


app_state = AppState()
