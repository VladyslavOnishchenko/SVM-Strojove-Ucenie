"""Trénovanie, predikcia, ukladanie a načítanie SVM klasifikátora."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from backend.app.ml.preprocessing import build_preprocessor, split_features_target, validate_schema
from backend.app.ml.types import ColumnType, TrainingResults, PredictionResult


class SVMClassifier:
    """SVM klasifikátor s plným predspracovaním a hodnotením."""

    def __init__(
        self,
        column_schema: dict[str, ColumnType],
        hyperparameters: dict,
    ) -> None:
        """Inicializuje klasifikátor so schémou stĺpcov a hyperparametrami."""
        self.column_schema = column_schema
        self.hyperparameters = hyperparameters
        self.pipeline: Pipeline | None = None
        self.label_encoder: LabelEncoder | None = None

    def fit(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        cv_folds: int = 5,
        random_state: int = 42,
    ) -> TrainingResults:
        """Natrénuje SVM, spustí CV a vráti TrainingResults; pri auto_tune=True použije GridSearchCV."""
        validate_schema(df, self.column_schema)
        X, y, self.label_encoder = split_features_target(df, self.column_schema)
        preprocessor = build_preprocessor(self.column_schema)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        base_svc = SVC(probability=True, random_state=random_state)
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("svm", base_svc),
        ])

        best_params: dict | None = None

        if self.hyperparameters.get("auto_tune", False):
            param_grid = [
                {
                    "svm__kernel": ["linear"],
                    "svm__C": [0.1, 1, 10],
                },
                {
                    "svm__kernel": ["rbf", "poly", "sigmoid"],
                    "svm__C": [0.1, 1, 10],
                    "svm__gamma": ["scale", "auto", 0.01, 0.1],
                },
            ]
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=cv_folds,
                scoring="accuracy",
                n_jobs=-1,
                refit=True,
            )
            grid_search.fit(X_train, y_train)
            self.pipeline = grid_search.best_estimator_
            best_params = dict(grid_search.best_params_)
        else:
            pipeline.set_params(
                svm__kernel=self.hyperparameters.get("kernel", "rbf"),
                svm__C=float(self.hyperparameters.get("C", 1.0)),
                svm__gamma=self.hyperparameters.get("gamma", "scale"),
            )
            pipeline.fit(X_train, y_train)
            self.pipeline = pipeline

        cv_scores_array: np.ndarray = cross_val_score(
            self.pipeline, X_train, y_train, cv=cv_folds, scoring="accuracy"
        )

        y_pred: np.ndarray = self.pipeline.predict(X_test)
        report: dict = classification_report(
            y_test,
            y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True,
            zero_division=0,
        )
        cm: list[list[int]] = confusion_matrix(y_test, y_pred).tolist()

        return TrainingResults(
            accuracy=float(report["accuracy"]),
            precision_macro=float(report["macro avg"]["precision"]),
            recall_macro=float(report["macro avg"]["recall"]),
            f1_macro=float(report["macro avg"]["f1-score"]),
            precision_weighted=float(report["weighted avg"]["precision"]),
            recall_weighted=float(report["weighted avg"]["recall"]),
            f1_weighted=float(report["weighted avg"]["f1-score"]),
            confusion_matrix=cm,
            classes=list(self.label_encoder.classes_),
            cv_mean_accuracy=float(cv_scores_array.mean()),
            cv_std_accuracy=float(cv_scores_array.std()),
            cv_scores=cv_scores_array.tolist(),
            classification_report=report,
            best_params=best_params,
        )

    def predict(self, input_data: dict) -> PredictionResult:
        """Vráti predikovanú triedu a pravdepodobnosti pre jeden vstupný riadok."""
        if self.pipeline is None or self.label_encoder is None:
            raise RuntimeError("Model musí byť natrénovaný pred predikciou. Zavolaj fit() najskôr.")

        row = pd.DataFrame([input_data])
        class_idx: int = int(self.pipeline.predict(row)[0])
        probas: np.ndarray = self.pipeline.predict_proba(row)[0]

        predicted_class: str = str(self.label_encoder.inverse_transform([class_idx])[0])
        probabilities: dict[str, float] = {
            str(cls): float(prob)
            for cls, prob in zip(self.label_encoder.classes_, probas)
        }

        return PredictionResult(
            predicted_class=predicted_class,
            probabilities=probabilities,
        )

    def save(self, path: Path) -> None:
        """Uloží pipeline, label encoder a schému na disk pomocou joblib."""
        if self.pipeline is None or self.label_encoder is None:
            raise RuntimeError("Model musí byť natrénovaný pred uložením.")

        joblib.dump(
            {
                "pipeline": self.pipeline,
                "label_encoder": self.label_encoder,
                "column_schema": self.column_schema,
                "hyperparameters": self.hyperparameters,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "SVMClassifier":
        """Načíta uložený model a vráti rekonštruovanú inštanciu SVMClassifier."""
        data: dict = joblib.load(path)
        instance = cls(data["column_schema"], data["hyperparameters"])
        instance.pipeline = data["pipeline"]
        instance.label_encoder = data["label_encoder"]
        return instance
