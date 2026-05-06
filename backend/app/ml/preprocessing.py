"""Predspracovanie dát pre SVM klasifikátor na základe schémy stĺpcov."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler

from backend.app.ml.types import ColumnType


def build_preprocessor(column_schema: dict[str, ColumnType]) -> ColumnTransformer:
    """Vytvorí sklearn ColumnTransformer podľa typov stĺpcov.

    Pre NUMERIC: SimpleImputer(median) + StandardScaler.
    Pre CATEGORICAL: SimpleImputer(most_frequent) + OneHotEncoder.
    Pre BINARY: SimpleImputer(most_frequent) + OrdinalEncoder.
    Stĺpce TARGET a IGNORE sú vynechané (remainder='drop').

    Args:
        column_schema: Mapovanie meno_stĺpca -> ColumnType.

    Returns:
        Nakonfigurovaný ColumnTransformer (ešte nie je natrénovaný).
    """
    numeric_cols = [c for c, t in column_schema.items() if t == ColumnType.NUMERIC]
    categorical_cols = [c for c, t in column_schema.items() if t == ColumnType.CATEGORICAL]
    binary_cols = [c for c, t in column_schema.items() if t == ColumnType.BINARY]

    transformers: list[tuple] = []

    if numeric_cols:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipeline, numeric_cols))

    if categorical_cols:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical_cols))

    if binary_cols:
        binary_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder()),
        ])
        transformers.append(("binary", binary_pipeline, binary_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def validate_schema(df: pd.DataFrame, column_schema: dict[str, ColumnType]) -> None:
    """Overí konzistentnosť schémy stĺpcov voči DataFrame.

    Args:
        df: Vstupný DataFrame.
        column_schema: Mapovanie meno_stĺpca -> ColumnType.

    Raises:
        ValueError: Ak schéma odkazuje na neexistujúce stĺpce, chýba TARGET,
                    je viac ako jeden TARGET, alebo všetky príznakové stĺpce sú IGNORE.
    """
    missing = [c for c in column_schema if c not in df.columns]
    if missing:
        raise ValueError(f"Stĺpce chýbajú v DataFrame: {missing}")

    target_cols = [c for c, t in column_schema.items() if t == ColumnType.TARGET]
    if len(target_cols) == 0:
        raise ValueError("Schéma musí obsahovať presne jeden TARGET stĺpec, nenašiel sa žiadny.")
    if len(target_cols) > 1:
        raise ValueError(f"Schéma obsahuje viac ako jeden TARGET stĺpec: {target_cols}")

    feature_cols = [
        c for c, t in column_schema.items()
        if t not in (ColumnType.TARGET, ColumnType.IGNORE)
    ]
    if len(feature_cols) == 0:
        raise ValueError("Všetky príznakové stĺpce sú označené ako IGNORE — nie sú žiadne príznaky.")


def split_features_target(
    df: pd.DataFrame,
    column_schema: dict[str, ColumnType],
) -> tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    """Rozdelí DataFrame na príznaky X, zakódovaný cieľ y a LabelEncoder.

    Args:
        df: Vstupný DataFrame.
        column_schema: Mapovanie meno_stĺpca -> ColumnType.

    Returns:
        Trojica (X, y_encoded, label_encoder) kde X obsahuje len príznakové stĺpce,
        y_encoded sú numericky zakódované triedy a label_encoder umožňuje spätné dekódovanie.
    """
    target_col = next(c for c, t in column_schema.items() if t == ColumnType.TARGET)
    feature_cols = [
        c for c, t in column_schema.items()
        if t not in (ColumnType.TARGET, ColumnType.IGNORE)
    ]

    X = df[feature_cols].copy()
    y_raw = df[target_col].astype(str)

    label_encoder = LabelEncoder()
    y_encoded = pd.Series(
        label_encoder.fit_transform(y_raw),
        name=target_col,
        index=df.index,
    )

    return X, y_encoded, label_encoder
