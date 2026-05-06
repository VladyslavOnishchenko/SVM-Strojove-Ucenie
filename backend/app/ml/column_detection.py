"""Heuristická detekcia typov stĺpcov z pandas DataFrame."""
import pandas as pd

from backend.app.ml.types import ColumnType


def suggest_column_types(df: pd.DataFrame) -> dict[str, ColumnType]:
    """Navrhne typ každého stĺpca na základe jednoduchej heuristiky.

    Pravidlá (v poradí priority):
    - Posledný stĺpec sa navrhuje ako TARGET.
    - Stĺpce s presne 2 unikátnymi hodnotami (numerické aj reťazcové) sa klasifikujú ako BINARY.
    - Ostatné číselné stĺpce sa klasifikujú ako NUMERIC.
    - Ostatné nečíselné stĺpce sa klasifikujú ako CATEGORICAL.

    Args:
        df: Vstupný DataFrame.

    Returns:
        Slovník mapujúci meno stĺpca na navrhovaný ColumnType.
    """
    columns = list(df.columns)
    schema: dict[str, ColumnType] = {}

    for i, col in enumerate(columns):
        if i == len(columns) - 1:
            schema[col] = ColumnType.TARGET
            continue

        n_unique = df[col].nunique(dropna=True)

        if n_unique == 2:
            schema[col] = ColumnType.BINARY
        elif pd.api.types.is_numeric_dtype(df[col]):
            schema[col] = ColumnType.NUMERIC
        else:
            schema[col] = ColumnType.CATEGORICAL

    return schema
