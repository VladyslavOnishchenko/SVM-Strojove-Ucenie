"""Generátor vzorových dátových sád pre SVM Strojové Učenie.

Spustenie z koreňa projektu:
    python scripts/generate_sample_datasets.py

Vygeneruje:
    data/examples/iris.csv            — Iris dataset (150 riadkov, 4 číselné príznaky, 3 triedy)
    data/examples/bank_marketing_sample.csv — Syntetický bankový dataset (500 riadkov, zmiešané typy)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "examples"


def generate_iris(output_path: Path) -> None:
    """Vygeneruje Iris dataset zo sklearn a uloží ako CSV.

    Args:
        output_path: Cieľová cesta pre súbor iris.csv.
    """
    iris = load_iris()
    df = pd.DataFrame(
        iris.data,
        columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    )
    df["species"] = [iris.target_names[i] for i in iris.target]
    df.to_csv(output_path, index=False)


def generate_bank_marketing(output_path: Path, n: int = 500, seed: int = 42) -> None:
    """Vygeneruje syntetický bankový marketing dataset s reprodukovateľným seedom.

    Cieľová premenná 'subscribed' koreluje s duration a balance (logistická funkcia).

    Args:
        output_path: Cieľová cesta pre súbor bank_marketing_sample.csv.
        n: Počet riadkov.
        seed: Náhodný seed pre reprodukovateľnosť.
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 96, n)
    job = rng.choice(
        ["admin", "blue-collar", "technician", "services", "management",
         "retired", "student", "unemployed"],
        n,
    )
    marital = rng.choice(["single", "married", "divorced"], n, p=[0.30, 0.55, 0.15])
    education = rng.choice(
        ["primary", "secondary", "tertiary", "unknown"],
        n,
        p=[0.15, 0.45, 0.35, 0.05],
    )
    default = rng.choice(["yes", "no"], n, p=[0.02, 0.98])
    balance = rng.integers(-500, 5001, n)
    housing = rng.choice(["yes", "no"], n, p=[0.55, 0.45])
    loan = rng.choice(["yes", "no"], n, p=[0.15, 0.85])
    duration = rng.integers(0, 3001, n)

    # Cieľová premenná: vyššie duration a balance → väčšia pravdepodobnosť subscribed=yes
    logit = duration / 2000.0 + balance / 5000.0 - 2.0
    prob_subscribe = 1.0 / (1.0 + np.exp(-logit))
    subscribed = np.where(rng.random(n) < prob_subscribe, "yes", "no")

    df = pd.DataFrame({
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "duration": duration,
        "subscribed": subscribed,
    })
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    iris_path = OUTPUT_DIR / "iris.csv"
    generate_iris(iris_path)
    print(f"Generated: {iris_path} ({len(pd.read_csv(iris_path))} rows)")

    bank_path = OUTPUT_DIR / "bank_marketing_sample.csv"
    generate_bank_marketing(bank_path)
    df_bank = pd.read_csv(bank_path)
    yes_rate = (df_bank["subscribed"] == "yes").mean()
    print(f"Generated: {bank_path} ({len(df_bank)} rows, {yes_rate:.1%} subscribed=yes)")
