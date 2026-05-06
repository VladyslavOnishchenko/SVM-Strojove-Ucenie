"""Generátor vzorových datových sad pre SVM Strojove Ucenie.

Spustenie z korena projektu:
    python scripts/generate_sample_datasets.py

Vygeneruje:
    data/examples/iris.csv                 -- Iris dataset (150 riadkov)
    data/examples/wine.csv                 -- Wine dataset (178 riadkov)
    data/examples/bank_marketing_sample.csv -- Synteticky bankovy dataset (500 riadkov)
    data/examples/heart_disease.csv        -- Synteticky heart disease dataset (300 riadkov)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "examples"


def generate_iris(output_path: Path) -> None:
    """Vygeneruje Iris dataset zo sklearn a ulozi ako CSV.

    Args:
        output_path: Cielova cesta pre subor iris.csv.
    """
    iris = load_iris()
    df = pd.DataFrame(
        iris.data,
        columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    )
    df["species"] = [iris.target_names[i] for i in iris.target]
    df.to_csv(output_path, index=False)


def generate_wine(output_path: Path) -> None:
    """Vygeneruje Wine dataset zo sklearn s ludsky citatelnym nazvom triedy.

    Triedy: 0 -> Barolo, 1 -> Grignolino, 2 -> Barbera (skutocne talianske odrudy).

    Args:
        output_path: Cielova cesta pre subor wine.csv.
    """
    wine = load_wine(as_frame=True)
    df = wine.data.copy()
    # Premenuj stlpec so lomkou, ktora je neplatna v niektorych prostrediach
    df.columns = [c.replace("/", "_per_") for c in df.columns]
    target_map = {0: "Barolo", 1: "Grignolino", 2: "Barbera"}
    df["wine_class"] = [target_map[i] for i in wine.target]
    df.to_csv(output_path, index=False)


def generate_bank_marketing(output_path: Path, n: int = 500, seed: int = 42) -> None:
    """Vygeneruje synteticky bankovy marketing dataset s reprodukovatelnym seedom.

    Cielova premenna 'subscribed' koreluje s duration a balance (logisticka funkcia).

    Args:
        output_path: Cielova cesta pre subor bank_marketing_sample.csv.
        n: Pocet riadkov.
        seed: Nahodny seed pre reprodukovatelnost.
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


def generate_heart_disease(output_path: Path, n: int = 300, seed: int = 42) -> None:
    """Vygeneruje synteticky Cleveland-style heart disease dataset.

    Cielova premenna 'target' koreluje s vekom, cholesterolom, exercise_angina
    a typom bolesti na hrudi (logisticka funkcia). Cielovy podiel choroby: ~50 %.

    Args:
        output_path: Cielova cesta pre subor heart_disease.csv.
        n: Pocet riadkov.
        seed: Nahodny seed pre reprodukovatelnost.
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(29, 78, n)
    sex = rng.choice(["male", "female"], n, p=[0.68, 0.32])
    chest_pain_type = rng.choice(
        ["typical_angina", "atypical_angina", "non_anginal", "asymptomatic"],
        n,
        p=[0.15, 0.25, 0.30, 0.30],
    )
    resting_bp = rng.integers(94, 201, n)
    cholesterol = rng.integers(126, 565, n)
    fasting_blood_sugar = rng.choice(["yes", "no"], n, p=[0.15, 0.85])
    resting_ecg = rng.choice(
        ["normal", "st_t_abnormality", "lv_hypertrophy"],
        n,
        p=[0.50, 0.30, 0.20],
    )
    max_heart_rate = rng.integers(71, 203, n)
    exercise_angina = rng.choice(["yes", "no"], n, p=[0.33, 0.67])
    oldpeak = np.round(rng.uniform(0.0, 6.2, n), 1)
    st_slope = rng.choice(["upsloping", "flat", "downsloping"], n, p=[0.35, 0.45, 0.20])

    logit = (
        0.05 * (age.astype(float) - 53.0)
        + 0.005 * (cholesterol.astype(float) - 200.0)
        + (-0.02) * (max_heart_rate.astype(float) - 100.0)
        + np.where(chest_pain_type == "asymptomatic", 1.0, 0.0)
        + np.where(exercise_angina == "yes", 0.8, 0.0)
        + np.where(sex == "male", 0.4, 0.0)
        - 0.8  # intercept kalibrovaný na ~50 % choroby
    )
    prob_disease = 1.0 / (1.0 + np.exp(-logit))
    target = np.where(rng.random(n) < prob_disease, "disease", "healthy")

    df = pd.DataFrame({
        "age": age,
        "sex": sex,
        "chest_pain_type": chest_pain_type,
        "resting_blood_pressure": resting_bp,
        "cholesterol": cholesterol,
        "fasting_blood_sugar": fasting_blood_sugar,
        "resting_ecg": resting_ecg,
        "max_heart_rate": max_heart_rate,
        "exercise_angina": exercise_angina,
        "oldpeak": oldpeak,
        "st_slope": st_slope,
        "target": target,
    })
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    iris_path = OUTPUT_DIR / "iris.csv"
    generate_iris(iris_path)
    print(f"Generated: {iris_path} ({len(pd.read_csv(iris_path))} rows)")

    wine_path = OUTPUT_DIR / "wine.csv"
    generate_wine(wine_path)
    df_wine = pd.read_csv(wine_path)
    print(f"Generated: {wine_path} ({len(df_wine)} rows, {df_wine['wine_class'].nunique()} classes)")

    bank_path = OUTPUT_DIR / "bank_marketing_sample.csv"
    generate_bank_marketing(bank_path)
    df_bank = pd.read_csv(bank_path)
    yes_rate = (df_bank["subscribed"] == "yes").mean()
    print(f"Generated: {bank_path} ({len(df_bank)} rows, {yes_rate:.1%} subscribed=yes)")

    hd_path = OUTPUT_DIR / "heart_disease.csv"
    generate_heart_disease(hd_path)
    df_hd = pd.read_csv(hd_path)
    disease_rate = (df_hd["target"] == "disease").mean()
    print(f"Generated: {hd_path} ({len(df_hd)} rows, {disease_rate:.1%} disease)")
