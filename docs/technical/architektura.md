# Technická dokumentácia

## Prehľad architektúry

Aplikácia je rozdelená do troch vrstiev, ktoré komunikujú jednosmerne zhora nadol. Frontend poskytuje používateľské rozhranie — načítava datasety, zobrazuje výsledky a vykresľuje vizualizácie priamo v prehliadači. Komunikuje s backendom výhradne cez HTTP (fetch API). Backend (FastAPI) spravuje stav aplikácie, vykonáva validáciu vstupov a deleguje všetku výpočtovú prácu na ML modul. ML modul je úplne nezávislý od webovej vrstvy — možno ho importovať a testovať bez spusteného servera.

```
Frontend (HTML/JS)  <-->  FastAPI Backend  <-->  ML modul (sklearn Pipeline)
                                    |
                              Storage (joblib)
```

## Štruktúra projektu

```
svm-strojove-ucenie/
├── backend/
│   ├── app/
│   │   ├── api/                    # FastAPI routery (endpoints)
│   │   │   ├── datasets.py         # Načítavanie a správa datasetov
│   │   │   ├── training.py         # Trénovanie modelu
│   │   │   ├── prediction.py       # Predikcia pre nové dáta
│   │   │   └── model_info.py       # Stav modelu, vizualizácia, export
│   │   ├── core/
│   │   │   ├── config.py           # Konfigurácia aplikácie
│   │   │   └── state.py            # Globálny stav (dataset, model)
│   │   ├── ml/
│   │   │   ├── types.py            # Enumerácie a TypedDicts
│   │   │   ├── preprocessing.py    # ColumnTransformer pipeline
│   │   │   ├── model.py            # SVMClassifier trieda
│   │   │   ├── visualization.py    # PCA vizualizácia
│   │   │   └── column_detection.py # Automatická detekcia typov stĺpcov
│   │   ├── schemas/                # Pydantic request/response modely
│   │   └── main.py                 # FastAPI aplikácia, montovanie routerov
│   ├── storage/                    # Uložené modely (.joblib)
│   └── tests/                      # Pytest testy
├── data/
│   └── examples/                   # Príkladové CSV datasety
├── docs/                           # Dokumentácia
├── frontend/
│   ├── index.html                  # Single-page aplikácia
│   ├── js/app.js                   # Celá logika frontendu
│   └── css/styles.css              # Vlastné štýly
├── scripts/                        # Pomocné skripty a overovanie
├── requirements.txt                # Produkčné závislosti
└── requirements-dev.txt            # Vývojové závislosti (pytest, httpx)
```

## Inštalácia a spustenie

### Požiadavky

- Python 3.11 alebo novší
- pip

### Postup inštalácie

```bash
# 1. Klonovanie repozitára
git clone https://github.com/VladyslavOnishchenko/SVM-Strojove-Ucenie.git
cd SVM-Strojove-Ucenie

# 2. Vytvorenie virtuálneho prostredia
python -m venv venv

# Windows:
.\venv\Scripts\Activate.ps1

# Linux/macOS:
source venv/bin/activate

# 3. Inštalácia závislostí
pip install -r requirements-dev.txt

# 4. Generovanie príkladových datasetov (ak chýbajú)
python scripts/generate_sample_datasets.py

# 5. Spustenie servera
uvicorn backend.app.main:app --reload
```

Aplikácia beží na http://localhost:8000. Automaticky generovaná Swagger dokumentácia API je dostupná na http://localhost:8000/docs.

### Spustenie testov

```bash
# Rýchle testy (43 testov, GridSearchCV testy sú preskočené)
pytest backend/tests/ -v

# Vrátane pomalých GridSearchCV testov (46 testov celkovo)
pytest backend/tests/ -v --run-slow
```

## REST API — prehľad endpointov

| Metóda | Endpoint | Popis |
|--------|----------|-------|
| GET | /api/health | Health check |
| GET | /api/datasets/examples | Zoznam zabudovaných datasetov |
| POST | /api/datasets/examples/{name}/load | Načítať zabudovaný dataset |
| POST | /api/datasets/upload | Nahrať vlastný CSV súbor |
| GET | /api/datasets/current/schema | Schéma a typy stĺpcov aktuálneho datasetu |
| POST | /api/train/ | Trénovať SVM model |
| POST | /api/predict/ | Predikcia triedy pre nové vstupné dáta |
| GET | /api/model/status | Informácie o natrénovanom modeli |
| GET | /api/model/visualization | Dáta pre 2D PCA vizualizáciu |
| GET | /api/model/download | Stiahnuť model ako .joblib súbor |

Interaktívna dokumentácia so schémami requestov a odpovedí je dostupná automaticky na `/docs` (Swagger UI generovaný FastAPI).

## Formát CSV súboru

Pri nahrávaní vlastného datasetu platia tieto požiadavky:

- Prvý riadok musí obsahovať názvy stĺpcov
- Minimálny počet riadkov: 10, minimálny počet stĺpcov: 2
- Jeden stĺpec musí byť označiteľný ako cieľový (klasifikačná trieda)
- Podporované typy dát: číselné (int/float), kategoriálne (textové hodnoty), binárne (presne 2 unikátne hodnoty)
- Odporúčaná maximálna veľkosť: niekoľko tisíc riadkov — väčšie datasety síce fungujú, ale trénovanie bude trvať dlhšie

Typ každého stĺpca sa navrhuje automaticky pri načítaní datasetu. Používateľ môže skontrolovať navrhnuté typy pred spustením trénovania.

## Príkladové datasety

| Názov | Súbor | Riadky | Stĺpce | Typ úlohy | Typy príznakov |
|-------|-------|--------|--------|-----------|----------------|
| Iris | iris.csv | 150 | 5 | Viactriedna (3 triedy) | Číselné |
| Wine | wine.csv | 178 | 14 | Viactriedna (3 triedy) | Číselné |
| Bank Marketing | bank_marketing_sample.csv | 500 | 10 | Binárna | Mix (číselné + kategoriálne) |
| Heart Disease | heart_disease.csv | 300 | 12 | Binárna | Mix (číselné + binárne + kategoriálne) |

## Hlavné triedy a funkcie ML modulu

### `SVMClassifier` (backend/app/ml/model.py)

Hlavná trieda zapuzdrujúca celý životný cyklus modelu.

- `fit(df, test_size, cv_folds, random_state)` — natrénuje sklearn Pipeline (preprocessing + SVC), spustí krížovú validáciu a vyhodnotí model na testovacej množine, vráti `TrainingResults` s kompletnou sadou metrík
- `predict(input_data: dict)` — prijme slovník hodnôt príznakov pre jeden vstupný riadok, vrátí predikovanú triedu a pravdepodobnosti pre všetky triedy
- `save(path)` — serializuje natrénovaný pipeline, label encoder a schému stĺpcov do súboru pomocou joblib
- `load(path)` — triedna metóda, načíta uložený model zo súboru a rekonštruuje inštanciu `SVMClassifier`

### `build_preprocessor(column_schema)` (backend/app/ml/preprocessing.py)

Zostaví `ColumnTransformer`, ktorý aplikuje rôzne transformácie podľa typu stĺpca:

- Pre číselné stĺpce: `Pipeline([SimpleImputer(median), StandardScaler()])`
- Pre kategoriálne stĺpce: `Pipeline([SimpleImputer(most_frequent), OneHotEncoder(handle_unknown="ignore")])`
- Pre binárne stĺpce: `Pipeline([SimpleImputer(most_frequent), OrdinalEncoder()])`

Stĺpce s typom `target` a `ignore` sú vynechané nastavením `remainder="drop"`.

### `compute_decision_boundary_data(model, df)` (backend/app/ml/visualization.py)

Projektuje trénovacie dáta do 2D priestoru pomocou PCA, vygeneruje pravidelnú mriežku bodov pokrývajúcu celú oblasť, predikuje triedu každého bodu mriežky a vráti štruktúrovaný slovník pripravený na vykreslenie v Plotly (scatter body, predikčná mriežka, explained variance ratio).

## Stav aplikácie

Aplikácia drží všetok stav v jednej inštancii `AppState` dataclassu v pamäti. Tento stav obsahuje aktuálny dataset (pandas DataFrame), meno datasetu, natrénovaný model a čas trénovania. Po natrénovaní sa model automaticky uloží do súboru `backend/storage/current_model.joblib`. Pri reštarte servera sa uložený model automaticky nenačíta — používateľ musí datasaet načítať a model natrénovať znovu. Toto je známe obmedzenie, ktoré by šlo odstrániť pridaním automatického načítania modelu pri štarte aplikácie.

## Rozšírenie projektu

Nasledujúce rozšírenia by prirodzene nadviazali na súčasnú architektúru:

- **Podpora pre regresné úlohy** — pridanie SVR (Support Vector Regression) ako alternatívy k SVC pre predikciu číselných hodnôt
- **Perzistencia histórie trénovania** — ukladanie viacerých natrénovaných modelov s metadátami (dataset, hyperparametre, skóre) a možnosť prepínania medzi nimi
- **Porovnanie klasifikátorov** — trénovanie Random Forest, Logistic Regression a ďalších algoritmov vedľa SVM a vizuálne porovnanie ich výsledkov
- **Podpora pre väčšie datasety** — spracovanie veľkých CSV súborov po častiach (chunk processing) na zníženie pamäťových nárokov pri načítaní a predikcii
