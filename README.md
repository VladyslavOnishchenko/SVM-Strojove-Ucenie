# SVM Strojové Učenie

Webová aplikácia na trénovanie a používanie klasifikátorov Support Vector Machine (SVM) na vlastných CSV dátach. Projekt vznikol v rámci predmetu Strojové učenie a je určený pre študentov, ktorí chcú si v praxi vyskúšať celý postup strojového učenia — od nahratia dát po hodnotenie modelu a generovanie predpovedí.

Aplikácia využíva implementáciu SVM z knižnice scikit-learn a podporuje vlastné CSV súbory aj štyri zabudované príkladové datasety rôznej náročnosti a typu.

## O projekte

Projekt bol vytvorený ako semestrálna práca v predmete Strojové učenie. Cieľom bolo implementovať kompletnú pipeline strojového učenia — od predspracovanie dát po interaktívne predikcie — s webovým rozhraním prístupným bez inštalácie špeciálneho softvéru.

Pre klasifikáciu (úloha priradiť vstupný vzor k jednej z predem definovaných tried) bol zvolený algoritmus SVM z dôvodu jeho preukázanej výkonnosti na stredne veľkých datasetoch, schopnosti pracovať s rôznymi typmi dát prostredníctvom kernelových funkcií (matematické transformácie, ktoré umožňujú SVM nájsť nelineárne hranice medzi triedami) a relatívne malého počtu hyperparametrov na ladenie.

Aplikácia umožňuje nahrať CSV súbor, označiť typ každého stĺpca (číselný, kategorický, binárny, cieľový alebo ignorovaný), nastaviť hyperparametre modelu alebo zapnúť automatické ladenie pomocou GridSearchCV, spustiť trénovanie, zobraziť metriky hodnotenia a 2D vizualizáciu rozhodovacích hraníc a nakoniec generovať predikcie pre nové vstupy.

## Funkcie

- Nahrávanie vlastných CSV súborov alebo výber zo štyroch zabudovaných datasetov
- Manuálne označenie typov stĺpcov s automatickými návrhmi na základe heuristiky
- Trénovanie SVM s výberom kernelu (linear, rbf, poly, sigmoid) a parametrov C a gamma
- Automatické ladenie hyperparametrov cez GridSearchCV (voliteľné)
- Hodnotenie modelu: presnosť, recall, F1, matica zámen, krížová validácia
- Interaktívne predikcie pre nové vstupné vzory
- 2D vizualizácia rozhodovacích hraníc pomocou PCA projekcie
- Stiahnutie natrénovaného modelu ako súbor `.joblib`

## Zabudované datasety

| Názov | Riadky | Stĺpce | Typ úlohy | Opis |
|-------|--------|--------|-----------|------|
| Iris | 150 | 5 | viactriedna | Tri druhy kosatcov podľa štyroch morfologických znakov |
| Wine | 178 | 14 | viactriedna | Chemická analýza talianskych vín zo troch pestovateľských oblastí |
| Bank Marketing | 500 | 10 | binárna | Syntetické bankové dáta — prihlásenie klienta na termínovaný vklad |
| Heart Disease | 300 | 12 | binárna | Syntetický Cleveland-style dataset srdcových ochorení |

Iris je klasický referenčný dataset vhodný na overenie základnej funkčnosti — všetky príznaky sú číselné a triedy sú dobre oddeliteľné. Wine rozširuje záber na viactriednu klasifikáciu s 13 číselnými chemickými znakmi. Bank Marketing demonštruje spracovanie zmiešaných dátových typov (číselné, kategorické, binárne stĺpce) v realistickejšom scenári. Heart Disease kombinuje oba aspekty — zmiešané typy a binárnu klasifikáciu — pre lekársky kontext, kde je interpretovateľnosť modelu dôležitá.

## Použité technológie

- **Python 3.11+** — základný programovací jazyk celého projektu
- **FastAPI** — framework pre REST API backend; zabezpečuje smerovanie, validáciu vstupov a automatickú dokumentáciu na `/docs`
- **scikit-learn** — implementácia SVM (`SVC`), predspracovanie dát, krížová validácia a GridSearchCV
- **pandas** — načítanie a manipulácia s CSV dátami vo forme DataFrame
- **NumPy** — numerické operácie, generovanie syntetických datasetov a PCA výpočty
- **joblib** — serializácia a deserializácia natrénovaného modelu na disk
- **Tailwind CSS** — štýlovanie frontendového rozhrania (Stage 4)
- **pytest** — jednotkové a integračné testy; konfigurácia pomalých testov cez vlastný marker
- **GitHub Actions** — automatické spúšťanie testov pri každom push a pull requeste

## Inštalácia

```powershell
git clone https://github.com/VladyslavOnishchenko/SVM-Strojove-Ucenie.git
cd SVM-Strojove-Ucenie

python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# Linux / macOS
# source venv/bin/activate

pip install -r requirements-dev.txt
```

Príkladové datasety sú súčasťou repozitára v adresári `data/examples/`. Ak by bolo potrebné ich znovu vygenerovať:

```powershell
python scripts/generate_sample_datasets.py
```

## Spustenie

```powershell
uvicorn backend.app.main:app --reload
```

Aplikácia bude dostupná na `http://localhost:8000`. Interaktívna dokumentácia API je automaticky generovaná FastAPI na adrese `http://localhost:8000/docs`.

## Použitie

1. Otvor aplikáciu v prehliadači na `http://localhost:8000`.
2. Vyber jeden zo zabudovaných datasetov alebo nahraj vlastný CSV súbor.
3. Skontroluj automaticky navrhnuté typy stĺpcov a prípadne ich uprav.
4. Nastav hyperparametre SVM (kernel, C, gamma) alebo zapni automatické ladenie.
5. Spusti trénovanie a počkaj na výsledky.
6. Prehliadni si metriky hodnotenia (presnosť, matica zámen, CV skóre) a 2D vizualizáciu rozhodovacích hraníc.
7. Zadaj hodnoty nového vzoru a získaj predikciu triedy s pravdepodobnosťami.

## Štruktúra projektu

```
SVM-Strojove-Ucenie/
├── backend/                  # Python backend (FastAPI + ML)
│   ├── app/
│   │   ├── api/              # FastAPI routery (datasety, trénovanie, predikcia, model)
│   │   ├── core/             # Konfigurácia a globálny stav aplikácie
│   │   ├── ml/               # ML modul (preprocessing, model, vizualizácia, typy)
│   │   ├── schemas/          # Pydantic schémy pre validáciu API požiadaviek
│   │   └── main.py           # Vstupný bod FastAPI aplikácie
│   ├── storage/              # Úložisko natrénovaného modelu (.joblib)
│   └── tests/                # Pytest testy (unit + API integračné)
├── data/
│   └── examples/             # Štyri príkladové CSV datasety
├── docs/                     # Technická a teoretická dokumentácia
├── frontend/                 # Statické súbory frontendu (HTML, CSS, JS)
├── scripts/                  # Pomocné skripty (generátor datasetov)
├── requirements.txt          # Produkčné závislosti
├── requirements-dev.txt      # Vývojové závislosti (+ pytest, httpx)
└── conftest.py               # Globálna pytest konfigurácia (slow marker, sys.path)
```

## Architektúra

Aplikácia je rozdelená na tri vrstvy: frontend (statické súbory obsluhované priamo FastAPI), backend REST API a ML modul. Frontend komunikuje s backendom výhradne cez JSON API — všetka logika strojového učenia zostáva na serveri.

Backend je organizovaný do štyroch routerov (`/api/datasets`, `/api/train`, `/api/predict`, `/api/model`), ktoré zdieľajú jednoduchý in-memory stav (`AppState`) s aktuálne načítaným datasetom a natrénovaným modelom. Tento prístup postačuje pre jednopouzivartelské nasadenie; pre produkčné použitie by stav musel byť uložený v databáze alebo cache.

ML modul (`backend/app/ml/`) je zámerné oddelený od API vrstvy — neimportuje FastAPI ani žiadne webové závislosti. Vďaka tomu je možné spúšťať unit testy ML logiky (predspracovanie, trénovanie, predikcia) nezávisle od API, čo skracuje čas spätnä väzby pri vývoji.

## Algoritmus SVM

Support Vector Machine hľadá hranicu (v 2D priamku, vo vyšších rozmeroch hyperplochu) medzi triedami v priestore príznakov tak, aby bol odstup (margin) medzi hranicou a najbližšími vzormi z každej triedy maximálny. Vzory ležiace najbližšie k hranici sa nazývajú support vectors a sú to jediné body, ktoré určujú polohu hranice.

Kernelová funkcia transformuje vstupné dáta do vyššiedimenziálneho priestoru, kde môže byť lineárna hranica dostatočná aj pre nelineárne oddeliteľné triedy. Lineárny kernel hľadá priamu hranicu v pôvodnom priestore príznakov a je rýchly na veľkých datasetoch. RBF (Radial Basis Function) kernel meria podobnosť bodov pomocou gaussovskej funkcie a dobre funguje ako predvolená voľba pri neznámej štruktúre dát. Polynomický kernel umožňuje polynomické hranice a hodí sa, keď sa predpokladá polynomická závislosť medzi príznakmi. Sigmoidný kernel je analogický neurónové sieti a používa sa zriedkavejšie.

Parameter C riadi kompromis medzi šírkou marginu a počtom chybne klasifikovaných trénovacích vzoriek. Malé C vytvára mäkkú hranicu, ktorá toleruje viac chýb na trénovacej množine, ale lepšie generalizuje. Veľké C presúva hranicu bližšie k trénovacím vzorkám (tvrdá hranica), čo môže viesť k pretrénovaniu.

Parameter gamma (pre RBF, poly a sigmoid kernel) určuje, ako ďaleko siaha vplyv jedného trénovacieho bodu. Malá gamma znamená, že každý bod ovplyvňuje širokú oblasť (hladšia hranica), veľká gamma obmedzuje vplyv na bezprostredné okolie bodu (ostrejšia hranica, riziko prílišného prispôsobenia). Nastavenie `gamma="scale"` (predvolené) automaticky škáluje hodnotu podľa počtu príznakov a rozptylu dát.

V projekte sa používa `SVC` (Support Vector Classifier) zo scikit-learn, ktorý implementuje všetky štyri kernely, podporuje viactriednu klasifikáciu stratégiou one-vs-one a vďaka parametru `probability=True` poskytuje aj pravdepodobnostné odhady tried.

## Spustenie testov

```powershell
# Rýchle testy (preprocessing, modely, API endpointy)
pytest backend/tests/ -v

# Vrátane pomalých testov (GridSearchCV auto-tune)
pytest backend/tests/ -v --run-slow
```

Projekt obsahuje 46 testov rozdelených do štyroch skupín: predspracovanie dát (`test_preprocessing.py`), trénovanie a predikcia na datasetoch Iris a Bank Marketing (`test_model_iris.py`, `test_model_mixed.py`), automatické ladenie hyperparametrov (`test_model_autotune.py` — 3 testy označené ako `slow`) a integračné testy REST API endpointov (`test_api_datasets.py`, `test_api_training.py`, `test_api_prediction.py`, `test_api_model_info.py`).

## API

| Metóda | Cesta | Popis |
|--------|-------|-------|
| GET | /api/health | Kontrola stavu aplikácie |
| GET | /api/datasets/examples | Zoznam zabudovaných datasetov |
| POST | /api/datasets/examples/{name}/load | Načítať príkladový dataset |
| POST | /api/datasets/upload | Nahrať vlastný CSV súbor |
| GET | /api/datasets/current/schema | Schéma aktuálneho datasetu s navrhovanými typmi |
| POST | /api/train | Trénovať SVM model |
| POST | /api/predict | Predikcia pre nový vstupný vzor |
| GET | /api/model/status | Stav natrénovaného modelu |
| GET | /api/model/visualization | Dáta pre 2D PCA vizualizáciu |
| GET | /api/model/download | Stiahnuť model ako .joblib súbor |

Interaktívnu dokumentáciu so možnosťou priameho testovania endpointov generuje FastAPI automaticky na adrese `/docs`.

## CI/CD

Každý push a pull request do vetvy `main` spustí GitHub Actions workflow, ktorý nainštaluje závislosti a spustí `pytest backend/tests/ -v` na Pythone 3.11. Pomalé testy (GridSearchCV) sú v CI vynechané — spúšťajú sa manuálne s prepínačom `--run-slow`.

## Licencia

MIT

## Autor

Vladyslav Onishchenko, predmet Strojové učenie, 2026.
