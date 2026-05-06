# SVM Strojové Učenie

Web aplikácia pre trénovanie a používanie Support Vector Machine klasifikátorov na používateľských CSV dátach.

## O projekte

SVM Strojové Učenie je interaktívna webová aplikácia, ktorá umožňuje používateľom nahrávať CSV datasety a trénovať na nich modely Support Vector Machine (SVM). Aplikácia poskytuje intuitívne rozhranie pre prípravu dát, trénovanie modelov a generovanie predpovedí.

## Funkcie

- 📤 **Nahrávanie CSV** - Podpora nahrávania vlastných dátových súborov
- 🏷️ **Označenie typov stĺpcov** - Konfigurácia typov premenných (vstupné, výstupné)
- 🤖 **Trénovanie SVM modelu** - Jednoduché a efektívne trénovanie klasifikátorov
- 📊 **Vizualizácia výsledkov** - Grafické zobrazenie výsledkov a metrík
- 🔮 **Interaktívne predikcie** - Testovanie modelu na nových dátach

## Inštalácia

1. Naklonujte repozitár:
```bash
git clone <url-repozitáru>
cd svm-strojove-ucenie
```

2. Vytvorte virtuálne prostredie:
```bash
python -m venv venv
source venv/bin/activate  # Na Windows: venv\Scripts\activate
```

3. Nainštalujte závislosti:
```bash
pip install -r requirements-dev.txt
```

## Spustenie

1. Spustite backend server:
```bash
uvicorn backend.app.main:app --reload
```

2. Otvorte aplikáciu v prehliadači:
```
http://localhost:8000
```

## Testovanie

Spustite testovaciu sadu:
```bash
pytest backend/tests/ -v
```

## Štruktúra projektu

```
svm-strojove-ucenie/
├── backend/                 # Backend časť aplikácie
│   ├── app/                # Hlavný balík aplikácie
│   │   ├── api/           # API endpointy
│   │   ├── ml/            # Modul strojového učenia
│   │   ├── schemas/       # Pydantic schémy
│   │   ├── core/          # Jadro aplikácie (konfigurácia)
│   │   └── main.py        # Vstupný bod FastAPI
│   ├── storage/           # Úložisko pre uložené modely
│   └── tests/             # Testy
├── frontend/              # Frontend časť aplikácie
│   ├── index.html         # Hlavná stránka
│   ├── css/               # Štýly
│   └── js/                # JavaScript skripty
├── data/                  # Dátové sady
│   └── examples/          # Príklady dátových súborov
├── docs/                  # Dokumentácia
│   ├── technical/         # Technická dokumentácia
│   └── theory/            # Teoretický základ
├── .github/
│   └── workflows/         # GitHub Actions workflows
├── requirements.txt       # Produkčné závislosti
├── requirements-dev.txt   # Vývojové závislosti
├── .gitignore            # Git ignore pravidlá
├── LICENSE               # MIT licencia
└── README.md             # Tento súbor
```

## Technológie

- **Backend:** Python 3.11+, FastAPI, Uvicorn
- **ML:** scikit-learn, pandas, numpy, joblib
- **Frontend:** Vanilla HTML/JS, Tailwind CSS (CDN)
- **Testing:** pytest
- **CI/CD:** GitHub Actions

## Licencia

Tento projekt je vydaný pod MIT licenciou. Pozrite si súbor [LICENSE](LICENSE) pre viac informácií.

---

**Poznámka:** Toto je fáza 1 (Stage 1) vývoja - Repository Infrastructure. Backend a frontend sú zatiaľ v základnej podobe a budú rozšírené v nasledujúcich fázach.
