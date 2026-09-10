# Customer Churn Prediction — MLOps Pipeline (TP 1)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Poetry](https://img.shields.io/badge/Packaging-Poetry-blueviolet.svg)](https://python-poetry.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Linter: Flake8](https://img.shields.io/badge/Linter-Flake8-informational.svg)](https://flake8.pycqa.org/)
[![Tests: Pytest](https://img.shields.io/badge/Tests-Pytest-green.svg)](https://docs.pytest.org/)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

Ce dépôt constitue le livrable de référence pour le **Travaux Pratiques N° 1** du module **MLOps (Machine Learning Operations)** destiné aux élèves ingénieurs en informatique (Spécialité Data Science & IA).

Il illustre la transformation d'un notebook expérimental monolithique vers un **package logiciel modulaire, standardisé, typé, documenté, testé et prêt pour l'industrialisation**.

---

## 👨‍🏫 Informations Générales

* **Enseignant Responsable :** Haythem Ghazouani
* **Contact officiel :** `haythemghz@yahoo.fr`
* **Public :** Élèves Ingénieurs Terminale (Bac+5) — DS & IA
* **Durée du TP :** 3 Heures
* **Module :** MLOps (42h global)

---

## 🏗️ Architecture du Projet

Le projet adopte les standards de génie logiciel appliqués à la Data Science (*Cookiecutter Data Science*) :

```text
mlops-tp1-customer-churn/
├── configs/
│   └── config.yaml               # Fichier centralisé de configuration (zéro hardcoding)
├── data/
│   └── raw/
│       └── customer_churn.csv    # Données brutes immuables (1000 clients réels)
├── models/
│   └── .gitkeep                  # Dossier de réception des modèles sérialisés (.pkl)
├── notebooks/
│   └── notebook_churn_messy.ipynb# Prototype initial avant refactorisation
├── src/                          # Package Python installable
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── load_data.py          # Ingestion et validation des données brutes
│   ├── features/
│   │   ├── __init__.py
│   │   └── preprocess.py         # Fonctions pures de nettoyage et One-Hot Encoding
│   └── models/
│       ├── __init__.py
│       └── train.py              # Pipeline CLI d'entraînement et évaluation
├── tests/                        # Suite de tests unitaires automatisés
│   ├── __init__.py
│   ├── test_data.py              # Validation d'ingestion et résilience aux erreurs
│   ├── test_features.py          # Validation des bornes [0, 1] et découpage train/test
│   └── test_model.py             # Validation de convergence et performance baseline
├── .gitignore                    # Exclusion stricte des artefacts et caches
├── .pre-commit-config.yaml       # Sentinelle locale : black, isort, flake8
├── pyproject.toml                # Gestion des dépendances déterministe avec Poetry
└── README.md                     # Documentation officielle du projet
```

---

## ⚡ Démarrage Rapide (Quickstart)

### 1. Cloner le dépôt
```bash
git clone <URL_DE_VOTRE_DEPOT_GITHUB>
cd mlops-tp1-customer-churn
```

### 2. Installer les dépendances avec Poetry
Assurez-vous d'avoir installé [Poetry](https://python-poetry.org/) sur votre machine :
```bash
poetry install
```
*Toutes les dépendances de production (`pandas`, `scikit-learn`, `numpy`, `pyyaml`) ainsi que les outils de test et qualité (`pytest`, `black`, `isort`, `flake8`, `pre-commit`) sont automatiquement isolées dans un environnement virtuel dédié.*

---

## 🚀 Exécution du Pipeline

### Entraîner et évaluer le modèle via l'interface CLI :
```bash
poetry run python src/models/train.py --config configs/config.yaml
```

**Sortie attendue dans la console :**
```text
[1/4] Ingestion des données depuis data/raw/customer_churn.csv...
[2/4] Preprocessing et Feature Engineering...
[3/4] Entraînement du modèle LogisticRegression...
[4/4] Évaluation des performances...
=== MÉTRIQUES EN PRODUCTION ===
  ACCURACY: 0.8050
  F1_SCORE: 0.5412
  ROC_AUC:  0.8099
Modèle sauvegardé avec succès dans : models/churn_logistic_regression.pkl
```

---

## 🧪 Tests Unitaires Automatisés (`pytest`)

Exécuter l'intégralité de la suite de tests :
```bash
poetry run pytest -v
```

**Résultat attendu :**
```text
tests/test_data.py::test_load_raw_data_valid PASSED
tests/test_data.py::test_load_raw_data_not_found PASSED
tests/test_features.py::test_clean_raw_dataframe PASSED
tests/test_features.py::test_encode_features_normalization_bounds PASSED
tests/test_features.py::test_split_data_proportions PASSED
tests/test_model.py::test_train_and_evaluate_accuracy_baseline PASSED
====== 6 passed in 1.20s ======
```

---

## 🛡️ Qualité de Code & Sentinelle Git (`pre-commit`)

### 1. Activer les hooks pre-commit
```bash
poetry run pre-commit install
```

### 2. Tester les linters manuellement
```bash
# Formatage du code avec Black
poetry run black src/ tests/

# Tri ordonné des imports avec isort
poetry run isort src/ tests/

# Analyse statique PEP 8 avec Flake8
poetry run flake8 src/ tests/ --max-line-length=88
```

### 3. Exécuter tous les hooks sur l'ensemble des fichiers
```bash
poetry run pre-commit run --all-files
```

---

## 📋 Règles de Software Craftsmanship Appliquées

1. **Typage Statique (`Type Hints`) :** Toutes les signatures de fonctions précisent explicitement les types d'entrée et de sortie (`pd.DataFrame`, `List[str]`, `Tuple`, etc.).
2. **Docstrings Conformes (Google Style) :** Chaque fonction est documentée avec son rôle, ses arguments (`Args`), sa valeur de retour (`Returns`) et les exceptions éventuelles (`Raises`).
3. **Fonctions Pures & Zéro Effets de Bord :** Les fonctions de preprocessing opèrent sur des copies de données (`df.copy()`) et ne modifient pas l'état global.
4. **Zéro Hardcoding :** Les chemins de fichiers, les hyperparamètres (`regularization_c`, `max_iter`) et les listes de colonnes sont tous externalisés dans `configs/config.yaml`.

---

## 📌 Prochaine Étape : Vers le Chapitre 2 (DVC)

Dans le prochain chapitre et TP, ce dépôt sera enrichi avec **DVC (Data Version Control)** :
* Versioning du fichier `customer_churn.csv` sans saturer Git.
* Construction d'un DAG de pipeline reproductible via `dvc.yaml` et `dvc repro`.

---

## 📄 Licence & Contact
* **Enseignant :** Haythem Ghazouani (`haythemghz@yahoo.fr`)
* **Usage :** Support pédagogique officiel pour le module MLOps.
