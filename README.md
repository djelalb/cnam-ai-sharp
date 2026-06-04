# 🖐️ SHARP - Substantial Hand Analysis & Recognition Pipeline

[![CI Pipeline](https://github.com/djelalb/cnam-ai-sharp/actions/workflows/ci.yml/badge.svg)](https://github.com/djelalb/cnam-ai-sharp/actions)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**SHARP** est un projet de Computer Vision industriel dédié à la détection et à l'analyse de mains (0 à 5 doigts). Il s'appuie sur une architecture **Cloud-Native MLOps** pilotée par l'API Ultralytics Platform.

---

## 🏗️ Architecture du Projet

Le projet suit une structure modulaire stricte garantissant la séparation des responsabilités :

```text
.
├── .github/workflows/   # CI/CD (GitHub Actions)
├── data/                # Données (exclu de Git)
├── docs/                # Documentation stratégique (Choix des hyperparamètres)
├── src/
│   ├── main.py          # Orchestrateur central (Point d'entrée unique)
│   ├── config.py        # Configuration Type-Safe (Pydantic V2)
│   └── pipeline/        # Briques logiques (Extraction, Training Cloud, etc.)
├── tests/               # Tests unitaires et smoke tests
└── .env.local           # Secrets et configurations locales (à créer)
```

---

## 🛠️ Installation & Setup

### 1. Prérequis
*   Python 3.12+
*   Un compte [Ultralytics HUB](https://platform.ultralytics.com/) et une clé API.

### 2. Installation
```bash
# Cloner et entrer dans le dossier
git clone https://github.com/djelalb/cnam-ai-sharp.git
cd cnam-ai-sharp

# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/macOS

# Installer les dépendances et les hooks
pip install -r requirements.txt
pre-commit install
```

### 3. Configuration des secrets
Créez un fichier `.env.local` à la racine :
```env
ULTRALYTICS_API_KEY=votre_cle_api_ici
ULTRALYTICS_USERNAME=votre_nom_utilisateur
ULTRALYTICS_PROJECT=ai-sharp
ULTRALYTICS_DATASET=hands
```

---

## 🚀 Utilisation de la Pipeline

Le projet utilise un **Orchestrateur Central**. Plus besoin de chercher quel script lancer.

### Exécution par étape
```bash
# 1. Résoudre l'ID du dataset distant
python -m src.main extraction

# 2. Valider et Préparer (Automatique / Cloud-Native)
python -m src.main validation
python -m src.main preparation

# 3. Lancer l'entraînement sur GPU Cloud
python -m src.main training
```

### Exécution complète (Recommandé)
```bash
python -m src.main all
```

---

## 🧠 Stratégie d'IA & Qualité

*   **Modèle** : YOLO11n (Nano) pour un maximum de FPS.
*   **Hyperparamètres** : Optimisés pour la robustesse (Mosaïque 1.0, Rotation 25°). Consultez [docs/training_strategy.md](docs/training_strategy.md) pour les justifications détaillées.
*   **Qualité du Code** : Strictement conforme à **Ruff** et validé par une **CI Pipeline**.
*   **Tests** : Lancez les tests avec `pytest tests/`.

---

## 📜 Normes de Contribution
*   **Conventional Commits** : `feat:`, `fix:`, `chore:`, `docs:`.
*   **Qualité** : Chaque Pull Request déclenche automatiquement la suite de tests et de linting.
