# SHARP - Hand Detection Project

## Description
SHARP (Substantial Hand Analysis & Recognition Pipeline) est un projet de Computer Vision dédié à la détection et à l'analyse de mains. Il s'appuie sur une architecture MLOps robuste pour garantir la reproductibilité et la qualité du code.

## Prérequis
- Python 3.12+
- Git

## Installation

1.  **Cloner le dépôt :**
    ```bash
    git clone https://github.com/votre-compte/cnam-ai-sharp.git
    cd cnam-ai-sharp
    ```

2.  **Créer un environnement virtuel :**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Sur Linux/macOS
    # ou
    .\venv\Scripts\activate  # Sur Windows
    ```

3.  **Installer les dépendances :**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

4.  **Installer les hooks de pre-commit :**
    ```bash
    pre-commit install
    ```

## Lancement

### Pipeline
Pour exécuter les différentes étapes du pipeline :
```bash
# Exemple pour l'extraction
python -m src.pipeline.extraction
```

### Serving
Pour lancer l'API ou le frontend :
```bash
# Lancement de l'API
python -m src.serving.api.main

# Lancement du Frontend
# (Commande à définir selon le framework choisi)
```

## Conventional Commits
Ce projet respecte la norme **Conventional Commits**.
