# 🖐️ SHARP (Smart Hand Automated Recognition Project)

[![CI Pipeline](https://github.com/djelalb/cnam-ai-sharp/actions/workflows/ci.yml/badge.svg)](https://github.com/djelalb/cnam-ai-sharp/actions)

Architecture MLOps de production pour la détection de mains (0-5 doigts) avec YOLO11.

## 🚀 Démarrage Rapide

### Local
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# Créer .env.local avec ULTRALYTICS_API_KEY
python -m src.main all  # Pipeline complète (Cloud Training)
python -m src.serving.api.main  # Lancer l'interface
```

### Docker
```bash
docker-compose up --build
```
Accès : `http://localhost:8000`

## 🏗️ Principes d'Architecture
- **SOLID** : Inférence découplée du transport (WebSocket).
- **Type-Safe** : Configuration validée par Pydantic.
- **CI/CD** : Linting (Ruff) et tests (Pytest) automatisés.
- **FIRST** : Tests unitaires rapides et isolés.

## 🛠️ Commandes Utiles
- **Tests** : `pytest tests/`
- **Lint** : `ruff check .`
- **Format** : `ruff format .`
