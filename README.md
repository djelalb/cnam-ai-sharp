# 🖐️ SHARP (Smart Hand Automated Recognition Project)

[![CI Pipeline](https://github.com/djelalb/cnam-ai-sharp/actions/workflows/ci.yml/badge.svg)](https://github.com/djelalb/cnam-ai-sharp/actions)

## 📝 Présentation
SHARP est une solution de vision par ordinateur dédiée à la **détection de mains en temps réel**. S'appuyant sur l'architecture **YOLO11**, ce projet propose une stack complète allant de la préparation des données à l'inférence haute performance.

L'objectif est de fournir un système capable de traiter des flux vidéo en direct via WebSocket, garantissant une latence minimale et une précision optimale pour des applications interactives.

## 🚀 Démarrage Rapide

### Déploiement via Docker (Recommandé)
Le moyen le plus simple de tester l'application est d'utiliser Docker. Cela lance l'API de serving et l'interface utilisateur sans nécessiter d'installation locale de Python.

```bash
docker-compose up --build
```
Accès à l'interface : **[http://localhost:8000](http://localhost:8000)**

### Installation Locale (Développement)
Pour entraîner le modèle ou contribuer au code :

1. **Environnement** :
   ```bash
   python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Configuration** : Créer un fichier `.env.local` avec vos variables d'environnement (notamment `ULTRALYTICS_API_KEY`).

## ⚙️ Utilisation

### Application Web (Serving)
Pour lancer l'interface de détection en temps réel localement :
```bash
python -m src.serving.api.main
```
Accès : **[http://localhost:8000](http://localhost:8000)**

### Pipeline ML
Le cycle de vie ML comporte **6 étapes**. Le sujet autorisant l'usage de la
plateforme Ultralytics, la gestion du dataset et l'entraînement GPU y sont
réalisés : les étapes correspondantes sont donc **déléguées** (tracées plutôt
qu'exécutées localement). Chaque étape reste matérialisée par un module dédié.

| # | Étape | Statut | Réalisation |
|---|-------|--------|-------------|
| 1 | Extraction | local | résout l'ID du dataset hébergé |
| 2 | Validation | délégué | intégrité/cohérence des annotations (outil d'annotation) |
| 3 | Préparation | délégué | format YOLO + split 60/20/20 (seed 42) sur la plateforme |
| 4 | Training | local | déclenche le job d'entraînement GPU cloud |
| 5 | Évaluation | délégué | `model.val()` auto sur le split test → experiment tracking |
| 6 | Sélection | manuel | run retenu après analyse du tracking (`exp-14.pt`) |

```bash
# Dérouler l'intégralité de la pipeline (les 6 étapes)
python -m src.main all

# Lancer une étape spécifique
python -m src.main [extraction|validation|preparation|training|evaluation|selection]
```

## 🧪 Qualité et Maintenance
Le projet respecte les standards de développement modernes pour garantir la fiabilité du code.

- **Tests unitaires** : `pytest tests/`
- **Linting & Formatage** : `ruff check .`
