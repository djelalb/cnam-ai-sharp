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

## ⚙️ Utilisation de la Pipeline ML
Le projet inclut un orchestrateur pour gérer les différentes étapes du cycle de vie du modèle :

```bash
# Lancer l'intégralité de la pipeline (extraction, préparation, training, validation)
python -m src.main all

# Lancer une étape spécifique
python -m src.main [extraction|preparation|training|validation]
```

## 🧪 Qualité et Maintenance
Le projet respecte les standards de développement modernes pour garantir la fiabilité du code.

- **Tests unitaires** : `pytest tests/`
- **Linting & Formatage** : `ruff check .`
