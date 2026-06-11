# 🖐️ SHARP — Smart Hand Automated Recognition Project

[![CI Pipeline](https://github.com/djelalb/cnam-ai-sharp/actions/workflows/ci.yml/badge.svg)](https://github.com/djelalb/cnam-ai-sharp/actions)

Détection en temps réel du **nombre de doigts levés** d'une ou plusieurs mains,
via un modèle **YOLO11** entraîné sur la plateforme Ultralytics. Le projet couvre
les deux livrables du sujet : une **pipeline ML** (extraction → évaluation) et une
**application de serving** dockerisée (API + dashboard webcam).

Les 6 classes détectées sont `0_doigt` … `5_doigts` ; chaque main visible reçoit
sa propre bounding box, et le dashboard affiche la **somme des doigts** de toutes
les mains à l'écran.

## 🗂️ Structure du projet

```
src/
├── config.py              # configuration centralisée (Pydantic Settings)
├── main.py                # orchestrateur de la pipeline ML (argparse)
├── pipeline/              # une étape ML par module (SRP)
│   ├── hub.py             # client REST Ultralytics (projet, poids des runs)
│   ├── extraction.py      # API HUB → manifeste NDJSON
│   ├── validation.py      # cohérence des annotations du manifeste
│   ├── preparation.py     # NDJSON → arborescence YOLO + config.yaml
│   ├── training.py        # pilotage du job d'entraînement GPU cloud
│   ├── evaluation.py      # télécharge le run choisi → model.val()
│   └── selection.py       # choix manuel du modèle retenu
└── serving/
    ├── api/               # FastAPI : inférence (inference.py) + WebSocket (main.py)
    └── frontend/          # dashboard webcam (HTML/Canvas/Tailwind)
models/                    # tous les poids .pt (modèle de prod + runs téléchargés)
tests/                     # tests unitaires (pytest)
```

## ⚙️ Prérequis & installation

- Python **3.12**, et un GPU NVIDIA / Apple Silicon recommandé pour l'entraînement.
- Une clé API Ultralytics pour la pipeline (l'app de serving n'en a pas besoin).

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Créer un fichier `.env.local` (lu automatiquement par `config.py`) :

```dotenv
ULTRALYTICS_API_KEY=<votre_clé>
ULTRALYTICS_USERNAME=<votre_compte>
ULTRALYTICS_DATASET=hands
ULTRALYTICS_PROJECT=ai-sharp
```

## 🧠 Pipeline ML

Le dataset est annoté et l'entraînement GPU exécuté **sur Ultralytics** ; l'export
se présente sous forme d'un **manifeste NDJSON** (URLs des images + boîtes), converti
localement au format YOLO. Chaque étape est un module dédié.

| # | Étape | Lieu | Réalisation |
|---|-------|------|-------------|
| 1 | Extraction | local | résout l'ID via l'API HUB et télécharge le manifeste NDJSON |
| 2 | Validation | local | cohérence des annotations du manifeste (classes, coords ∈ [0,1]) |
| 3 | Préparation | local | NDJSON → images + labels YOLO + `config.yaml` |
| 4 | Training | local → cloud | déclenche le job d'entraînement GPU sur Ultralytics |
| 5 | Évaluation | local | télécharge les poids du run choisi → `model.val()` (split val) |
| 6 | Sélection | manuel | run retenu après analyse de l'experiment tracking (`ai-sharp-exp-prod.pt`) |

```bash
# Dérouler toute la pipeline
python -m src.main all

# Ou une étape précise
python -m src.main [extraction|validation|preparation|training|evaluation|selection]

# Évaluer un run précis sans saisie interactive (nom visible dans l'URL du run)
python -m src.main evaluation --model ai-sharp-exp-prod
```

> 🔎 L'**évaluation** porte sur le modèle *réellement entraîné* : le nom du run est
> demandé (ou passé via `--model`), ses poids `.pt` sont téléchargés depuis la
> plateforme, puis évalués. Le modèle de production (`ai-sharp-exp-prod.pt`) reste réservé au
> serving et à l'étape de sélection.

> ⏳ La **préparation** télécharge les images depuis des URLs CDN signées (durée de
> vie limitée) : si elles ont expiré, relancer `extraction` pour régénérer l'export.
>
> ⚠️ Le **training** déclenche un vrai job GPU cloud (consomme le quota Ultralytics).

## 🎥 Application de serving

API **FastAPI** : le frontend capture la webcam, envoie chaque frame via **WebSocket**,
le backend exécute YOLO et renvoie les boîtes + la somme des doigts, superposées sur
le flux. Tous les poids vivent dans `models/` ; le modèle de production
(`models/ai-sharp-exp-prod.pt`, défini par `MODEL_PATH`) est chargé au démarrage du conteneur.

```bash
# Docker (recommandé) — charge le modèle et expose le dashboard
docker-compose up --build

# Ou en local
python -m src.serving.api.main
```

Interface : **http://localhost:8000** — santé : `curl http://localhost:8000/health`

> 📦 Les poids `.pt` ne sont pas versionnés (gitignorés, fichiers lourds). Avant de
> lancer l'app ou de builder l'image Docker, s'assurer que le modèle de production
> est présent dans `models/` :
> ```bash
> python -m src.main evaluation --model ai-sharp-exp-prod   # télécharge models/ai-sharp-exp-prod.pt
> ```
> (ou copier le fichier directement dans `models/`). Pour servir un autre modèle,
> placer son `.pt` dans `models/` et mettre à jour `MODEL_PATH` dans `config.py`.

## 🧪 Qualité

```bash
pytest tests/                 # tests unitaires
ruff check . && ruff format --check .   # lint + format
pre-commit run --all-files    # hooks (lint/format à chaque commit)
```
