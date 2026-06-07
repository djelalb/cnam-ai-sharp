"""
Étape 4 — TRAINING (local : pilotage / cloud : exécution).

L'entraînement s'exécute sur les GPU d'Ultralytics. Ce module pilote ce job
distant via l'API REST : création de l'entité modèle (hyper-paramètres et
augmentations issus de la config) puis déclenchement du compute cloud.
Le suivi des métriques (loss, mAP, etc.) se fait dans l'experiment tracking.
"""

import logging
import sys

import requests

from src.config import settings

logger = logging.getLogger(__name__)


class CloudTrainer:
    """Pilote un entraînement YOLO sur l'infrastructure cloud d'Ultralytics."""

    def __init__(self) -> None:
        self.headers = {
            "Authorization": f"Bearer {settings.ULTRALYTICS_API_KEY}",
            "Content-Type": "application/json",
        }
        self.api_url = "https://platform.ultralytics.com/api"

    def train(self) -> None:
        """Crée l'entité modèle puis démarre le compute cloud."""
        dataset_id = self._read_dataset_id()
        project_id = self._resolve_project_id()

        model_id = self._create_model(dataset_id, project_id)
        logger.info("Modèle créé avec l'ID : %s", model_id)

        self._start_compute(model_id)

    @staticmethod
    def _read_dataset_id() -> str:
        """Relit l'ID produit par l'étape d'extraction."""
        if not settings.DATASET_ID_FILE.exists():
            raise FileNotFoundError("ID dataset manquant. Lancez l'extraction.")
        return settings.DATASET_ID_FILE.read_text().strip()

    def _resolve_project_id(self) -> str:
        """Résout l'ID du projet Ultralytics cible."""
        response = requests.get(f"{self.api_url}/projects", headers=self.headers)
        response.raise_for_status()

        project = next(
            (
                p
                for p in response.json().get("projects", [])
                if p.get("slug") == settings.ULTRALYTICS_PROJECT
                and p.get("username") == settings.ULTRALYTICS_USERNAME
            ),
            None,
        )
        if not project:
            raise ValueError(f"Projet {settings.ULTRALYTICS_PROJECT} introuvable.")
        return project.get("_id") or project.get("id")

    def _train_args(self) -> dict:
        """Hyper-paramètres et augmentations partagés (source : config)."""
        return {
            "model": settings.MODEL_VARIANT,
            "epochs": settings.EPOCHS,
            "patience": settings.PATIENCE,
            "imgsz": settings.IMG_SIZE,
            "degrees": settings.AUG_DEGREES,
            "hsv_v": settings.AUG_HSV_V,
            "mosaic": settings.AUG_MOSAIC,
            "mixup": settings.AUG_MIXUP,
            "fliplr": settings.AUG_FLIPLR,
        }

    def _create_model(self, dataset_id: str, project_id: str) -> str:
        """Crée l'entité modèle sur la plateforme et renvoie son ID."""
        logger.info("Création de l'entité Modèle sur la plateforme...")
        payload = {
            "name": settings.ULTRALYTICS_EXP_NAME,
            "datasetId": dataset_id,
            "projectId": project_id,
            "task": "detect",
            "method": "cloud",
            "cfg": {**self._train_args(), "task": "detect", "mode": "train"},
        }
        response = requests.post(
            f"{self.api_url}/models", headers=self.headers, json=payload
        )
        response.raise_for_status()

        data = response.json()
        model_id = (
            data.get("modelId") or data.get("id") or data.get("data", {}).get("id")
        )
        if not model_id:
            raise ValueError(f"ID du modèle introuvable. Réponse : {data}")
        return model_id

    def _start_compute(self, model_id: str) -> None:
        """Déclenche l'entraînement GPU pour le modèle créé."""
        logger.info("Démarrage de l'infrastructure Cloud (GPU)...")
        dataset_uri = (
            f"ul://{settings.ULTRALYTICS_USERNAME}/"
            f"datasets/{settings.ULTRALYTICS_DATASET}"
        )
        payload = {
            "modelId": model_id,
            "trainArgs": {**self._train_args(), "data": dataset_uri},
        }
        response = requests.post(
            f"{self.api_url}/training/start", headers=self.headers, json=payload
        )

        if response.status_code in (200, 201, 202):
            logger.info("Entraînement Cloud démarré avec succès.")
            logger.info("Suivi : https://platform.ultralytics.com/models/%s", model_id)
        else:
            raise RuntimeError(
                f"Échec du Compute Cloud (HTTP {response.status_code}) : "
                f"{response.text}"
            )


def run() -> None:
    """Point d'entrée de l'étape de training."""
    try:
        CloudTrainer().train()
    except Exception as exc:
        logger.error("Échec de l'entraînement : %s", exc)
        sys.exit(1)
